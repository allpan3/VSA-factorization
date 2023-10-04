import torch
from vsa import VSA, Resonator
from colorama import Fore
import os.path
from tqdm import tqdm
import json
from config import *
import time
import csv
from dataset import VSADataset
from torch.utils.data import DataLoader
from typing import List

def name_fd(m):
    return '-' + str(FOLD_DIM) + 'fd' if m=='HARDWARE' else ''

def name_v(num_codevectors):
    if type(num_codevectors) == int:
        return f"{num_codevectors}v"
    else:
        s = ""
        for i in range(len(num_codevectors)):
            s += f"{num_codevectors[i]},"
        return s[:-1] + "v"

def name_q(quantize):
    return "quantized" if quantize else "expanded"

def name_act(act):
    return act.lower()

def name_argmax(abs):
    return "argmax_abs" if abs else "argmax"

def name_res(res):
    return res[0:3].lower()

def name_obj(n):
    if type(n) == int:
        return f"{n}obj"
    else:
        return f"{tuple(n)}obj"

def collate_fn(batch):
    samples = torch.stack([x[0] for x in batch], dim=0)
    labels = [x[1] for x in batch]
    return samples, labels

def get_similarity(v1, v2, quantized):
    """
    Return the hamming similarity.
    Always compare the similarity between quantized vectors. If inputs are not quantized, quantize them first.
    Hamming similarity is linear and should reflect the noise level
    """
    if not quantized:
        if VSA_MODE == "SOFTWARE":
            # Compare the quantized vectors
            positive = torch.tensor(1, device=v1.device)
            negative = torch.tensor(-1, device=v1.device)
            v1 = torch.where(v1 >= 0, positive, negative)
            v2 = torch.where(v2 >= 0, positive, negative)
        else:
            positive = torch.tensor(1, device=v1.device)
            negative = torch.tensor(0, device=v1.device)
            v1 = torch.where(v1 >= 0, positive, negative)
            v2 = torch.where(v2 >= 0, positive, negative)

    return torch.sum(torch.where(v1 == v2, 1, 0), dim=-1) / DIM
            

def algo1(vsa, rn, inputs, init_estimates, codebooks, orig_indices, quantized):
    """
    Explain away every extracted vector by subtracting it from the input, followed by another round of factorization.
    The vector is only subtracted if it's similar enough to the original input. Note this relies on stochasticity to generate
    a different outcome in the next trial, otherwise the same vector will be extracted again and again.
    In total, MAX_TRIALS rounds are performed, unless the energy left in the residual input is too low, in which case the process stops early.
    When the vector count is known, we rank the extracted candidates by their similarity to the input and only keep the top n vectors.
    When the count is unknown, we only keep the candidates whose similarity to the input (after quantization) are above a certain threshold.
    This method only works with unquantized inputs, and must quantize the input before running the resonator

    Notice we have two variations with regard to detecting whether an answer is similar enough to the input: one is to compare with the original
    vector, the other is to compare with the remaining vector after explaining away in each iteration. The first method makes more sense in the
    first glence because that should be the true confidence of a particular answer. But we sometimes see the case that a correct answer falls below
    the threshold when compared to the original input, especially when the input contains noise. This causes the particualr answer not getting subtracted
    from the input. When the count is known, this may not cause huge issue as we will rank all answers in the end (it will just take more trials in order
    to hopefully extract all vectors). However, when the count is unknown, this causes trouble because the count is solely determined by the similarity comparison.
    If we use the second method, the supposedly correct answer will be more similar to the remaining input and more likely be above the threshold. Even if it
    is still below the treshold and not get explained away, it will still have the chance to get extracted again later, at which time it's more likely that it's
    even more similar to the remaining input. 
    Note this solution works well because we are confident that every vector we explain away are truly correct answer (so that the similarity to the remaining input
    gets higher each time the same vector gets extracted), which is in turn consolidated by enforcing the similarity threshold for explain-away.

    There are also three ways to select candidates for final ranking/filtering. The first way is to add every extracted vector to the candidate list, regardless of the
    similarity. The second way is only to add the vector if it's similar enough to the original input. The third way is the same, but compared to the remaining input.
    The main advantage of second/third method compared to the first is that it reduces the chance which the same vector is added multiple times to the list (assume it
    only appears once in the true label). This can happen when a candidate is above detection threshold but below explain-away threshold, so it's not subtracted and 
    will likely be extracted again (this is most likely because different thresholds/similarities are used for explain-away and detection). However, if we know there are
    no duplicated vectors in the true label, we can do post-processing by removing duplicated vectors.
    The second/third method usually turns ou to be better

    Sometimes, there can be a case where a uncontained vector is very similar to the remaining input after some stages of explain-away, and all prior extracted vectors
    are correct. This is typically because the input vector is corrupted by noise, but it can even happen when no noise is applied, which means the current dimensionality
    is not high enough to perfectly support the number of vectors we are trying to superpose - bundled vectors after quantization are no longer orthogonal to a uncontained bound vector.
    A potential soluition is to adopt a hybrid method - explain away with the remaining input but to select final outcomes using the similarity to the original input.
    I've seen mixed result with this method - it doesn't always lead to better. Dimensionality is still the key cure.
    (Different thresholds should be employed; similarity to original input is typically lower as it contains more superposed vectors.)
    (Sometimes reverting back to the first method can also improve the accuracy)

    Another observation is that when the bundled input is superposed by the same vector multiple times, the result is pushed toward that majority
    vector and the minority vector ends up being almost orthogonal to the compositional vector, so the similarity is very low. This issue is also
    alleviated by the second method because we now expect to compare the minority vector with the remaining input after subtracting the majority vector
    (either some of all of the instances), and the similarity will be high enough for it to be considered valid.
    """

    _inputs = inputs.clone()
    inputs_q = VSA.quantize(inputs)
    init_estimates = init_estimates.clone()

    outcomes = [[] for _ in range(inputs.size(0))]  # num of batches
    iters = [[] for _ in range(inputs.size(0))]
    unconverged = [0] * inputs.size(0)
    sim_to_orig = [[] for _ in range(inputs.size(0))]
    sim_to_remain = [[] for _ in range(inputs.size(0))]
    debug_message = ""

    assert(not quantized)

    for _ in range(MAX_TRIALS):
        inputs_ = VSA.quantize(_inputs)
        # Randomizing initial estimates do not seem to be critical
        # if vsa.mode == "HARDWARE":
        #     # This is one way to randomize the initial estimates
        #     init_estimates = vsa.ca90(init_estimates)
        # elif vsa.mode == "SOFTWARE":
        #     # Replace this with true random vector
        #     init_estimates = vsa.apply_noise(init_estimates, 0.5)

        outcome, iter, converge = rn(inputs_, init_estimates, codebooks, orig_indices) 

        # Split batch results
        for i in range(len(outcome)):
            unconverged[i] += 1 if converge == False else 0
            iters[i].append(iter)
            # Get the compositional vector and subtract it from the input
            vector = vsa.get_vector(outcome[i])
            sim_orig = vsa.dot_similarity(inputs_q[i], vector)
            sim_remain = vsa.dot_similarity(inputs_[i], vector)
            # Only explain away the vector if it's similar enough to the input
            # Also only consider it as the final candidate if so
            explained = "NOT EXPLAINED"
            if sim_remain >= int(vsa.dim * SIM_EXPLAIN_THRESHOLD):
                sim_to_orig[i].append(sim_orig)
                sim_to_remain[i].append(sim_remain)
                outcomes[i].append(outcome[i])
                _inputs[i] = _inputs[i] - VSA.expand(vector)
                explained = "EXPLAINED"
        
            debug_message += f"DEBUG: outcome = {outcome[i]}, sim_orig = {round(sim_orig.item()/DIM, 3)}, sim_remain = {round(sim_remain.item()/DIM, 3)}, energy_left = {round(VSA.energy(_inputs[i]).item()/DIM,3)}, {converge}, {explained}\n"
        # If energy left in the input is too low, likely no more vectors to be extracted and stop
        # When inputs are batched, must wait until all inputs are exhausted
        if (all(VSA.energy(_inputs) <= int(vsa.dim * ENERGY_THRESHOLD))):
            break

    if COUNT_KNOWN: 
        # Among all the outcomes, select the n cloests to the input
        # Split batch results
        for i in range(len(inputs)):
            debug_message += f"DEBUG: pre-ranked{outcomes[i]}\n"
            # It's possible that none of the vectors extracted are similar enough to be considered as condidates
            if len(outcomes[i]) != 0:
                # Ranking by similarity to the original input makes more sense
                sim_to_orig[i], outcomes[i] = list(zip(*sorted(zip(sim_to_orig[i], outcomes[i]), key=lambda k: k[0], reverse=True)))
            # Only keep the top n
            outcomes[i] = outcomes[i][0:NUM_VEC_SUPERPOSED]
    else:
        # Split batch results
        for i in range(len(inputs)):
            debug_message += f"DEBUG: pre-filtered: {outcomes[i]}\n"
            outcomes[i] = [outcomes[i][j] for j in range(len(outcomes[i])) if sim_to_orig[i][j] >= int(vsa.dim * SIM_DETECT_THRESHOLD)]
    
    counts = [len(outcomes[i]) for i in range(len(outcomes))]

    return outcomes, unconverged, iters, counts, debug_message

def algo2(vsa, rn, inputs, d, f, codebooks, orig_indices, quantize):
    """
    Run the resonator a predefined number of times with different, randomly generated initial estimates.
    Reduce the final results to remove the duplicated vectors. The remainder are expected to contain all
    superposed vectors and should not contain any non-existing vectors.
    """
    inputs = inputs.clone()

    outcomes = [set() for _ in range(inputs.size(0))]  # num of batches
    iters = [[] for _ in range(inputs.size(0))]
    unconverged = [0] * inputs.size(0)

    if quantize:
        inputs = VSA.quantize(inputs)

    # TODO maybe we can throw away cases that don't converge and try again (still up to TRAILS times)
    # because if it doesn't converge, it's most likely not a valid vector, and will stop the rn early since
    # we've reached the number of objects. It didnt converge most likely due to a bad initial estimate.
    for _ in range(MAX_TRIALS):
        # Pure random
        init_estimates = vsa.random((f, d)).repeat(inputs.size(0), 1, 1)

        outcome, iter, converge = rn(inputs, init_estimates, codebooks, orig_indices)

        # Split batch results
        for i in range(len(outcome)):
            outcomes[i].add(outcome[i])
            unconverged[i] += 1 if converge == False else 0
            iters[i].append(iter) 
        # As soon as the required number of vectors are extracted, stop
        if all([len(outcomes[i]) == NUM_VEC_SUPERPOSED for i in range(len(outcomes))]):
            break
        
    # TODO add support for COUNT_KNOWN
    counts = [len(outcomes[i]) for i in range(len(outcomes))]
    return outcomes, unconverged, iters, counts


def algo3(vsa, rn, inputs, quantize):
    """
    We want to combine multiple (compositional) vectors into one superposed vector and be able to extract the original component
    vectors and their individual factors.
    In this algorithm we add one more factor, ID, on top of the original factors composing the component vector. We bind each
    vector with an ID vector before bundling them.
    The key question is how to decide which ID to bind with which vector. The naive way would just be binding ID0 with the first 
    vector in the label list and so on, but this would cause the same vector to be bound with different IDs when it appears in 
    different positions/indices in the label list. This is fine and the resonator network will still work, but it messes with the
    but it messes with the perception frontend, which is often done using neural networks, because the "ID" is not associated
    with the concept represented by the superposed vector. The same component vector can be bound with different IDs and produce
    completely different vectors if it's located in a different position in the label list, and the final, superposed vector can 
    also be different. This means multiple groudtruth vectors can be assoicated with each concept and will cause confusion in the
    perception frontend. Imagine a concept composed of component vectors A, B and C is different from one composed of A, C and B,
    with the order being the only difference.

    In this algorithm, the ID being bound with each vector in the set is determined by a predefined priority list. Here we construct
    the priority list by ordering the first two factors. Lower IDs (lower indices in the ID codebook) are assigned to vectors with
    higher priority. These factors are features of the concept (e.g. the position x, y of an object in an image) and must be unique
    across all components vectors (e.g. each position in the image can only be occupied by one object). If the second condition is
    not met, there's still a chance that different IDs are assigned to the same component vector when it's combined (unless we define
    a tiebreaker, which essentially expands the priority list), but this often represents the same confusion that the perception fronend
    would also face (imagine two objects are overlapped in the same position, the perception frontend will also have trouble 
    distinguishing them).

    With this method, for a given concept to encode, no matter how the data was generated, the same component vector is always bound
    with the same ID, so a unique vector will be produced for that concept as the label for the perception frontend.
    Note, in different contexts (when superposed with a different set of other vectors), the same component vector may gets assigned
    to a differnet ID because it is determined by the relative priority of that component in the context. But this is fine because 
    all we care is to produce unique vectors/labels for the same concept.
    """
    inputs = inputs.clone()

    codebooks = vsa.codebooks[0:-1]
    try:
        codebooks = torch.stack(codebooks)
    except:
        pass
    orig_indices = None
    id_cb = vsa.codebooks[-1]

    # Reordering and init estiamtes are the same for every sample, but we have to put it here since we must remove the ID codebook first
    if REORDER_CODEBOOKS:
        codebooks, orig_indices = rn.reorder_codebooks(codebooks)

    init_estimates = rn.get_init_estimates(codebooks).unsqueeze(0).repeat(inputs.size(0),1,1)

    if quantize:
        inputs = VSA.quantize(inputs)

    outcomes = [[] for _ in range(inputs.size(0))]  # num of batches
    unconverged = [0] * inputs.size(0)
    iters = [[] for _ in range(inputs.size(0))]

    for k in range(MAX_TRIALS):
        # Unbind the ID
        inputs_ = vsa.bind(inputs, id_cb[k])

        outcome, iter, converge = rn(inputs_, init_estimates, codebooks, orig_indices)

        # Split batch results
        for i in range(len(outcome)):
            outcomes[i].append(outcome[i])
            unconverged[i] += 1 if converge == False else 0
            iters[i].append(iter) 

    # TODO add support for COUNT_KNOWN
    counts = [len(outcomes[i]) for i in range(len(outcomes))]
    return outcomes, unconverged, iters, counts

def algo4(vsa, rn, inputs, init_estimates, codebooks, orig_indices, quantized):
    """
    This algorithm differs from algo1 in that here we run multiple trials of the resonator network in parallel and choose the outcome
    with the highest similarity to the input as the outcome of the stage and subtract from the unquantized input, if above the similarity
    threshold.
    Note the way I implemented algo1 and algo4 are different from the paper. I subtract the vector only if it's above a threshold. With
    this prior, this method hasn't really shown any advantage over algo1; the difference is very minimal, and the execution time is longer.
    """
    _inputs = inputs.clone()
    inputs_q = VSA.quantize(inputs)

    outcomes = [[] for _ in range(inputs.size(0))]  # num of batches
    iters = [[] for _ in range(inputs.size(0))]
    unconverged = [0] * inputs.size(0)
    similarities = [[] for l in range(BATCH_SIZE)]

    assert(not quantized)

    for _ in range(MAX_TRIALS):
        inputs_ = VSA.quantize(_inputs)

        outcome = [[] for l in range(BATCH_SIZE)]
        iter = [0] * PARALLEL_TRIALS
        converge = [0] * PARALLEL_TRIALS

        # TODO: need to merge this into one tensor and run in parallel
        for l in range(PARALLEL_TRIALS):
            if vsa.mode == "HARDWARE":
                init_estimates = vsa.ca90(init_estimates)
            elif vsa.mode == "SOFTWARE":
                # Replace this with true random vector
                init_estimates = vsa.apply_noise(init_estimates, 0.5)
            out, iter[l], converge[l] = rn(inputs_, init_estimates, codebooks, orig_indices) 
            for i in range(BATCH_SIZE):
                outcome[i].append(out[i])

        # Split batch results
        for i in range(len(outcome)):
            # Select the one with highest similarity
            vectors = torch.stack([vsa.get_vector(outcome[i][j]) for j in range(len(outcome[i]))]) 
            sim_remain = vsa.dot_similarity(inputs_[i], vectors)
            sim_orig = vsa.dot_similarity(inputs_q[i], vectors)
            best_idx = torch.max(sim_remain, dim=-1)[1].item()
            # best_idx = torch.max(sim_orig, dim=-1)[1].item()
            # print("sim_remain: ", sim_remain.tolist())
            # print("sim_orig: ", sim_orig.tolist())
            # print(outcome[i])
            outcomes[i].append(outcome[i][best_idx])
            unconverged[i] += sum([converge[i] == False for i in range(len(converge))])
            iters[i].append(sum(iter))
            # similarities[i].append(sim_remain[best_idx])
            similarities[i].append(sim_orig[best_idx])
            # # Only subtract the vector if it's similar enough to the input
            if sim_remain[best_idx] >= int(vsa.dim * SIM_EXPLAIN_THRESHOLD):
                _inputs[i] = _inputs[i] - VSA.expand(vectors[best_idx])
        
        # If energy left in the input is too low, likely no more vectors to be extracted and stop
        # When inputs are batched, must wait until all inputs are exhausted
        # print(f"Energy = {VSA.energy(_inputs).item()}, {converge}")
        if (all(VSA.energy(_inputs) <= int(vsa.dim * ENERGY_THRESHOLD))):
            break

    if COUNT_KNOWN: 
        # Among all the outcomes, select the n cloests to the input
        # Split batch results
        for i in range(len(inputs)):
            # print(outcomes[i])
            similarities[i], outcomes[i] = list(zip(*sorted(zip(similarities[i], outcomes[i]), key=lambda k: k[0], reverse=True)))
            # Only keep the top n
            outcomes[i] = outcomes[i][0:NUM_VEC_SUPERPOSED]
    else:
        # Split batch results
        for i in range(len(inputs)):
            # print(outcomes[i])
            outcomes[i] = [outcomes[i][j] for j in range(len(outcomes[i])) if similarities[i][j] >= int(vsa.dim * SIM_DETECT_THRESHOLD)]

    counts = [len(outcomes[i]) for i in range(len(outcomes))]

    return outcomes, unconverged, iters, counts


def run_factorization(
        m = VSA_MODE,
        d = DIM,
        f = FACTORS,
        v = CODEVECTORS,
        n = NOISE_LEVEL,
        it = ITERATIONS,
        res = RESONATOR_TYPE,
        q = QUANTIZE,
        act = ACTIVATION,
        abs = ARGMAX_ABS,
        device = "cpu",
        verbose = 0):

    test_dir = f"tests/{m}-{d}d-{f}f-{name_v(v)}{name_fd(m)}"

    # Checkpoint
    cp = os.path.join(test_dir, f"{m}-{d}d-{f}f-{name_v(v)}-{n}n-{name_res(res)}-{it}i-{name_q(q)}-{name_act(act)}-{name_argmax(abs)}-{name_obj(NUM_VEC_SUPERPOSED)}-{NUM_SAMPLES}s.checkpoint")
    if CHECKPOINT and os.path.exists(cp):
        if verbose >= 1:
            print(Fore.LIGHTYELLOW_EX + f"Test with {(m, d, f, v, n, name_res(res), it, name_q(q), name_act(act), (name_argmax(abs)), {name_obj(NUM_VEC_SUPERPOSED)}, NUM_SAMPLES)} already exists, skipping..." + Fore.RESET)
        return

    vsa = VSA(
        root=test_dir,
        mode=m,
        dim=d,
        num_factors=f,
        num_codevectors=v,
        fold_dim = FOLD_DIM,
        ehd_bits = EHD_BITS,
        sim_bits = SIM_BITS,
        seed = SEED,
        device=device
    )

    # Generate test samples
    ds = VSADataset(test_dir, NUM_SAMPLES, vsa, algo=ALGO, num_vectors_superposed=NUM_VEC_SUPERPOSED, quantize=q, noise=n)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    rn = Resonator(vsa, m, type=res, activation=act, iterations=it, argmax_abs=abs, act_val=ACT_VALUE, stoch=STOCHASTICITY, randomness=RANDOMNESS, early_converge=EARLY_CONVERGE, seed=SEED, device=device)

    codebooks = None
    orig_indices = None

    if REORDER_CODEBOOKS:
        codebooks, orig_indices = rn.reorder_codebooks(codebooks)

    #TODO probably can make this a class variable
    init_estimates = rn.get_init_estimates(codebooks).unsqueeze(0).repeat(BATCH_SIZE,1,1)

    incorrect_count = 0
    unconverged = [0, 0] # Unconverged successful, unconverged failed
    total_iters = 0
    debug_message = ""
    j = 0
    for data, labels in tqdm(dl, desc=f"Progress", leave=True if verbose >= 1 else False):
        # For single-vector factorization, directly run resonator network
        if (type(NUM_VEC_SUPERPOSED) == int and NUM_VEC_SUPERPOSED == 1):
            outcome, iter, converge = rn(data, init_estimates, codebooks, orig_indices)
            # Make them the same format as multi-vector for easier parsing
            outcomes = [[outcome[x]] for x in range(BATCH_SIZE)]
            convergences = [1 if converge == False else 0] * BATCH_SIZE
            iters = [[iter]] * BATCH_SIZE
            counts = [1] * BATCH_SIZE
        else:
            if ALGO == "ALGO1":
                outcomes, convergences, iters, counts, debug_message = algo1(vsa, rn, data, init_estimates, codebooks, orig_indices, q)
            elif ALGO == "ALGO2":
                outcomes, convergences, iters, counts = algo2(vsa, rn, data, d, f, codebooks, orig_indices, q)
            elif ALGO == "ALGO3":
                outcomes, convergences, iters, counts = algo3(vsa, rn, data, q)
            elif ALGO == "ALGO4":
                outcomes, convergences, iters, counts = algo4(vsa, rn, data, init_estimates, codebooks, orig_indices, q)

        ## Analyze results
        # Batch
        for k in range(len(labels)):
            message = ""
            incorrect = False
            label = labels[k]
            outcome = outcomes[k]
            iter = iters[k]
            convergence = convergences[k]
            count = counts[k]
            sim_per_vec = []

            # Calculate the similarity between the input compositional vector and the groundtruth
            # This is to verify the effectiveness of noise
            if NUM_VEC_SUPERPOSED != 1 and ALGO == "ALGO3":
                # Get the correctly bound (with ID) groundtruth vectors 
                gt_vecs = ds.lookup_algo3(label, bundled=False)
                similarity = round(get_similarity(ds.lookup_algo3(label), data[k], q).item(), 3)
            else:
                similarity = round(get_similarity(data[k], vsa.get_vector(label, q), q).item(), 3)

            total_iters += sum(iter)

            if (count != len(label)):
                incorrect = True
                message += Fore.RED + "Incorrect number of vectors detected, got {}, expected {}".format(count, len(label)) + Fore.RESET + "\n"
            else:
                message += f"Correct number of vectors detected: {count} \n"

            # Multiple vectors superposed
            for i in range(len(label)):
                if (label[i] not in outcome):
                    incorrect = True
                    message += Fore.RED + "Vector {} is not detected.".format(label[i]) + Fore.RESET + "\n"
                else:
                    message += "Vector {} is correctly detected.".format(label[i]) + "\n" 

                v = data[k] if q else VSA.quantize(data[k])
                # Per vector similarity to the input compound vector. Always compare the quantized vectors
                if ALGO == "ALGO3":
                    sim_per_vec.append(round(get_similarity(gt_vecs[i], v, True).item(), 3))
                else:
                    sim_per_vec.append(round(get_similarity(vsa.get_vector(label[i]), v, True).item(), 3))
            
            # Print results
            if incorrect:
                incorrect_count += 1
                unconverged[1] += convergence
                if verbose >= 2:
                    print(Fore.BLUE + f"Test {j} Failed" + Fore.RESET)
                    print("Input similarity = {}".format(similarity))
                    print("Per-vector similarity = {}".format(sim_per_vec))
                    print(f"unconverged: {convergence}")
                    print(f"iterations: {iter}")
                    print(message[:-1])
                    print(f"Outcome = {outcome}")
                    print(debug_message)
            else:
                unconverged[0] += convergence
                if verbose >= 3:
                    print(Fore.BLUE + f"Test {j} Passed" + Fore.RESET)
                    print("Input similarity = {}".format(similarity))
                    print("Per-vector similarity = {}".format(sim_per_vec))
                    print(f"unconverged: {convergence}")
                    print(f"iterations: {iter}")
                    print(message[:-1])
                    print(debug_message)
            j += 1

    accuracy = (NUM_SAMPLES - incorrect_count) / NUM_SAMPLES
    if verbose >= 1:
        print(f"Accuracy: {accuracy}    Unconverged: {unconverged}    Average iterations: {total_iters / NUM_SAMPLES}")

    # Checkpoint
    with open(cp, "w") as fp:
        pass

    return accuracy, unconverged, (m, d, f, v, n, res, it, q, act, abs, NUM_VEC_SUPERPOSED), (accuracy, unconverged)

# TODO haven't maintained these tests for a while, need to check
# Test various dimensions, factors, and codevectors
# def test_dim_fac_vec(device="cpu", verbose=0):
#     print(Fore.CYAN + f"Test Setup: mode = {VSA_MODE}, quantize = {QUANTIZE}, activation = {ACTIVATION}, argmax_abs = {ARGMAX_ABS}, noise = {NOISE_LEVEL}, resonator = {RESONATOR_TYPE}, iterations = {ITERATIONS}, superposed = {NUM_VEC_SUPERPOSED}, samples = {NUM_SAMPLES}" + Fore.RESET)

#     csv_file = f'tests/table-{VSA_MODE}-{NOISE_LEVEL}n-{name_res(RESONATOR_TYPE)}-{ITERATIONS}i-{name_q(QUANTIZE)}-{name_act(ACTIVATION)}-{name_argmax(ARGMAX_ABS)}-{NUM_VEC_SUPERPOSED}s.csv'

#     with open(csv_file, mode='w') as c:
#         writer = csv.DictWriter(c, fieldnames=FIELDS)
#         writer.writeheader()
#         for d in DIM_RANGE:
#             skip_rest_f = False
#             for f in FACTOR_RANGE:
#                 if not skip_rest_f:
#                     skip_rest_v = False 
#                 for v in CODEVECTOR_RANGE:
#                     if skip_rest_v:
#                         if verbose >= 1:
#                             print(Fore.YELLOW + f"Skipping {d}d-{f}f-{name_v(v)}" + Fore.RESET)
#                         continue

#                     if verbose >= 1:
#                         print(Fore.BLUE + f"Running test with {d} dimensions, {f} factors, {v} codevectors" + Fore.RESET)

#                     ret = run_factorization(d=d, f=f, v=v, device=device, verbose=verbose)
#                     if ret is None:
#                         continue
#                     accuracy, _, key, val = ret

#                     # If accuracy is less than 30%, skip the rest of the tests
#                     if accuracy <= 0.3:
#                         skip_rest_v = True
#                         # If the first stage fails, skip the rest of f
#                         if v == CODEVECTOR_RANGE[0]:
#                             skip_rest_f = True
                    
#                     writer.writerow({FIELDS[0]:key[0],FIELDS[1]:key[1],FIELDS[2]:key[2],FIELDS[3]:key[3],FIELDS[4]:key[4],FIELDS[5]:key[5],FIELDS[6]:key[6],FIELDS[7]:key[7],FIELDS[8]:key[8],FIELDS[9]:key[9],FIELDS[10]:key[10],FIELDS[11]:val[0],FIELDS[12]:val[1][0],FIELDS[13]:val[1][1],FIELDS[14]:NUM_SAMPLES})

#         print(Fore.GREEN + f"Saved table to {csv_file}" + Fore.RESET)

# def test_noise_iter(device="cpu", verbose=0):
#     print(Fore.CYAN + f"Test Setup: mode = {VSA_MODE}, dim = {DIM}, factors = {FACTORS}, codevectors = {CODEVECTORS}, resonator = {RESONATOR_TYPE}, quantize = {QUANTIZE}, activation = {ACTIVATION}, argmax_abs = {ARGMAX_ABS}, superposed = {NUM_VEC_SUPERPOSED}, samples = {NUM_SAMPLES}" + Fore.RESET)

#     csv_file = f'tests/table-{VSA_MODE}-{DIM}d-{FACTORS}f-{name_v(CODEVECTORS)}-{name_res(RESONATOR_TYPE)}-{name_q(QUANTIZE)}-{name_act(ACTIVATION)}-{name_argmax(ARGMAX_ABS)}-{NUM_VEC_SUPERPOSED}s.csv'

#     with open(csv_file, mode='w') as c:
#         writer = csv.DictWriter(c, fieldnames=FIELDS)
#         writer.writeheader()
#         skip_rest_n = False
#         for n in NOISE_RANGE:
#             if not skip_rest_n:
#                 skip_rest_i = False
#             for it in ITERATION_RANGE:
#                 if skip_rest_i:
#                     if verbose >= 1:
#                         print(Fore.YELLOW + f"Skipping {n}n-{it}i" + Fore.RESET)
#                     continue
                
#                 if verbose >= 1:
#                     print(Fore.BLUE + f"Running test with noise = {n}, iterations = {it}" + Fore.RESET)

#                 ret = run_factorization(n=n, it=it, device=device, verbose=verbose)
#                 if ret is None:
#                     continue
#                 accuracy, unconverged, key, val = ret

#                 # If no incorrect answer is unconverted, more iterations don't matter
#                 if (unconverged[1]) == 0:
#                     skip_rest_i = True
#                 # If accuracy is less than 10% for current noise level and more iterations don't have, skip
#                 if accuracy <= 0.1 and skip_rest_i:
#                     skip_rest_n = True

#                 writer.writerow({FIELDS[0]:key[0],FIELDS[1]:key[1],FIELDS[2]:key[2],FIELDS[3]:key[3],FIELDS[4]:key[4],FIELDS[5]:key[5],FIELDS[6]:key[6],FIELDS[7]:key[7],FIELDS[8]:key[8],FIELDS[9]:key[9],FIELDS[10]:key[10],FIELDS[11]:val[0],FIELDS[12]:val[1][0],FIELDS[13]:val[1][1],FIELDS[14]:NUM_SAMPLES})

#     print(Fore.GREEN + f"Saved table to {csv_file}" + Fore.RESET)


# def test_norm_act_res(device="cpu", verbose=0):
#     print(Fore.CYAN + f"Test Setup: mode = {VSA_MODE}, dim = {DIM}, factors = {FACTORS}, codevectors = {CODEVECTORS}, noise = {NOISE_LEVEL}, iterations = {ITERATIONS}, argmax_abs = {ARGMAX_ABS}, superposed = {NUM_VEC_SUPERPOSED}, samples = {NUM_SAMPLES}" + Fore.RESET)

#     csv_file = f'tests/table-{VSA_MODE}-{DIM}d-{FACTORS}f-{name_v(CODEVECTORS)}-{NOISE_LEVEL}n-{ITERATIONS}i-{name_argmax(ARGMAX_ABS)}-{NUM_VEC_SUPERPOSED}s.csv'

#     with open(csv_file, mode='w') as c:
#         writer = csv.DictWriter(c, fieldnames=FIELDS)
#         writer.writeheader()
#         for r in RESONATOR_TYPE_RANGE:
#             skip_rest_n = False
#             for n in QUANTIZE_RANGE:
#                 for a in ACTIVATION_RANGE:
#                     print(Fore.BLUE + f"Running test with resonator = {r}, quantize = {n}, activation = {a}" + Fore.RESET)
#                     ret = run_factorization(res=r, q=n, act=a, device=device, verbose=verbose)
#                     if ret is None:
#                         continue
#                     _, _, key, val = ret

#                     writer.writerow({FIELDS[0]:key[0],FIELDS[1]:key[1],FIELDS[2]:key[2],FIELDS[3]:key[3],FIELDS[4]:key[4],FIELDS[5]:key[5],FIELDS[6]:key[6],FIELDS[7]:key[7],FIELDS[8]:key[8],FIELDS[9]:key[9],FIELDS[10]:key[10],FIELDS[11]:val[0],FIELDS[12]:val[1][0],FIELDS[13]:val[1][1],FIELDS[14]:NUM_SAMPLES})

#     print(Fore.GREEN + f"Saved table to {csv_file}" + Fore.RESET)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # torch.set_default_device(device)

    os.makedirs("tests", exist_ok=True)

    print(f"Running tests on {device}, batch_size = {BATCH_SIZE}")

    start = time.time()
    if RUN_MODE == "single":
        print(Fore.CYAN + f"""
Test Setup: mode = {VSA_MODE}, dim = {DIM}, factors = {FACTORS}, codevectors = {CODEVECTORS}, \
n_superposed = {NUM_VEC_SUPERPOSED}, algo = {ALGO}, max_trials = {MAX_TRIALS}, energy_thresh = {ENERGY_THRESHOLD} \
similarity_explain_thresh = {SIM_EXPLAIN_THRESHOLD}, similarity_detect_thresh = {SIM_DETECT_THRESHOLD}, \
expanded_hd_bits = {EHD_BITS}, int_reg_bits = {SIM_BITS}, noise = {NOISE_LEVEL}, quantize = {QUANTIZE}, \
resonator = {RESONATOR_TYPE}, iterations = {ITERATIONS}, stochasticity = {STOCHASTICITY}, randomness = {RANDOMNESS}, \
activation = {ACTIVATION}, act_val = {ACT_VALUE}, early_converge_thresh = {EARLY_CONVERGE}, argmax_abs = {ARGMAX_ABS}, \
samples = {NUM_SAMPLES}
""" + Fore.RESET)
        run_factorization(device=device, verbose=VERBOSE)
    # elif RUN_MODE == "dim-fac-vec":
    #     test_dim_fac_vec(device=device, verbose=VERBOSE)
    # elif RUN_MODE == "noise-iter":
    #     test_noise_iter(device=device, verbose=VERBOSE)
    # elif RUN_MODE == "norm-act-res":
    #     test_norm_act_res(device=device, verbose=VERBOSE)

    end = time.time()
    print(f"Time elapsed: {end - start}s")
