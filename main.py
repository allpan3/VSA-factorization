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

def v_name(num_codevectors):
    if type(num_codevectors) == int:
        return f"{num_codevectors}v"
    else:
        s = ""
        for i in range(len(num_codevectors)):
            s += f"{num_codevectors[i]},"
        return s[:-1] + "v"

def norm_name(norm):
    return "norm" if norm else "no_norm"

def act_name(act):
    return "act_" + act.lower() if act != "NONE" else "no_act"

def argmax_name(abs):
    return "argmax_abs" if abs else "argmax"

def res_name(res):
    return res[0:3].lower()

def collate_fn(batch):
    samples = torch.stack([x[0] for x in batch], dim=0)
    labels = [x[1] for x in batch]
    return samples, labels

def get_similarity(v1, v2, quantized):
    """
    Return the hamming similarity for quantized vectors, and cosine similarity for unquantized vectors
    Hamming similarity is linear and should reflect the noise level
    Cosine similarity is non-linear and may not reflect the noise level
    """
    if quantized:
        if VSA_MODE == "SOFTWARE":
            # Compare the quantized vectors
            positive = torch.tensor(1, device=v1.device)
            negative = torch.tensor(-1, device=v1.device)
            v1_ = torch.where(v1 >= 0, positive, negative)
            v2_ = torch.where(v2 >= 0, positive, negative)
            return torch.sum(torch.where(v1_ == v2_, 1, 0), dim=-1) / DIM
        else:
            positive = torch.tensor(1, device=v1.device)
            negative = torch.tensor(0, device=v1.device)
            v1_ = torch.where(v1 >= 0, positive, negative)
            v2_ = torch.where(v2 >= 0, positive, negative)
            return torch.sum(torch.where(v1 == v2, 1, 0), dim=-1) / DIM
    else:
            v1_dot = torch.sum(v1 * v1, dim=-1)
            v1_mag = torch.sqrt(v1_dot)
            v2_dot = torch.sum(v2 * v2, dim=-1)
            v2_mag = torch.sqrt(v2_dot)
            magnitude = v1_mag * v2_mag
            magnitude = torch.clamp(magnitude, min=1e-08)
            return torch.matmul(v1.type(torch.float32), v2.type(torch.float32)) / magnitude
            

def algo1(vsa, rn, inputs, init_estimates, codebooks, orig_indices, quantized):
    """
    Subtract every extracted vector from the original input and repeat the process NUM_VEC_SUPERPOSED times 
    This method only works with unquantized inputs, and can choose to quantize the input before running the resonator
    (We do allow the setup where the input is not quantized just for experiment purpose)
    """

    inputs = inputs.clone()

    outcomes = [[] for _ in range(inputs.size(0))]  # num of batches
    iters = [[] for _ in range(inputs.size(0))]
    unconverged = [0] * inputs.size(0)


    for _ in range(TRIALS):
        # Input to resonator must be quantized, make sure don't do it again if it's already quantized
        if not quantized:
            inputs_ = VSA.quantize(inputs)
        else:
            inputs_ = inputs

        outcome, iter = rn(inputs_, init_estimates, codebooks, orig_indices) 

        # Split batch results
        for i in range(len(outcome)):
            outcomes[i].append(outcome[i])
            unconverged[i] += 1 if iter == ITERATIONS - 1 else 0
            iters[i].append(iter)
            # Get the compositional vector and subtract it from the input
            vector = vsa.get_vector(outcome[i], quantize=True)
            inputs[i] = inputs[i] - VSA.expand(vector)
    
    return outcomes, unconverged, iters

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
    for _ in range(TRIALS):
        # Pure random
        init_estimates = vsa.random(f, d).repeat(inputs.size(0), 1, 1)

        outcome, iter = rn(inputs, init_estimates, codebooks, orig_indices)

        # Split batch results
        for i in range(len(outcome)):
            outcomes[i].add(outcome[i])
            unconverged[i] += 1 if iter == ITERATIONS - 1 else 0
            iters[i].append(iter) 
        # As soon as the required number of vectors are extracted, stop
        if all([len(outcomes[i]) == NUM_VEC_SUPERPOSED for i in range(len(outcomes))]):
            break
        
    return outcomes, unconverged, iters


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
        inputs = inputs.quantize(inputs)

    outcomes = [[] for _ in range(inputs.size(0))]  # num of batches
    unconverged = [0] * inputs.size(0)
    iters = [[] for _ in range(inputs.size(0))]

    for k in range(TRIALS):
        # Unbind the ID
        inputs_ = vsa.bind(inputs, id_cb[k])

        outcome, iter = rn(inputs_, init_estimates, codebooks, orig_indices)

        # Split batch results
        for i in range(len(outcome)):
            outcomes[i].append(outcome[i])
            unconverged[i] += 1 if iter == ITERATIONS - 1 else 0
            iters[i].append(iter) 
    return outcomes, unconverged, iters


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

    test_dir = f"tests/{m}-{d}d-{f}f-{v_name(v)}"

    # Checkpoint
    cp = os.path.join(test_dir, f"{m}-{d}d-{f}f-{v_name(v)}-{n}n-{res_name(res)}-{it}i-{norm_name(q)}-{act_name(act)}-{argmax_name(abs)}-{NUM_VEC_SUPERPOSED}s-{NUM_SAMPLES}s.checkpoint")
    if CHECKPOINT and os.path.exists(cp):
        if verbose >= 1:
            print(Fore.LIGHTYELLOW_EX + f"Test with {(m, d, f, v, n, res_name(res), it, norm_name(q), act_name(act), (argmax_name(abs)), {NUM_VEC_SUPERPOSED}, NUM_SAMPLES)} already exists, skipping..." + Fore.RESET)
        return

    vsa = VSA(
        root=test_dir,
        mode=m,
        dim=d,
        num_factors=f,
        num_codevectors=v,
        seed = SEED,
        device=device
    )

    # Generate test samples
    ds = VSADataset(test_dir, NUM_SAMPLES, vsa, algo=ALGO, num_vectors_superposed=NUM_VEC_SUPERPOSED, quantize=q, noise=n)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    rn = Resonator(vsa, type=res, activation=act, iterations=it, argmax_abs=abs, lambd=LAMBD, stoch=STOCHASTICITY, early_converge=EARLY_CONVERGE, device=device)

    codebooks = None
    orig_indices = None

    if REORDER_CODEBOOKS:
        codebooks, orig_indices = rn.reorder_codebooks(codebooks)

    init_estimates = rn.get_init_estimates(codebooks).unsqueeze(0).repeat(BATCH_SIZE,1,1)

    incorrect_count = 0
    unconverged = [0, 0] # Unconverged successful, unconverged failed
    total_iters = 0
    j = 0
    for data, labels in tqdm(dl, desc=f"Progress", leave=True if verbose >= 1 else False):

        if (NUM_VEC_SUPERPOSED == 1):
            outcome, iter = rn(data, init_estimates, codebooks, orig_indices)
            # Make them the same format as multi-vector for easier parsing
            outcomes = [[outcome[x]] for x in range(BATCH_SIZE)]
            convergences = [1 if iter == ITERATIONS-1 else 0] * BATCH_SIZE
            iters = [[iter]] * BATCH_SIZE
        else:
            if ALGO == "ALGO1":
                outcomes, convergences, iters = algo1(vsa, rn, data, init_estimates, codebooks, orig_indices, q)
            elif ALGO == "ALGO2":
                outcomes, convergences, iters = algo2(vsa, rn, data, d, f, codebooks, orig_indices, q)
            elif ALGO == "ALGO3":
                outcomes, convergences, iters = algo3(vsa, rn, data, q)

        ## Analyze results
        # Batch
        for k in range(len(labels)):
            message = ""
            incorrect = False
            label = labels[k]
            outcome = outcomes[k]
            iter = iters[k]
            convergence = convergences[k]
            sim_per_vec = []

            # Calculate the similarity between the input compositional vector and the groundtruth
            # This is to verify the effectiveness of noise
            if NUM_VEC_SUPERPOSED > 1 and ALGO == "ALGO3":
                # Get the correctly bound (with ID) groundtruth vectors 
                gt_vecs = ds.lookup_algo3(label, bundled=False)
                similarity = round(get_similarity(ds.lookup_algo3(label), data[k], q).item(), 3)
            else:
                similarity = round(get_similarity(data[k], vsa.get_vector(label), q).item(), 3)

            # Multiple vectors superposed
            for i in range(len(label)):
                total_iters += iter[i]
                if (label[i] not in outcome):
                    incorrect = True
                    message += Fore.RED + "Vector {} is not detected.".format(label[i]) + Fore.RESET + "\n"
                else:
                    message += "Vector {} is correctly detected.".format(label[i]) + "\n" 

                if NUM_VEC_SUPERPOSED > 1:
                    # Per vector similarity.
                    if ALGO == "ALGO3":
                        sim_per_vec.append(round(get_similarity(gt_vecs[i], data[k], q).item(), 3))
                    else:
                        sim_per_vec.append(round(get_similarity(vsa.get_vector(label[i]), data[k], q).item(), 3))
            
            # Print results
            if incorrect:
                incorrect_count += 1
                unconverged[1] += convergence
                if verbose >= 2:
                    print(Fore.BLUE + f"Test {j} Failed" + Fore.RESET)
                    print("Input similarity = {}".format(similarity))
                    if NUM_VEC_SUPERPOSED > 1:
                        print("Per-vector similarity = {}".format(sim_per_vec))
                    print(f"unconverged: {convergence}")
                    print(f"iterations: {iter}")
                    print(message[:-1])
                    print(f"Outcome = {outcome}")
            else:
                unconverged[0] += convergence
                if verbose >= 3:
                    print(Fore.BLUE + f"Test {j} Passed" + Fore.RESET)
                    print("Input similarity = {}".format(similarity))
                    if NUM_VEC_SUPERPOSED > 1:
                        print("Per-vector similarity = {}".format(sim_per_vec))
                    print(f"unconverged: {convergence}")
                    print(f"iterations: {iter}")
                    print(message[:-1])
            j += 1

    accuracy = (NUM_SAMPLES - incorrect_count) / NUM_SAMPLES
    if verbose >= 1:
        print(f"Accuracy: {accuracy}    Unconverged: {unconverged}    Average iterations: {total_iters / NUM_SAMPLES}")

    # Checkpoint
    with open(cp, "w") as fp:
        pass

    return accuracy, unconverged, (m, d, f, v, n, res, it, q, act, abs, NUM_VEC_SUPERPOSED), (accuracy, unconverged)

# Test various dimensions, factors, and codevectors
def test_dim_fac_vec(device="cpu", verbose=0):
    print(Fore.CYAN + f"Test Setup: mode = {VSA_MODE}, quantize = {QUANTIZE}, activation = {ACTIVATION}, argmax_abs = {ARGMAX_ABS}, noise = {NOISE_LEVEL}, resonator = {RESONATOR_TYPE}, iterations = {ITERATIONS}, superposed = {NUM_VEC_SUPERPOSED}, samples = {NUM_SAMPLES}" + Fore.RESET)

    csv_file = f'tests/table-{VSA_MODE}-{NOISE_LEVEL}n-{res_name(RESONATOR_TYPE)}-{ITERATIONS}i-{norm_name(QUANTIZE)}-{act_name(ACTIVATION)}-{argmax_name(ARGMAX_ABS)}-{NUM_VEC_SUPERPOSED}s.csv'

    with open(csv_file, mode='w') as c:
        writer = csv.DictWriter(c, fieldnames=FIELDS)
        writer.writeheader()
        for d in DIM_RANGE:
            skip_rest_f = False
            for f in FACTOR_RANGE:
                if not skip_rest_f:
                    skip_rest_v = False 
                for v in CODEVECTOR_RANGE:
                    if skip_rest_v:
                        if verbose >= 1:
                            print(Fore.YELLOW + f"Skipping {d}d-{f}f-{v_name(v)}" + Fore.RESET)
                        continue

                    if verbose >= 1:
                        print(Fore.BLUE + f"Running test with {d} dimensions, {f} factors, {v} codevectors" + Fore.RESET)

                    ret = run_factorization(d=d, f=f, v=v, device=device, verbose=verbose)
                    if ret is None:
                        continue
                    accuracy, _, key, val = ret

                    # If accuracy is less than 30%, skip the rest of the tests
                    if accuracy <= 0.3:
                        skip_rest_v = True
                        # If the first stage fails, skip the rest of f
                        if v == CODEVECTOR_RANGE[0]:
                            skip_rest_f = True
                    
                    writer.writerow({FIELDS[0]:key[0],FIELDS[1]:key[1],FIELDS[2]:key[2],FIELDS[3]:key[3],FIELDS[4]:key[4],FIELDS[5]:key[5],FIELDS[6]:key[6],FIELDS[7]:key[7],FIELDS[8]:key[8],FIELDS[9]:key[9],FIELDS[10]:key[10],FIELDS[11]:val[0],FIELDS[12]:val[1][0],FIELDS[13]:val[1][1],FIELDS[14]:NUM_SAMPLES})

        print(Fore.GREEN + f"Saved table to {csv_file}" + Fore.RESET)

def test_noise_iter(device="cpu", verbose=0):
    print(Fore.CYAN + f"Test Setup: mode = {VSA_MODE}, dim = {DIM}, factors = {FACTORS}, codevectors = {CODEVECTORS}, resonator = {RESONATOR_TYPE}, quantize = {QUANTIZE}, activation = {ACTIVATION}, argmax_abs = {ARGMAX_ABS}, superposed = {NUM_VEC_SUPERPOSED}, samples = {NUM_SAMPLES}" + Fore.RESET)

    csv_file = f'tests/table-{VSA_MODE}-{DIM}d-{FACTORS}f-{v_name(CODEVECTORS)}-{res_name(RESONATOR_TYPE)}-{norm_name(QUANTIZE)}-{act_name(ACTIVATION)}-{argmax_name(ARGMAX_ABS)}-{NUM_VEC_SUPERPOSED}s.csv'

    with open(csv_file, mode='w') as c:
        writer = csv.DictWriter(c, fieldnames=FIELDS)
        writer.writeheader()
        skip_rest_n = False
        for n in NOISE_RANGE:
            if not skip_rest_n:
                skip_rest_i = False
            for it in ITERATION_RANGE:
                if skip_rest_i:
                    if verbose >= 1:
                        print(Fore.YELLOW + f"Skipping {n}n-{it}i" + Fore.RESET)
                    continue
                
                if verbose >= 1:
                    print(Fore.BLUE + f"Running test with noise = {n}, iterations = {it}" + Fore.RESET)

                ret = run_factorization(n=n, it=it, device=device, verbose=verbose)
                if ret is None:
                    continue
                accuracy, unconverged, key, val = ret

                # If no incorrect answer is unconverted, more iterations don't matter
                if (unconverged[1]) == 0:
                    skip_rest_i = True
                # If accuracy is less than 10% for current noise level and more iterations don't have, skip
                if accuracy <= 0.1 and skip_rest_i:
                    skip_rest_n = True

                writer.writerow({FIELDS[0]:key[0],FIELDS[1]:key[1],FIELDS[2]:key[2],FIELDS[3]:key[3],FIELDS[4]:key[4],FIELDS[5]:key[5],FIELDS[6]:key[6],FIELDS[7]:key[7],FIELDS[8]:key[8],FIELDS[9]:key[9],FIELDS[10]:key[10],FIELDS[11]:val[0],FIELDS[12]:val[1][0],FIELDS[13]:val[1][1],FIELDS[14]:NUM_SAMPLES})

    print(Fore.GREEN + f"Saved table to {csv_file}" + Fore.RESET)


def test_norm_act_res(device="cpu", verbose=0):
    print(Fore.CYAN + f"Test Setup: mode = {VSA_MODE}, dim = {DIM}, factors = {FACTORS}, codevectors = {CODEVECTORS}, noise = {NOISE_LEVEL}, iterations = {ITERATIONS}, argmax_abs = {ARGMAX_ABS}, superposed = {NUM_VEC_SUPERPOSED}, samples = {NUM_SAMPLES}" + Fore.RESET)

    csv_file = f'tests/table-{VSA_MODE}-{DIM}d-{FACTORS}f-{v_name(CODEVECTORS)}-{NOISE_LEVEL}n-{ITERATIONS}i-{argmax_name(ARGMAX_ABS)}-{NUM_VEC_SUPERPOSED}s.csv'

    with open(csv_file, mode='w') as c:
        writer = csv.DictWriter(c, fieldnames=FIELDS)
        writer.writeheader()
        for r in RESONATOR_TYPE_RANGE:
            skip_rest_n = False
            for n in QUANTIZE_RANGE:
                for a in ACTIVATION_RANGE:
                    print(Fore.BLUE + f"Running test with resonator = {r}, quantize = {n}, activation = {a}" + Fore.RESET)
                    ret = run_factorization(res=r, q=n, act=a, device=device, verbose=verbose)
                    if ret is None:
                        continue
                    _, _, key, val = ret

                    writer.writerow({FIELDS[0]:key[0],FIELDS[1]:key[1],FIELDS[2]:key[2],FIELDS[3]:key[3],FIELDS[4]:key[4],FIELDS[5]:key[5],FIELDS[6]:key[6],FIELDS[7]:key[7],FIELDS[8]:key[8],FIELDS[9]:key[9],FIELDS[10]:key[10],FIELDS[11]:val[0],FIELDS[12]:val[1][0],FIELDS[13]:val[1][1],FIELDS[14]:NUM_SAMPLES})

    print(Fore.GREEN + f"Saved table to {csv_file}" + Fore.RESET)


# %%

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # torch.set_default_device(device)

    os.makedirs("tests", exist_ok=True)

    print(f"Running tests on {device}, batch_size = {BATCH_SIZE}")

    start = time.time()
    if RUN_MODE == "single":
        print(Fore.CYAN + f"Test Setup: mode = {VSA_MODE}, dim = {DIM}, factors = {FACTORS}, codevectors = {CODEVECTORS}, noise = {NOISE_LEVEL}, resonator = {RESONATOR_TYPE}, iterations = {ITERATIONS}, quantize = {QUANTIZE}, activation = {ACTIVATION}, argmax_abs = {ARGMAX_ABS}, superposed = {NUM_VEC_SUPERPOSED}, samples = {NUM_SAMPLES}" + Fore.RESET)
        run_factorization(device=device, verbose=VERBOSE)
    elif RUN_MODE == "dim-fac-vec":
        test_dim_fac_vec(device=device, verbose=VERBOSE)
    elif RUN_MODE == "noise-iter":
        test_noise_iter(device=device, verbose=VERBOSE)
    elif RUN_MODE == "norm-act-res":
        test_norm_act_res(device=device, verbose=VERBOSE)

    end = time.time()
    print(f"Time elapsed: {end - start}s")

    
# %%
