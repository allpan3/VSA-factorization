import torch
from torch import Tensor
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

def get_similarity(v1, v2, norm=True):
    """
    Return the hamming similarity for normalized vectors, and cosine similarity for unnormalized vectors
    Hamming similarity is linear and should reflect the noise level
    Cosine similarity is non-linear and may not reflect the noise level
    By default, always normalize the inputs vectors before comparison (only applies to software mode because
    in hardware mode vectors are always normalized), but allow the option to disable it if that's desired
    """
    if VSA_MODE == "SOFTWARE":
        if norm:
            # Compare the normalized vectors
            positive = torch.tensor(1, device=v1.device)
            negative = torch.tensor(-1, device=v1.device)
            v1_ = torch.where(v1 >= 0, positive, negative)
            v2_ = torch.where(v2 >= 0, positive, negative)
            return torch.sum(torch.where(v1_ == v2_, 1, 0), dim=-1) / DIM
        else:
            v1_dot = torch.sum(v1 * v1, dim=-1)
            v1_mag = torch.sqrt(v1_dot)
            v2_dot = torch.sum(v2 * v2, dim=-1)
            v2_mag = torch.sqrt(v2_dot)
            magnitude = v1_mag * v2_mag
            magnitude = torch.clamp(magnitude, min=1e-08)
            return torch.matmul(v1.type(torch.float32), v2.type(torch.float32)) / magnitude
    else:
        return torch.sum(torch.where(v1 == v2, 1, 0), dim=-1) / DIM

def algo1(vsa, rn, inputs, init_estimates, codebooks, orig_indices, norm):
    """
    Subtract every extracted vector from the original input and repeat the process NUM_VEC_SUPERPOSED times 
    This method only works with unnormalized inputs, i.e. hardware mode will not work
    Initial estiamtes are the multiset of all codevectors
    """
    assert(VSA_MODE == "SOFTWARE")

    inputs = inputs.clone()

    outcomes = [[] for _ in range(inputs.size(0))]  # num of batches
    convergence = [0] * inputs.size(0)

    for _ in range(NUM_VEC_SUPERPOSED):
        if norm:
            inputs_ = vsa.normalize(inputs)
        else:
            inputs_ = inputs

        outcome, converg = rn(inputs_, init_estimates, codebooks, orig_indices) 

        # Split batch results
        for i in range(len(outcome)):
            outcomes[i].append(outcome[i])
            convergence[i] += converg

            # Get the compositional vector and subtract it from the input
            vector = vsa.get_vector(outcome[i])
            inputs[i] = inputs[i] - vector 
    
    return outcomes, convergence

def algo2(vsa, rn, inputs, d, f, codebooks, orig_indices, norm):
    """
    Run the resonator a predefined number of times with different, randomly generated initial estimates.
    Reduce the final results to remove the duplicated vectors. The remainder are expected to contain all
    superposed vectors and should not contain any non-existing vectors.
    """
    inputs = inputs.clone()

    outcomes = [set() for _ in range(inputs.size(0))]  # num of batches
    convergence = [0] * inputs.size(0)

    if norm:
        # This is essentially the same as hardware mode
        inputs = vsa.normalize(inputs)

    for _ in range(TRIALS):
        # Pure random
        init_estimates = vsa.random(f, d).repeat(inputs.size(0), 1, 1)

        outcome, converg = rn(inputs, init_estimates, codebooks, orig_indices)

        # Split batch results
        for i in range(len(outcome)):
            outcomes[i].add(outcome[i])
            convergence[i] += converg
        
        # As soon as the required number of vectors are extracted, stop
        if all([len(outcomes[i]) == NUM_VEC_SUPERPOSED for i in range(len(outcomes))]):
            break
        
    return outcomes, convergence


def algo3(vsa, rn, inputs, norm):
    """
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

    init_estimates = rn.get_init_estimates(codebooks, BATCH_SIZE)

    if norm:
        # This is essentially the same as hardware mode
        inputs = vsa.normalize(inputs)
        init_estimates = vsa.normalize(init_estimates)

    outcomes = [[] for _ in range(inputs.size(0))]  # num of batches
    convergence = [0] * inputs.size(0)

    for k in range(NUM_VEC_SUPERPOSED):
        # Unbind the ID
        inputs_ = vsa.bind(inputs, id_cb[k])

        outcome, converg = rn(inputs_, init_estimates, codebooks, orig_indices)

        # Split batch results
        for i in range(len(outcome)):
            outcomes[i].append(outcome[i])
            convergence[i] += converg
        
    return outcomes, convergence


def run_factorization(
        m = VSA_MODE,
        d = DIM,
        f = FACTORS,
        v = CODEVECTORS,
        n = NOISE_LEVEL,
        it = ITERATIONS,
        res = RESONATOR_TYPE,
        norm = NORMALIZE,
        act = ACTIVATION,
        abs = ARGMAX_ABS,
        device = "cpu",
        verbose = 0):

    test_dir = f"tests/{m}-{d}d-{f}f-{v_name(v)}"

    # Checkpoint
    cp = os.path.join(test_dir, f"{m}-{d}d-{f}f-{v_name(v)}-{n}n-{res_name(res)}-{it}i-{norm_name(norm)}-{act_name(act)}-{argmax_name(abs)}-{NUM_VEC_SUPERPOSED}s-{NUM_SAMPLES}s.checkpoint")
    if CHECKPOINT and os.path.exists(cp):
        if verbose >= 1:
            print(Fore.LIGHTYELLOW_EX + f"Test with {(m, d, f, v, n, res_name(res), it, norm_name(norm), act_name(act), (argmax_name(abs)), {NUM_VEC_SUPERPOSED}, NUM_SAMPLES)} already exists, skipping..." + Fore.RESET)
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
    ds = VSADataset(test_dir, NUM_SAMPLES, vsa, algo=ALGO, num_vectors_superposed=NUM_VEC_SUPERPOSED, noise=n)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    rn = Resonator(vsa, type=res, activation=act, iterations=it, argmax_abs=abs, device=device)

    codebooks = None
    orig_indices = None

    if REORDER_CODEBOOKS:
        codebooks, orig_indices = rn.reorder_codebooks(codebooks)

    init_estimates = rn.get_init_estimates(codebooks, BATCH_SIZE)
    if norm:
        init_estimates = vsa.normalize(init_estimates)

    incorrect_count = 0
    unconverged = [0, 0] # Unconverged successful, unconverged failed
    j = 0
    for data, labels in tqdm(dl, desc=f"Progress", leave=True if verbose >= 1 else False):

        if (NUM_VEC_SUPERPOSED == 1):
            outcome, convergence = rn(data, init_estimates, codebooks, orig_indices)
            # Make them the same format as multi-vector for easier parsing
            outcomes = [[outcome[x]] for x in range(BATCH_SIZE)]
            convergences = [convergence] * BATCH_SIZE
        else:
            if ALGO == "ALGO1":
                outcomes, convergences = algo1(vsa, rn, data, init_estimates, codebooks, orig_indices, norm)
            elif ALGO == "ALGO2":
                outcomes, convergences = algo2(vsa, rn, data, d, f, codebooks, orig_indices, norm)
            elif ALGO == "ALGO3":
                outcomes, convergences = algo3(vsa, rn, data, norm)

        ## Analyze results
        # Batch
        for k in range(len(labels)):
            message = ""
            incorrect = False
            label = labels[k]
            outcome = outcomes[k]
            convergence = convergences[k]
            sim_per_vec = []

            # Calculate the similarity between the input compositional vector and the groundtruth
            # This is to verify the effectiveness of noise
            if NUM_VEC_SUPERPOSED > 1 and ALGO == "ALGO3":
                # Get the correctly bound (with ID) groundtruth vectors 
                gt_vecs = ds.lookup_algo3(label, bundled=False)
                similarity = round(get_similarity(ds.lookup_algo3(label), data[k], norm).item(), 3)
            else:
                similarity = round(get_similarity(data[k], vsa.get_vector(label), norm).item(), 3)


            # Multiple vectors superposed
            for i in range(len(label)):
                if (label[i] not in outcome):
                    incorrect = True
                    message += Fore.RED + "Vector {} is not detected.".format(label[i]) + Fore.RESET + "\n"
                else:
                    message += "Vector {} is correctly detected.".format(label[i]) + "\n" 

                # Per vector similarity.
                if ALGO == "ALGO3":
                    sim_per_vec.append(round(get_similarity(gt_vecs[i], data[k], norm).item(), 3))
                else:
                    sim_per_vec.append(round(get_similarity(vsa.get_vector(label[i]), data[k], norm).item(), 3))
            
            # Print results
            if incorrect:
                incorrect_count += 1
                unconverged[1] += 1 if convergence >= it-1 else 0
                if verbose >= 2:
                    print(Fore.BLUE + f"Test {j} Failed: Convergence: {convergence}" + Fore.RESET)
                    print("Input similarity = {}".format(similarity))
                    if NUM_VEC_SUPERPOSED > 1:
                        print("Per-vector similarity = {}".format(sim_per_vec))
                    print(message[:-1])
                    print(f"Outcome = {outcome}")
            else:
                unconverged[0] += 1 if convergence >= it-1 else 0
                if verbose >= 3:
                    print(Fore.BLUE + f"Test {j} Passed: Convergence: {convergence}" + Fore.RESET)
                    print("Input similarity = {}".format(similarity))
                    if NUM_VEC_SUPERPOSED > 1:
                        print("Per-vector similarity = {}".format(sim_per_vec))
                    print(message[:-1])
            j += 1

    accuracy = (NUM_SAMPLES - incorrect_count) / NUM_SAMPLES
    if verbose >= 1:
        print(f"Accuracy: {accuracy}    Unconverged: {unconverged}/{NUM_SAMPLES}")

    # Checkpoint
    with open(cp, "w") as fp:
        pass

    return accuracy, unconverged, (m, d, f, v, n, res, it, norm, act, abs, NUM_VEC_SUPERPOSED), (accuracy, unconverged)

# Test various dimensions, factors, and codevectors
def test_dim_fac_vec(device="cpu", verbose=0):
    print(Fore.CYAN + f"Test Setup: mode = {VSA_MODE}, normalize = {NORMALIZE}, activation = {ACTIVATION}, argmax_abs = {ARGMAX_ABS}, noise = {NOISE_LEVEL}, resonator = {RESONATOR_TYPE}, iterations = {ITERATIONS}, superposed = {NUM_VEC_SUPERPOSED}, samples = {NUM_SAMPLES}" + Fore.RESET)

    csv_file = f'tests/table-{VSA_MODE}-{NOISE_LEVEL}n-{res_name(RESONATOR_TYPE)}-{ITERATIONS}i-{norm_name(NORMALIZE)}-{act_name(ACTIVATION)}-{argmax_name(ARGMAX_ABS)}-{NUM_VEC_SUPERPOSED}s.csv'
    if os.path.exists(csv_file):
        print(Fore.RED + f"Table {csv_file} already exists, please remove it before running the test." + Fore.RESET)
        return

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
    print(Fore.CYAN + f"Test Setup: mode = {VSA_MODE}, dim = {DIM}, factors = {FACTORS}, codevectors = {CODEVECTORS}, resonator = {RESONATOR_TYPE}, normalize = {NORMALIZE}, activation = {ACTIVATION}, argmax_abs = {ARGMAX_ABS}, superposed = {NUM_VEC_SUPERPOSED}, samples = {NUM_SAMPLES}" + Fore.RESET)

    csv_file = f'tests/table-{VSA_MODE}-{DIM}d-{FACTORS}f-{v_name(CODEVECTORS)}v-{res_name(RESONATOR_TYPE)}-{norm_name(NORMALIZE)}-{act_name(ACTIVATION)}-{argmax_name(ARGMAX_ABS)}-{NUM_VEC_SUPERPOSED}s.csv'
    if os.path.exists(csv_file):
        print(Fore.RED + f"Table {csv_file} already exists, please remove it before running the test." + Fore.RESET)
        return

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

    csv_file = f'tests/table-{VSA_MODE}-{DIM}d-{FACTORS}f-{v_name(CODEVECTORS)}v-{NOISE_LEVEL}n-{ITERATIONS}i-{argmax_name(ARGMAX_ABS)}-{NUM_VEC_SUPERPOSED}s.csv'
    if os.path.exists(csv_file):
        print(Fore.RED + f"Table {csv_file} already exists, please remove it before running the test." + Fore.RESET)
        return

    with open(csv_file, mode='w') as c:
        writer = csv.DictWriter(c, fieldnames=FIELDS)
        writer.writeheader()
        for r in RESONATOR_TYPE_RANGE:
            skip_rest_n = False
            for n in NORMALIZE_RANGE:
                for a in ACTIVATION_RANGE:
                    print(Fore.BLUE + f"Running test with resonator = {r}, normalize = {n}, activation = {a}" + Fore.RESET)
                    ret = run_factorization(res=r, norm=n, act=a, device=device, verbose=verbose)
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
        print(Fore.CYAN + f"Test Setup: mode = {VSA_MODE}, dim = {DIM}, factors = {FACTORS}, codevectors = {CODEVECTORS}, noise = {NOISE_LEVEL}, resonator = {RESONATOR_TYPE}, iterations = {ITERATIONS}, normalize = {NORMALIZE}, activation = {ACTIVATION}, argmax_abs = {ARGMAX_ABS}, superposed = {NUM_VEC_SUPERPOSED}, samples = {NUM_SAMPLES}" + Fore.RESET)
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
