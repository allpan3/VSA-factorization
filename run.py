# %%
import torchhd as hd
import torch
from vsa import VSA
from colorama import Fore
import os.path
from tqdm import tqdm
import json
from const import *
import time
from resonator import Resonator
import csv
from dataset import VSADataset
from torch.utils.data import DataLoader
from typing import List

# %%
RUN_MODE = "single"
# RUN_MODE = "dim-fac-vec" 
# RUN_MODE = "noise-iter"
# RUN_MODE = "norm-act-res"

VERBOSE = 0
CHECKPOINT = False
NUM_SAMPLES = 400 # test data
BATCH_SIZE = 1

def v_name(num_codevectors):
    if type(num_codevectors) == int:
        return f"{num_codevectors}v"
    else:
        s = ""
        for i in range(len(num_codevectors)):
            s += f"{num_codevectors[i]},"
        return s[:-1] + "v"

def collate_fn(batch):
    samples = torch.stack([x[0] for x in batch], dim=0)
    labels = [x[1] for x in batch]
    return samples, labels


def gen_init_estimates(codebooks: hd.VSATensor or List[hd.VSATensor], batch_size) -> hd.VSATensor:
    if (type(codebooks) == list):
        guesses = [None] * len(codebooks)
        for i in range(len(codebooks)):
            guesses[i] = hd.multiset(codebooks[i])
        init_estimates = torch.stack(guesses)
    else:
        init_estimates = hd.multiset(codebooks)
    
    return init_estimates.unsqueeze(0).repeat(batch_size,1,1)

def run_factorization(
        m = VSA_MODEL,
        d = DIM,
        f = FACTORS,
        v = CODEVECTORS,
        n = NOISE_LEVEL,
        it = ITERATIONS,
        r = RESONATOR_TYPE,
        norm = NORMALIZE,
        act = ACTIVATION,
        device = "cpu",
        verbose = 0):

    test_dir = f"tests/{m}-{d}d-{f}f-{v_name(v)}"

    # Checkpoint
    cp = os.path.join(test_dir, f"{m}-{d}-{f}-{v}-{n}-{r}-{it}-{norm}-{act}-{NUM_SAMPLES}.checkpoint")
    if CHECKPOINT and os.path.exists(cp):
        if verbose >= 1:
            print(Fore.LIGHTYELLOW_EX + f"Test with {(m, d, f, v, n, r, it, norm, act, NUM_SAMPLES)} already exists, skipping..." + Fore.RESET)
        return

    vsa = VSA(
        root=test_dir,
        dim=d,
        model=m,
        num_factors=f,
        num_codevectors=v,
        device=device
    )

    # Generate test samples
    ds = VSADataset(test_dir, NUM_SAMPLES, vsa, num_vectors_superposed=1, noise=n)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    resonator_network = Resonator(vsa, type=r, norm=norm, activation=act, iterations=it, device=device)
    
    init_estimates = gen_init_estimates(vsa.codebooks, BATCH_SIZE)

    incorrect = 0
    unconverged = [0, 0] # Unconverged successful, unconverged failed
    j = 0
    for samples, labels in tqdm(dl, desc=f"Progress", leave=True if verbose >= 1 else False):

        outcomes, convergence = resonator_network(samples, init_estimates)

        for i in range(len(labels)):
            label = labels[i]
            if (outcomes[i] not in label):
                incorrect += 1
                unconverged[1] += 1 if convergence == it-1 else 0
                if verbose >= 2:
                    print(Fore.RED + f"Test {j} Failed: " + f"Label = {label}, Outcome = {outcomes[i]}" + Fore.RESET + f"    Convergence: {convergence}")
            else:
                unconverged[0] += 1 if convergence == it-1 else 0
                if verbose >= 3:
                    print(f"Test {j} Passed: " + f"Label = {label}, Outcome = {outcomes[i]}    Convergence: {convergence}")
            j += 1


    accuracy = (NUM_SAMPLES - incorrect) / NUM_SAMPLES
    if verbose >= 1:
        print(f"Accuracy: {accuracy}    Unconverged: {unconverged}/{NUM_SAMPLES}")

    # Checkpoint
    with open(cp, "w") as fp:
        pass

    return accuracy, unconverged, (m, d, f, v, n, r, it, norm, act), (accuracy, unconverged)

# Test various dimensions, factors, and codevectors
def test_dim_fac_vec(device="cpu", verbose=0):
    print(Fore.CYAN + f"Test Setup: model = {VSA_MODEL}, normalize = {NORMALIZE}, activation = {ACTIVATION}, noise = {NOISE_LEVEL}, resonator = {RESONATOR_TYPE}, iterations = {ITERATIONS}, samples = {NUM_SAMPLES}" + Fore.RESET)

    csv_file = f'tests/table-{VSA_MODEL}-{NOISE_LEVEL}n-{RESONATOR_TYPE[0:3].lower()}-{ITERATIONS}i{"-norm" if NORMALIZE else ""}{"-" + ACTIVATION.lower() if ACTIVATION != "NONE" else ""}.csv'
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
                    
                    writer.writerow({FIELDS[0]:key[0],FIELDS[1]:key[1],FIELDS[2]:key[2],FIELDS[3]:key[3],FIELDS[4]:key[4],FIELDS[5]:key[5],FIELDS[6]:key[6],FIELDS[7]:key[7],FIELDS[8]:key[8],FIELDS[9]:val[0],FIELDS[10]:val[1][0],FIELDS[11]:val[1][1],FIELDS[12]:NUM_SAMPLES})

        print(Fore.GREEN + f"Saved table to {csv_file}" + Fore.RESET)

def test_noise_iter(device="cpu", verbose=0):
    print(Fore.CYAN + f"Test Setup: model = {VSA_MODEL}, dim = {DIM}, factors = {FACTORS}, codevectors = {CODEVECTORS}, resonator = {RESONATOR_TYPE}, normalize = {NORMALIZE}, activation = {ACTIVATION}, samples = {NUM_SAMPLES}" + Fore.RESET)

    csv_file = f'tests/table-{VSA_MODEL}-{DIM}d-{FACTORS}f-{CODEVECTORS}v-{RESONATOR_TYPE[0:3].lower()}{"-norm" if NORMALIZE else ""}{"-" + ACTIVATION.lower() if ACTIVATION != "NONE" else ""}.csv'
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

                writer.writerow({FIELDS[0]:key[0],FIELDS[1]:key[1],FIELDS[2]:key[2],FIELDS[3]:key[3],FIELDS[4]:key[4],FIELDS[5]:key[5],FIELDS[6]:key[6],FIELDS[7]:key[7],FIELDS[8]:key[8],FIELDS[9]:val[0],FIELDS[10]:val[1][0],FIELDS[11]:val[1][1],FIELDS[12]:NUM_SAMPLES})

    print(Fore.GREEN + f"Saved table to {csv_file}" + Fore.RESET)


def test_norm_act_res(device="cpu", verbose=0):
    print(Fore.CYAN + f"Test Setup: model = {VSA_MODEL}, dim = {DIM}, factors = {FACTORS}, codevectors = {CODEVECTORS}, noise = {NOISE_LEVEL}, iterations = {ITERATIONS}, samples = {NUM_SAMPLES}" + Fore.RESET)

    csv_file = f'tests/table-{VSA_MODEL}-{DIM}d-{FACTORS}f-{CODEVECTORS}v-{NOISE_LEVEL}n-{ITERATIONS}i.csv'
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

                    writer.writerow({FIELDS[0]:key[0],FIELDS[1]:key[1],FIELDS[2]:key[2],FIELDS[3]:key[3],FIELDS[4]:key[4],FIELDS[5]:key[5],FIELDS[6]:key[6],FIELDS[7]:key[7],FIELDS[8]:key[8],FIELDS[9]:val[0],FIELDS[10]:val[1][0],FIELDS[11]:val[1][1],FIELDS[12]:NUM_SAMPLES})

    print(Fore.GREEN + f"Saved table to {csv_file}" + Fore.RESET)


# %%

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    torch.set_default_device(device)

    os.makedirs("tests", exist_ok=True)

    print(f"Running tests on {device}, batch_size = {BATCH_SIZE}")

    start = time.time()
    if RUN_MODE == "single":
        print(Fore.CYAN + f"Test Setup: model = {VSA_MODEL}, dim = {DIM}, factors = {FACTORS}, codevectors = {CODEVECTORS}, noise = {NOISE_LEVEL}, resonator = {RESONATOR_TYPE}, iterations = {ITERATIONS}, normalize = {NORMALIZE}, activation = {ACTIVATION}, samples = {NUM_SAMPLES}" + Fore.RESET)
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
