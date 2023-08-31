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

# %%
RUN_MODE = "single"
# RUN_MODE = "dim-fac-vec" 
# RUN_MODE = "noise-iter"
# RUN_MODE = "norm-act"

VERBOSE = 1
NUM_SAMPLES = 400 # test data

def v_name(num_codevectors):
    if type(num_codevectors) == int:
        return f"{num_codevectors}v"
    else:
        s = ""
        for i in range(len(num_codevectors)):
            s += f"{num_codevectors[i]},"
        return s[:-1] + "v"

def run_factorization(
        m = VSA_MODEL,
        d = DIM,
        f = FACTORS,
        v = CODEVECTORS,
        n = NOISE_LEVEL,
        it = ITERATIONS,
        t = RESONATOR_TYPE,
        norm = NORMALIZE,
        act = ACTIVATION,
        device = "cpu",
        verbose = 0):

    test_dir = f"tests/{m}-{d}d-{f}f-{v_name(v)}"

    # Checkpoint
    cp = os.path.join(test_dir, f"{m}-{d}-{f}-{v}-{n}-{it}-{norm}-{act}-{NUM_SAMPLES}.checkpoint")
    if os.path.exists(cp):
        if verbose >= 1:
            print(Fore.LIGHTYELLOW_EX + f"Test with {(m, d, f, v, n, it, norm, act, NUM_SAMPLES)} already exists, skipping..." + Fore.RESET)
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
    sample_file = os.path.join(test_dir, f"samples-{NUM_SAMPLES}s-{n}n.pt")
    if (os.path.exists(sample_file)):
        labels, samples = torch.load(sample_file)
        samples = vsa.ensure_vsa_tensor(samples)
    else:   
        labels, samples = vsa.sample(NUM_SAMPLES, num_vectors_supoerposed=1, noise=n)
        torch.save((labels, samples), sample_file)

    samples = samples.to(device)
    resonator_network = Resonator(vsa, type=t, norm=norm, activation=act, iterations=it, device=device)

    incorrect = 0
    unconverged = [0, 0] # Unconverged successful, unconverged failed
    for j in tqdm(range(len(labels)), desc=f"Progress", leave=True if verbose >= 1 else False):
        input = samples[j]
        label = labels[j]

        outcome, convergence = resonator_network(input)

        if (outcome not in label):
            incorrect += 1
            unconverged[1] += 1 if convergence == it-1 else 0
            if verbose >= 2:
                print(Fore.RED + f"Test {j} failed:" + f"Label = {label}, Outcome = {outcome}" + Fore.RESET + f"    Convergence: {convergence}")
        else:
            unconverged[0] += 1 if convergence == it-1 else 0
            if verbose >= 3:
                print(f"Test {j} passed:" + f"Label = {label}, Outcome = {outcome}    Convergence: {convergence}")
        
    accuracy = (NUM_SAMPLES - incorrect) / NUM_SAMPLES
    if verbose >= 1:
        print(f"Accuracy: {accuracy}    Unconverged: {unconverged}/{len(labels)}")

    # Checkpoint
    with open(cp, "w") as fp:
        pass

    return accuracy, unconverged, (m, d, f, v, n, it, norm, act), (accuracy, unconverged)

# Test various dimensions, factors, and codevectors
def test_dim_fac_vec(device="cpu", verbose=0):
    print(Fore.CYAN + f"Test Setup: model = {VSA_MODEL}, normalize = {NORMALIZE}, activation = {ACTIVATION}, noise = {NOISE_LEVEL}, iterations = {ITERATIONS}, samples = {NUM_SAMPLES}" + Fore.RESET)

    csv_file = f'tests/table-{VSA_MODEL}-{ITERATIONS}i-{NOISE_LEVEL}n{"-norm" if NORMALIZE else ""}{"-" + ACTIVATION.lower() if ACTIVATION != "NONE" else ""}.csv'
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
                    
                    writer.writerow({FIELDS[0]:key[0],FIELDS[1]:key[1],FIELDS[2]:key[2],FIELDS[3]:key[3],FIELDS[4]:key[4],FIELDS[5]:key[5],FIELDS[6]:key[6],FIELDS[7]:key[7],FIELDS[8]:val[0],FIELDS[9]:val[1][0],FIELDS[10]:val[1][1], FIELDS[11]:NUM_SAMPLES})

        print(Fore.GREEN + f"Saved table to {csv_file}" + Fore.RESET)

def test_noise_iter(device="cpu", verbose=0):
    print(Fore.CYAN + f"Test Setup: model = {VSA_MODEL}, dim = {DIM}, factors = {FACTORS}, codevectors = {CODEVECTORS}, normalize = {NORMALIZE}, activation = {ACTIVATION}, samples = {NUM_SAMPLES}" + Fore.RESET)

    csv_file = f'tests/table-{VSA_MODEL}-{DIM}d-{FACTORS}f-{CODEVECTORS}v{"-norm" if NORMALIZE else ""}{"-" + ACTIVATION.lower() if ACTIVATION != "NONE" else ""}.csv'
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

                writer.writerow({FIELDS[0]:key[0],FIELDS[1]:key[1],FIELDS[2]:key[2],FIELDS[3]:key[3],FIELDS[4]:key[4],FIELDS[5]:key[5],FIELDS[6]:key[6],FIELDS[7]:key[7],FIELDS[8]:val[0],FIELDS[9]:val[1][0],FIELDS[10]:val[1][1], FIELDS[11]:NUM_SAMPLES})

    print(Fore.GREEN + f"Saved table to {csv_file}" + Fore.RESET)


def test_norm_act(device="cpu", verbose=0):
    print(Fore.CYAN + f"Test Setup: model = {VSA_MODEL}, dim = {DIM}, factors = {FACTORS}, codevectors = {CODEVECTORS}, noise = {NOISE_LEVEL}, iterations = {ITERATIONS}, samples = {NUM_SAMPLES}" + Fore.RESET)

    csv_file = f'tests/table-{VSA_MODEL}-{DIM}d-{FACTORS}f-{CODEVECTORS}v-{ITERATIONS}i-{NOISE_LEVEL}n.csv'
    if os.path.exists(csv_file):
        print(Fore.RED + f"Table {csv_file} already exists, please remove it before running the test." + Fore.RESET)
        return

    with open(csv_file, mode='w') as c:
        writer = csv.DictWriter(c, fieldnames=FIELDS)
        writer.writeheader()
        skip_rest_n = False
        for n in NORMALIZE_RANGE:
            for a in ACTIVATION_RANGE:
                print(Fore.BLUE + f"Running test with normalize = {n}, activation = {a}" + Fore.RESET)
                ret = run_factorization(norm=n, act=a, device=device, verbose=verbose)
                if ret is None:
                    continue
                _, _, key, val = ret

                writer.writerow({FIELDS[0]:key[0],FIELDS[1]:key[1],FIELDS[2]:key[2],FIELDS[3]:key[3],FIELDS[4]:key[4],FIELDS[5]:key[5],FIELDS[6]:key[6],FIELDS[7]:key[7],FIELDS[8]:val[0],FIELDS[9]:val[1][0],FIELDS[10]:val[1][1], FIELDS[11]:NUM_SAMPLES})

    print(Fore.GREEN + f"Saved table to {csv_file}" + Fore.RESET)


# %%

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("Using {} device".format(device))
    torch.set_default_device(device)

    start = time.time()
    if RUN_MODE == "single":
        print(Fore.CYAN + f"Test Setup: model = {VSA_MODEL}, dim = {DIM}, factors = {FACTORS}, codevectors = {CODEVECTORS}, noise = {NOISE_LEVEL}, iterations = {ITERATIONS}, normalize = {NORMALIZE}, activation = {ACTIVATION}, samples = {NUM_SAMPLES}" + Fore.RESET)
        run_factorization(device=device, verbose=VERBOSE)
    elif RUN_MODE == "dim-fac-vec":
        test_dim_fac_vec(device=device, verbose=VERBOSE)
    elif RUN_MODE == "noise-iter":
        test_noise_iter(device=device, verbose=VERBOSE)
    elif RUN_MODE == "norm-act":
        test_norm_act(device=device, verbose=VERBOSE)

    end = time.time()
    print(f"Time elapsed: {end - start}s")

    
# %%
