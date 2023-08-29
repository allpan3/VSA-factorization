# %%
import torchhd as hd
import torch
from vsa import VSA
from colorama import Fore
import os.path
from typing import Literal
from tqdm import tqdm
import itertools
import json
from const import *

# %%
RUN_SINGLE = False
NUM_SAMPLES = 400 # test data

assert(type(NUM_CODEVECTORS) == int or len(NUM_CODEVECTORS) == NUM_FACTORS)

def codevector_filename(num_codevectors):
    if type(num_codevectors) == int:
        return f"{num_codevectors}v"
    else:
        s = ""
        for i in range(len(num_codevectors)):
            s = f"{num_codevectors[i]},"
        return s[:-1] + "v"

def resonator(input,
              estimates: hd.VSATensor or list,
              codebooks: hd.VSATensor or list,
              activation: Literal['NONE', 'ABS', 'ZERO'] = 'NONE'
            ):
    if type(estimates) is list:
        pass
    else:
        n = estimates.size(-2)

        # Get binding inverse of the estimates
        inv_estimates = estimates.inverse()

        # Roll over the number of estimates to align each row with the other symbols
        # Example: for factorizing x, y, z the stacked matrix has the following estimates:
        # [[z, y],
        #  [x, z],
        #  [y, x]]
        rolled = []
        for i in range(1, n):
            rolled.append(inv_estimates.roll(i, -2))

        inv_estimates = torch.stack(rolled, dim=-2)

        # First bind all the other estimates together: z * y, x * z, y * z
        inv_others = hd.multibind(inv_estimates)
        # Then unbind all other estimates from the input: s * (x * y), s * (x * z), s * (y * z)
        new_estimates = hd.bind(input.unsqueeze(-2), inv_others)

        similarity = hd.dot_similarity(new_estimates.unsqueeze(-2), codebooks)
        if (activation == 'ABS'):
            similarity = torch.abs(similarity)

        output = hd.dot_similarity(similarity, codebooks.transpose(-2, -1)).squeeze(-2)
        # This should be normalizing back to 1 and -1, but sign can potentially keep 0 at 0. It's very unlikely to see a 0 and sign() is fast 
        output = output.sign()
        return output



def factorization(input, estimates, codebooks):

    old_estimates = estimates.clone()
    for k in range(NUM_ITERATIONS):
        estimates = resonator(input, estimates, codebooks, activation=ACTIVATION)
        if all((estimates == old_estimates).flatten().tolist()):
            break
        old_estimates = estimates.clone()

    return estimates, k 


def run(m,d,f,v, verbose=False):
    print(Fore.BLUE + f"Running test with {m} model, {d} dimensions, {f} factors, {v} codevectors" + Fore.RESET)

    test_dir = f"tests/{m}-{d}d-{f}f-{codevector_filename(v)}"

    vsa = VSA(
        root=test_dir,
        dim=d,
        model=m,
        num_factors=f,
        num_codevectors=v
    )

    # Generate test samples
    sample_file = os.path.join(test_dir, f"samples-{NUM_SAMPLES}s-{NOISE_LEVEL}n.pt")
    if (os.path.exists(sample_file)):
        labels, samples = torch.load(sample_file)
    else:   
        labels, samples = vsa.sample(NUM_SAMPLES, num_vectors_supoerposed=1, noise=NOISE_LEVEL)
        torch.save((labels, samples), sample_file)

    codebooks = torch.stack(vsa.codebooks)
    init_estimate = hd.multiset(codebooks)

    incorrect = 0
    unconverged = 0
    for i in tqdm(range(len(labels))):
        input = samples[i]
        label = labels[i]

        output, convergence = factorization(input, init_estimate, codebooks)

        # values: the similarity values
        # indices: the indices of the codevectors in the codebooks
        values, outcome = vsa.cleanup(output)

        if (convergence == NUM_ITERATIONS-1):
            unconverged += 1

        if (outcome not in label):
            incorrect += 1
            if verbose:
                print(Fore.RED + f"Test {i} failed:" + f"Label = {label}, Outcome = {outcome}" + Fore.RESET)
                print(f"Convergence: {convergence}")
        
    accuracy = (NUM_SAMPLES - incorrect) / NUM_SAMPLES
    print(f"Accuracy: {accuracy}")
    print(f"Unconverged: {unconverged}/{len(labels)}")
    return accuracy, unconverged


if __name__ == '__main__':
    table = {}

    if RUN_SINGLE:
        run(VSA_MODEL, NUM_DIMENSIONS, NUM_FACTORS, NUM_CODEVECTORS, verbose=True)
    else:
        for d in range(NUM_DIMENSIONS, MAX_DIMENSIONS+1, STEP_DIMENSIONS):
            skip_rest_f = False
            for f in range(NUM_FACTORS, MAX_FACTORS+1, STEP_FACTORS):
                if not skip_rest_f:
                    skip_rest_v = False 
                for v in range(NUM_CODEVECTORS, MAX_CODEVECTORS+1, STEP_CODEVECTORS):
                    if skip_rest_v:
                        print(Fore.YELLOW + f"Skipping {VSA_MODEL}-{d}d-{f}f-{codevector_filename(v)}" + Fore.RESET)
                        continue
                    accuracy, unconverged = run(VSA_MODEL, d,f,v, verbose=False)
                    table[str((VSA_MODEL, d, f, v))] = (accuracy, unconverged)
                    # If accuracy is less than 30%, skip the rest of the tests
                    if (accuracy <= 0.3):
                        skip_rest_v = True
                        # If the first stage fails, skip the rest of f
                        if (v == NUM_CODEVECTORS):
                            skip_rest_f = True
        
        with open(f'tests/table-{NUM_ITERATIONS}i-{NOISE_LEVEL}n.json', 'w') as f:
            json.dump(table, f)
    
# %%
