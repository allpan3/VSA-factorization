# %%
VSA_MODEL = 'HARDWARE' # 'SOFTWARE', 'HARDWARE'

# Default values
DIM = 2000
FACTORS = 4
CODEVECTORS = 10
# CODEVECTORS : tuple = (3,3,3,10) 
NOISE_LEVEL = 0.3    # compositional vector noise
ITERATIONS = 1000   # max number of iterations for factorization
NORMALIZE = True    # normalize the initial estimate and the input vector (when the input is a bundled vector)
ACTIVATION = 'NONE' # 'NONE', 'ABS', 'NONNEG'
RESONATOR_TYPE = "SEQUENTIAL" # "CONCURRENT", "SEQUENTIAL"
ARGMAX_ABS = True
assert(type(CODEVECTORS) == int or len(CODEVECTORS) == FACTORS)

if VSA_MODEL == 'HARDWARE':
    # Normalize is always required for hardware model
    assert(NORMALIZE == True)

# Test ranges
DIM_RANGE = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
FACTOR_RANGE = [2, 3, 4, 5, 6, 7, 8, 9, 10]
CODEVECTOR_RANGE = [3, 6, 10, 15, 20, 25, 30, 40, 50]
NORMALIZE_RANGE = [False, True]
ACTIVATION_RANGE = ['NONE', 'ABS', 'NONNEG']
RESONATOR_TYPE_RANGE = ["CONCURRENT", "SEQUENTIAL"]
NOISE_RANGE = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
ITERATION_RANGE = [100, 1000, 5000]

FIELDS = ['Model', 'Dimensions', 'Factors', 'Codevectors', 'Noise', 'Resonator', 'Iterations', 'Normalize', 'Activation', 'Argmax Absolute', 'Accuracy', 'Unconverged Correct', 'Unconverged Incorrect', "Samples"]

# %%
