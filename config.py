
RUN_MODE = "single"
# RUN_MODE = "dim-fac-vec" 
# RUN_MODE = "noise-iter"
# RUN_MODE = "norm-act-res"

VERBOSE = 3
CHECKPOINT = False
NUM_SAMPLES = 300 # test data
BATCH_SIZE = 1
SEED = 0

##################
# Test Parameters
##################
# Multi-vector factorization
NUM_VEC_SUPERPOSED = 5
COUNT_KNOWN = False
# OVERLAP = False
ALGO = "ALGO1" # ALGO1, ALGO2, ALGO3, ALGO4
MAX_TRIALS = NUM_VEC_SUPERPOSED * 3
PARALLEL_TRIALS = 2
ENERGY_THRESHOLD = 0.22             # Below this value, it is considered that all vectors have been extracted
SIM_EXPLAIN_THRESHOLD = 0.15        # Above this value, the vector is explained away
SIM_DETECT_THRESHOLD = 0.08         # Above this value, the vector is considered a valid vector (for when count is unknown)

VSA_MODE = 'HARDWARE' # 'SOFTWARE', 'HARDWARE'
DIM = 1024
FACTORS = 4
# CODEVECTORS = 3
# CODEVECTORS: tuple = (25,30,40,50)
CODEVECTORS : tuple = (3,3,7,10)
# CODEVECTORS : tuple = (110,110,110,2)
NOISE_LEVEL = 0.0  # apply noise to the input compositional vector
# Resonator Network Parameters
ITERATIONS = 2000             # max number of iterations for factorization
QUANTIZE = True               # Quantize all bundled vectors, only applies when multiple vectors are superposed
STOCHASTICITY = "SIMILARITY"  # apply stochasticity: "NONE", "VECTOR", "SIMILARITY"
RANDOMNESS = 0.05             # randomness for stochasticity, value of standard deviation, 0.02 ~ 0.05
ACTIVATION = 'THRESHOLD'      # 'IDENTITY', 'ABS', 'THRESHOLD', 'HARDSHRINK'
LAMBD = 0.0                    # activation threshold; 0.0 ~ 0.1
RESONATOR_TYPE = "SEQUENTIAL" # "CONCURRENT", "SEQUENTIAL"
EARLY_CONVERGE = None         # stop when the estimate similarity reaches this value (out of 1.0)
ARGMAX_ABS = False
REORDER_CODEBOOKS = False    # Place codebooks with larger number of codevectors first, affects sequential resonator

if NUM_VEC_SUPERPOSED == 1:
    QUANTIZE = True
elif ALGO == "ALGO1" or "ALGO4":
    QUANTIZE = False

assert(type(CODEVECTORS) == int or len(CODEVECTORS) == FACTORS)

###############
# Test ranges
###############
DIM_RANGE = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
FACTOR_RANGE = [2, 3, 4, 5, 6, 7, 8, 9, 10]
CODEVECTOR_RANGE = [3, 6, 10, 15, 20, 25, 30, 40, 50]
QUANTIZE_RANGE = [False, True]
ACTIVATION_RANGE = ['NONE', 'ABS', 'THRESHOLD', 'HARDSHRINK']
RESONATOR_TYPE_RANGE = ["SEQUENTIAL"]
NOISE_RANGE = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
ITERATION_RANGE = [100, 1000, 5000]

FIELDS = ['Mode', 'Dimensions', 'Factors', 'Codevectors', 'Noise', 'Resonator', 'Iterations', 'QUANTIZE', 'Activation', 'Argmax Absolute', 'Vectors Superposed', 'Accuracy', 'Unconverged Correct', 'Unconverged Incorrect', "Samples"]