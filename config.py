
RUN_MODE = "single"
# RUN_MODE = "dim-fac-vec" 
# RUN_MODE = "noise-iter"
# RUN_MODE = "norm-act-res"

VERBOSE = 2
CHECKPOINT = False
NUM_SAMPLES = 100 # test data
BATCH_SIZE = 1
SEED = 0

##################
# Test Parameters
##################
# Multi-vector factorization
NUM_VEC_SUPERPOSED = 1
ALGO = "ALGO1" # ALGO1, ALGO2, ALGO3
TRIALS = NUM_VEC_SUPERPOSED

VSA_MODE = 'SOFTWARE' # 'SOFTWARE', 'HARDWARE'
DIM = 1000
FACTORS = 4
CODEVECTORS = 10
# CODEVECTORS: tuple = (5,5,5,5,2)
# CODEVECTORS : tuple = (3,3,7,10)
# CODEVECTORS : tuple = (110,110,110,2)
NOISE_LEVEL = 0.0  # apply noise to the input compositional vector
# Resonator Network Parameters
ITERATIONS = 100             # max number of iterations for factorization
QUANTIZE = False              # Quantize all bundled vectors, only applies when multiple vectors are superposed
STOCHASTICITY = None       # apply stochasticity, 0.0136, 0.01074
ACTIVATION = 'IDENTITY'       # 'IDENTITY', 'ABS', 'THRESHOLD', 'HARDSHRINK'
LAMBD = 0.1074                # activation threshold; 0.136, 0.1074
RESONATOR_TYPE = "CONCURRENT" # "CONCURRENT", "SEQUENTIAL"
EARLY_CONVERGE = None         # stop when the estimate similarity reaches this value (out of 1.0)
ARGMAX_ABS = True
REORDER_CODEBOOKS = False     # True, False

if NUM_VEC_SUPERPOSED == 1:
    QUANTIZE = True
elif ALGO == "ALGO1":
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