
# RUN_MODE = "single"
# RUN_MODE = "dim-fac-vec" 
# RUN_MODE = "noise-iter"
RUN_MODE = "norm-act-res"

VERBOSE = 1
CHECKPOINT = False
NUM_SAMPLES = 400 # test data
BATCH_SIZE = 1
SEED = 0

##################
# Test Parameters
##################
# Multi-vector factorization
NUM_VEC_SUPERPOSED = 3
ALGO = "ALGO2" # ALGO1, ALGO2, ALGO3
TRIALS = 20    # for ALGO2

VSA_MODE = 'SOFTWARE' # 'SOFTWARE', 'HARDWARE'
DIM = 2000
FACTORS = 4
# CODEVECTORS = 10
CODEVECTORS : tuple = (3,3,7,10)
# CODEVECTORS : tuple = (10,10,10,10,10,3)
NOISE_LEVEL = 0.2  # compositional vector noise
ITERATIONS = 5000    # max number of iterations for factorization
NORMALIZE = True   # for SOFTWARE mode. Normalize the initial estimate and the input vector (when the input is a bundled vector)
ACTIVATION = 'NONE'  # 'NONE', 'ABS', 'THRESHOLD', 'HARDSHRINK'
RESONATOR_TYPE = "SEQUENTIAL" # "CONCURRENT", "SEQUENTIAL"
ARGMAX_ABS = True
REORDER_CODEBOOKS = True

if VSA_MODE == 'HARDWARE':
    NORMALIZE = None

assert(type(CODEVECTORS) == int or len(CODEVECTORS) == FACTORS)

###############
# Test ranges
###############
DIM_RANGE = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
FACTOR_RANGE = [2, 3, 4, 5, 6, 7, 8, 9, 10]
CODEVECTOR_RANGE = [3, 6, 10, 15, 20, 25, 30, 40, 50]
# NORMALIZE_RANGE = [False, True]
NORMALIZE_RANGE = [True]
ACTIVATION_RANGE = ['NONE', 'ABS', 'THRESHOLD', 'HARDSHRINK']
RESONATOR_TYPE_RANGE = ["CONCURRENT", "SEQUENTIAL"]
NOISE_RANGE = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
ITERATION_RANGE = [100, 1000, 5000]

FIELDS = ['Mode', 'Dimensions', 'Factors', 'Codevectors', 'Noise', 'Resonator', 'Iterations', 'Normalize', 'Activation', 'Argmax Absolute', 'Vectors Superposed', 'Accuracy', 'Unconverged Correct', 'Unconverged Incorrect', "Samples"]