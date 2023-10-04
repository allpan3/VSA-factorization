
import math

RUN_MODE = "single"
# RUN_MODE = "dim-fac-vec" 
# RUN_MODE = "noise-iter"
# RUN_MODE = "norm-act-res"

VERBOSE = 2
CHECKPOINT = False
NUM_SAMPLES = 300 # test data (for each count)
BATCH_SIZE = 1
SEED = 0

##################
# VSA Parameters
##################
VSA_MODE = 'HARDWARE'   # 'SOFTWARE', 'HARDWARE'
DIM = 2048
FACTORS = 4
CODEVECTORS = 4
# CODEVECTORS: tuple = (25,30,40,50)
# CODEVECTORS : tuple = (4,5,6)
# CODEVECTORS : tuple = (3,3,7,10)
FOLD_DIM = 256
EHD_BITS = 8                           # Expanded HD per-dimension bits, for hardware mode
SIM_BITS = 13         # Similarity value bits, for hardware mode
assert(type(CODEVECTORS) == int or len(CODEVECTORS) == FACTORS)

##################
# Test Parameters
##################
# Multi-vector factorization
NUM_VEC_SUPERPOSED = range(1,4)           # an integer, a list, or a range
COUNT_KNOWN = False
# OVERLAP = False
ALGO = "ALGO1" # ALGO1, ALGO2, ALGO3, ALGO4
MAX_TRIALS = max(NUM_VEC_SUPERPOSED) * 3
PARALLEL_TRIALS = 2
ENERGY_THRESHOLD = 0.25             # Below this value, it is considered that all vectors have been extracted
# Similarity thresholds are affected by the maximum number of vectors superposed. These values need to be lowered when more vectors are superposed
SIM_EXPLAIN_THRESHOLD = 0.22        # Above this value, the vector is explained away
SIM_DETECT_THRESHOLD = 0.12         # Above this value, the vector is considered a valid vector (for when count is unknown)

NOISE_LEVEL = 0.0                   # Apply noise to the input compositional vector
QUANTIZE = True               # Quantize all bundled vectors, only applies when multiple vectors are superposed
if NUM_VEC_SUPERPOSED == 1:
    # For simple-vector factorization, we don't need to sample the expanded vectors (algorithms don't apply)
    QUANTIZE = True
elif ALGO == "ALGO1" or "ALGO4":
    # For explain away methods we need to sample the expanded vectors
    QUANTIZE = False

assert not COUNT_KNOWN or type(NUM_VEC_SUPERPOSED) == int, "When the count is known we cannot have different numbers of vectors superposed"


##################
# Resonator Network Parameters
##################
ITERATIONS = 200             # max number of iterations for factorization
STOCHASTICITY = "SIMILARITY"  # apply stochasticity: "NONE", "VECTOR", "SIMILARITY"
RANDOMNESS = 0.03             # randomness for stochasticity, value of standard deviation, 0.02 ~ 0.05
ACTIVATION = 'THRESH_AND_SCALE'      # 'IDENTITY', 'THRESHOLD', 'SCALEDOWN', "THRESH_AND_SCALE"
ACT_VALUE = 32                 # Activation value, either a similarity threshold or a scale down factor
                              # Typical threshold range = [0, 100], scale down factor is the divisor, which is effectively a threshold
RESONATOR_TYPE = "SEQUENTIAL" # "CONCURRENT", "SEQUENTIAL"
EARLY_CONVERGE = 0.6         # stop when the estimate similarity reaches this value (out of 1.0)
ARGMAX_ABS = True
REORDER_CODEBOOKS = False    # Place codebooks with larger number of codevectors first, affects sequential resonator

# In hardware mode, the activation value needs to be a power of two
if VSA_MODE == "HARDWARE" and (ACTIVATION == "SCALEDOWN" or ACTIVATION == "THRESH_AND_SCALE"):
    def biggest_power_two(n):
        """Returns the biggest power of two <= n"""
        # if n is a power of two simply return it
        if not (n & (n - 1)):
            return n
        # else set only the most significant bit
        return int("1" + (len(bin(n)) - 3) * "0", 2)
    ACT_VALUE = biggest_power_two(ACT_VALUE)

# If activation is scaledown, then the early convergence threshold needs to scale down accordingly
if EARLY_CONVERGE is not None and (ACTIVATION == "SCALEDOWN" or ACTIVATION == "THRESHOLDED_SCALEDOWN"):
    EARLY_CONVERGE = EARLY_CONVERGE / ACT_VALUE

###############
# Test Ranges
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