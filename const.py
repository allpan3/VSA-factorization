VSA_MODEL = 'MAP'

# Default values
DIM = 1000
FACTORS = 3
CODEVECTORS = 3
# CODEVECTORS = [3, 3, 5, 7]
NOISE_LEVEL = 0.0    # compositional vector noise
ITERATIONS = 100  # max number of iterations for factorization
NORMALIZE = False    # normalize the initial estimate and the input vector
ACTIVATION = 'NONE' # 'NONE', 'ABS'

assert(type(CODEVECTORS) == int or len(CODEVECTORS) == FACTORS)

# Test ranges
DIM_RANGE = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
FACTOR_RANGE = [2, 3, 4, 5, 6, 7, 8, 9, 10]
CODEVECTOR_RANGE = [2, 3, 6, 10, 15, 20, 25, 30, 40, 50]
NORMALIZE_RANGE = [False, True]
ACTIVATION_RANGE = ['NONE', 'ABS']
NOISE_RANGE = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
ITERATION_RANGE = [100, 1000, 5000, 10000]