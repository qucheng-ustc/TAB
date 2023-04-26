import numpy as np

def set_random_seed(seed):
    import utils.tfutil
    np.random.seed(seed)
    utils.tfutil.set_random_seed(seed)
