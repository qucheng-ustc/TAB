import numpy as np
import utils.tfutil

def set_random_seed(seed):
    np.random.seed(seed)
    utils.tfutil.set_random_seed(seed)
