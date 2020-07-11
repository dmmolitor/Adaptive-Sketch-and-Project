import numpy as np
from zlib import crc32
random = np.random.RandomState(crc32(str.encode(__file__)))

def gaussian(*, mean=0, std=1, shape):
    return random.normal(mean, std, shape)

def uniform(*, low=0, high=1, shape):
    return random.uniform(low, high, shape)
