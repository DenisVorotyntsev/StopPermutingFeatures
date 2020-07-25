import numpy as np


def rank_array(array: np.array) -> np.array:
    array = np.array(array)
    order = array.argsort()
    ranks = order.argsort()
    return ranks
