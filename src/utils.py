import numpy as np


def rank_array(array: np.array) -> np.array:
    array = np.array(array)
    order = array.argsort()
    ranks = order.argsort()
    return ranks


def get_lr(x, y, degree: int = 2, message: str = ""):
    z = np.polyfit(x, y, degree)
    p = np.poly1d(z)
    y_hat = p(x)
    corr = np.corrcoef(y, y_hat)[0][1]
    print(f"R2 score {message}: {corr:.4f}")
    return y_hat
