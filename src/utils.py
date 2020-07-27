import numpy as np
import pandas as pd


def rank_array(array: np.array) -> np.array:
    """
    Rank input 1d array
    :param array:
    :return:
    """
    array = np.array(array)
    order = array.argsort()
    ranks = order.argsort()
    return ranks


def get_lr(
        x: pd.Series,
        y: pd.Series,
        degree: int = 2,
        message: str = ""
) -> np.array:
    """
    Fit Linear Regression for a given data and calculate it's correlation.
    :param x:
    :param y:
    :param degree:
    :param message:
    :return:
    """
    z = np.polyfit(x, y, degree)
    p = np.poly1d(z)
    y_hat = p(x)
    corr = np.corrcoef(y, y_hat)[0][1]
    print(f"R2 score {message}: {corr:.4f}")
    return y_hat
