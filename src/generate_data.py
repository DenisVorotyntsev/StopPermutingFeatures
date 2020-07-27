from typing import Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.special import expit


def generate_weights_gamma(
    gamma: float = 1,
    scale: float = 1,
    n_features: int = 20,
    seed: int = 42
) -> np.array:
    """
    Generate gamma-distributed weights. Sum of weights = 1.
    :param gamma: gamma parameter of gamma distribution
    :param scale: scale parameter of gamma distribution
    :param n_features: number of features (i.e. lengths of weights)
    :param seed: random state
    :return:
    """
    np.random.seed(seed)
    weights = np.random.gamma(gamma, scale, size=n_features)
    weights = weights / np.sum(weights)
    return weights


def get_correlated_data_stats(
        data: np.array
) -> Dict[str, float]:
    """
    Calculated correlation statistics of the given dataset
    :param data: input data
    :return:
    """
    n_features = data.shape[1]
    corr = pd.DataFrame(data).corr()
    corr = np.array(corr)

    assert corr.shape[0] == corr.shape[1] == n_features

    pair_correlations = []
    for i in range(n_features):
        for j in range(n_features):
            if i > j:
                pair_correlations.append(corr[i, j])
    abs_pair_correlations = [abs(c) for c in pair_correlations]

    assert len(pair_correlations) == (n_features * n_features - n_features) / 2

    data_corr_stats = {
        "correlation_min": np.min(pair_correlations),
        "correlation_max": np.max(pair_correlations),
        "correlation_median": np.median(pair_correlations),
        "correlation_mean": np.mean(pair_correlations),
        "correlation_std": np.std(pair_correlations),

        "abs_correlation_min": np.min(abs_pair_correlations),
        "abs_correlation_max": np.max(abs_pair_correlations),
        "abs_correlation_median": np.median(abs_pair_correlations),
        "abs_correlation_mean": np.mean(abs_pair_correlations),
        "abs_correlation_std": np.std(abs_pair_correlations)
    }
    return data_corr_stats


def generate_normal_correlated_data(
        mu: float = 0,
        var: float = 1,
        n_features: int = 20,
        n_samples: int = 2000,
        max_correlation: float = 0.99,
        noise_magnitude_max: float = 3,
        seed: int = 42
) -> np.array:
    """
    Generate normally distributed uncorrelated data and add noise to it.
    :param mu: mean
    :param var: variance
    :param n_features: number of features in generated data
    :param n_samples: number of samples in generated data
    :param max_correlation: max pair correlation between features
    :param noise_magnitude_max: magnitude of noise to add to data.
    Noise will be generated uniformly from [-0.5, 0.5] * noise_magnitude_max range
    :param seed: random state
    :return:
    """
    r = np.ones((n_features, n_features)) * max_correlation * var ** 2
    for i in range(n_features):
        r[i, i] = var

    np.random.seed(seed)
    x = np.random.multivariate_normal([mu] * n_features, r, size=n_samples)

    np.random.seed(seed + 1)
    noise_magnitudes = np.random.random(n_features) * noise_magnitude_max
    for ind, noise_magniture in enumerate(noise_magnitudes):
        np.random.seed(seed + 1 + ind)
        noise = (np.random.random(n_samples) - 0.5) * noise_magniture
        x[:, ind] = x[:, ind] + noise
    x = StandardScaler().fit_transform(x)
    return x


def generate_normal_data(
        mu: float = 0,
        var: float = 1,
        n_features: int = 20,
        n_samples: int = 2000,
        seed: int = 42
) -> np.array:
    """
    Generate normally distributed uncorrelated data
    :param mu: mean
    :param var: variance
    :param n_features: number of features in generated data
    :param n_samples: number of samples in generated data
    :param seed: random state
    :return:
    """
    x = []
    for i in range(n_features):
        np.random.seed(seed + i)
        x_ = np.random.normal(mu, var, n_samples).reshape(-1, 1)
        x.append(x_)
    x = np.hstack(x)
    x = StandardScaler().fit_transform(x)
    return x


def generate_normal_target(
        data: np.array,
        weights: np.array,
        task: str = "classification"
) -> np.array:
    """
    Generate a target for regression or classification task.
    Target is linear combination of data features and corresponding weights (sign selected at random).
    :param data: input features
    :param weights: weight of each feature
    :param task: "classification" (output - binary labels) or "regression" (output - target within (-3,3) range)
    :return:
    """
    n_samples, n_features = data.shape
    assert n_features == len(weights)

    y = np.zeros(n_samples)
    for ind in range(n_features):
        x = data[:, ind]
        weight = weights[ind]

        # randomly select sign of influence - +/-
        np.random.seed(ind)
        if np.random.rand() >= 0.5:
            y = y + x * weight
        else:
            y = y - x * weight

    # min max scale into pre-defined range to avoid sigmoid+round problems
    y = StandardScaler().fit_transform(y.reshape(-1, 1))[:, 0]

    if task == "classification":
        y = expit(y)     # sigmoid
        y = np.round(y)  # get labels
    return y


def generate_normal_target_functions(
        data: np.array,
        task: str = "classification"
) -> np.array:
    n_samples, n_features = data.shape

    functions_to_select_from = {
        "linear": lambda x: x,
        "**2": lambda x: x**2,
        "**3": lambda x: x**3,
        "exp": lambda x: np.exp(x),
        ">0": lambda x: float(x > 0.5),
        "sigmoid": lambda x: expit(x)
    }
    functions_to_select_from = list(functions_to_select_from.items())

    # TODO: check how correlations affect ICI plots
    pass

