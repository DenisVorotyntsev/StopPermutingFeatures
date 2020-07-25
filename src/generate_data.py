from typing import Tuple, Dict

import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.special import expit


def generate_weights_gamma(
    gamma: float = 1,
    scale: float = 1,
    n_features: int = 20,
    weights_range: Tuple[float, float] = (0.2, 1),
    seed: int = 42
) -> np.array:
    np.random.seed(seed)
    weights = np.random.gamma(gamma, scale, size=n_features)
    weights = weights / np.sum(weights)
    weights = MinMaxScaler(feature_range=weights_range).fit_transform(weights.reshape(-1, 1))[:, 0]
    return weights


def generate_weights_uniform(
        n_features: int = 20,
        weights_range: Tuple[float, float] = (0.2, 1),
        seed: int = 42
) -> np.array:
    min_, max_ = weights_range
    weights = []
    for i in range(n_features):
        random.seed(seed+i)
        w = random.randint(min_, max_)
        weights.append(w)
    weights = np.array(weights)
    weights = MinMaxScaler(feature_range=weights_range).fit_transform(weights.reshape(-1, 1))[:, 0]
    return weights


def get_correlated_data_stats(data: np.array) -> Dict[str, float]:
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
        noise_magnitude_loc: float = 3,
        noise_magnitude_scale: float = 1,
        seed: int = 42
) -> np.array:
    r = np.ones((n_features, n_features)) * 0.99 * var ** 2
    for i in range(n_features):
        r[i, i] = var

    np.random.seed(seed)
    x = np.random.multivariate_normal([mu] * n_features, r, size=n_samples)

    np.random.seed(seed + 1)
    noise_magnitudes = np.random.normal(noise_magnitude_loc, noise_magnitude_scale, n_features)
    for ind, noise_magniture in enumerate(noise_magnitudes):
        np.random.seed(seed + 1 + ind)
        x[:, ind] = x[:, ind] + np.random.random(n_samples) * noise_magniture
    x = StandardScaler().fit_transform(x)
    return x


def generate_normal_data(
        mu: float = 0,
        var: float = 1,
        n_features: int = 20,
        n_samples: int = 2000,
        seed: int = 42
) -> np.array:
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
        y = expit(y)   # sigmoid
        y = np.round(y)

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

