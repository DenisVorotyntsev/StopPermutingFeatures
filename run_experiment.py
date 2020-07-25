from typing import Dict
import collections

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr

import shap
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

from src.generate_data import (
    generate_weights_gamma,
    get_correlated_data_stats,
    generate_normal_correlated_data,
    generate_normal_target
)
from src.calculate_importance import calculate_permutation_importance
from src.utils import rank_array


def run_experiment(experiment_params: Dict[str, any]) -> Dict[str, float]:
    experiment_results = collections.OrderedDict()

    # generate data
    data = generate_normal_correlated_data(
        mu=experiment_params["mu"],
        var=experiment_params["var"],
        n_features=experiment_params["n_features"],
        n_samples=experiment_params["n_samples"],
        noise_magnitude_loc=experiment_params["noise_magnitude_loc"],
        noise_magnitude_scale=experiment_params["noise_magnitude_scale"],
        seed=experiment_params["seed"]
    )
    data_stats = get_correlated_data_stats(data)
    for key, value in data_stats.items():
        experiment_results[f"corr_data_{key}"] = value

    # generate weights of features
    weights = generate_weights_gamma(
        n_features=data.shape[1],
        gamma=experiment_params["gamma"],
        scale=experiment_params["scale"],
        weights_range=experiment_params["weights_range"],
        seed=experiment_params["seed"]
    )
    expected_ranks = rank_array(-weights)

    # generate target
    y = generate_normal_target(
        data,
        weights,
        task=experiment_params["task"]
    )
    data = pd.DataFrame(data)

    # permutation importance
    model, score, importances, importance_ranks = calculate_permutation_importance(
        LGBMClassifier(**experiment_params["model_params"]),
        data, y,
        scoring_function=roc_auc_score,
        n_repeats=experiment_params["n_repeats_permutations"],
    )
    permutation_ranks_corr = spearmanr(expected_ranks, importance_ranks)[0]
    experiment_results["model_roc_auc"] = score
    experiment_results["permutation_ranks_corr"] = permutation_ranks_corr

    # shap
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
    shap_values = explainer.shap_values(data)[1]  # class 1 SHAP values
    shap_values = abs(shap_values)
    shap_fe = shap_values.sum(axis=0)
    shap_ranks_corr = spearmanr(expected_ranks, -shap_fe)[0]
    experiment_results["shap_ranks_corr"] = shap_ranks_corr

    # gain
    model_fe = model.booster_.feature_importance(importance_type='gain')
    gain_ranks_corr = spearmanr(expected_ranks, -model_fe)[0]
    experiment_results["gain_ranks_corr"] = gain_ranks_corr
    return experiment_results


def main(
        num_seeds: int = 3,
        results_save_path: str = "./data/experiment_results.csv"
) -> None:
    # create params for experiments
    experiments_grid = ParameterGrid(
        {
            "task": ["classification"],

            # constant params - data generation
            "mu": [1],
            "var": [1],
            "n_features": [50],
            "n_samples": [10000],

            # constant params - weights
            "gamma": [1],
            "scale": [1],
            "weights_range": [(0.2, 1)],

            # importance params
            "model_params": [
                {"learning_rate": 0.01, "n_estimators": 100, "random_state": 42}
            ],
            "n_repeats_permutations": [5],
            "shap_data_sample": [100],

            # changeable params
            "seed": list(range(num_seeds)),
            "noise_magnitude_loc": np.arange(0, 11, 1),
            "noise_magnitude_scale": np.arange(1, 3, 1),
        }
    )
    experiments_grid = list(experiments_grid)

    # run experiments
    results = []
    for experiment_params in tqdm(experiments_grid):
        experiment_results = run_experiment(experiment_params)
        results.append(experiment_results)

    # save
    results = pd.DataFrame(results)
    results.to_csv(results_save_path, index=False)


if __name__ == "__main__":
    main(num_seeds=5, results_save_path="./data/experiment_results.csv")
