from typing import Dict
import collections
from multiprocessing import Pool

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr

import shap
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMModel

from src.generate_data import (
    generate_weights_gamma,
    get_correlated_data_stats,
    generate_normal_correlated_data,
    generate_normal_target
)
from src.calculate_importance import (
    calculate_drop_and_relearn_importance,
    calculate_permutation_importance,
    calculate_permute_and_relearn_importance,
)
from src.metrics import negative_mean_squared_error
from src.utils import rank_array


def run_experiment(
        experiment_params: Dict[str, any]
) -> Dict[str, float]:
    experiment_results = collections.OrderedDict()

    # generate data
    data = generate_normal_correlated_data(
        mu=experiment_params["mu"],
        var=experiment_params["var"],
        n_features=experiment_params["n_features"],
        n_samples=experiment_params["n_samples"],
        max_correlation=experiment_params["max_correlation"],
        noise_magnitude_max=experiment_params["noise_magnitude_max"],
        seed=experiment_params["seed"]
    )

    # get data's correlation statistics
    data_stats = get_correlated_data_stats(data)
    for key, value in data_stats.items():
        experiment_results[f"corr_data_{key}"] = value

    # generate weights of features
    weights = generate_weights_gamma(
        n_features=data.shape[1],
        gamma=experiment_params["gamma"],
        scale=experiment_params["scale"],
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
    model, score, importances, importances_ranks = calculate_permutation_importance(
        LGBMModel(**experiment_params["model_params"]),
        data, y,
        scoring_function=experiment_params["metric"],
        n_repeats=experiment_params["n_repeats_permutations"],
    )
    permutation_ranks_corr = spearmanr(expected_ranks, importances_ranks)[0]
    experiment_results["model_roc_auc"] = score
    experiment_results["permutation_ranks_corr"] = permutation_ranks_corr

    # shap
    explainer = shap.TreeExplainer(model.booster_, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(data)
    if len(shap_values) == 2:  # 2 - list of 2 elements for classification, select class 1
        shap_values = shap_values[1]
    shap_values = abs(shap_values)
    shap_fe = shap_values.sum(axis=0)
    shap_ranks_corr = spearmanr(expected_ranks, -shap_fe)[0]
    experiment_results["shap_ranks_corr"] = shap_ranks_corr

    # gain
    model_fe = model.booster_.feature_importance(importance_type='gain')
    gain_ranks_corr = spearmanr(expected_ranks, -model_fe)[0]
    experiment_results["gain_ranks_corr"] = gain_ranks_corr

    if experiment_params["apply_relearn"]:
        # drop and relearn
        _, _, _, importances_ranks_drop = calculate_drop_and_relearn_importance(
            LGBMModel(**experiment_params["model_params"]),
            data, y,
            scoring_function=experiment_params["metric"],
        )
        drop_and_relearn_ranks_corr = spearmanr(expected_ranks, importances_ranks_drop)[0]
        experiment_results["drop_and_relearn_ranks_corr"] = drop_and_relearn_ranks_corr

        # permute and relearn
        _, _, _, importances_ranks_permute = calculate_permute_and_relearn_importance(
            LGBMModel(**experiment_params["model_params"]),
            data, y,
            scoring_function=experiment_params["metric"],
        )
        permute_and_relearn_ranks_corr = spearmanr(expected_ranks, importances_ranks_permute)[0]
        experiment_results["permute_and_relearn_ranks_corr"] = permute_and_relearn_ranks_corr
    return experiment_results


def main(
        num_seeds: int = 3,
        results_save_path: str = "./data/experiment_results.csv"
) -> None:
    # create params for experiments
    experiments_grid = ParameterGrid(
        {
            "task": ["classification"],
            "apply_relearn": [False],

            # constant params - data generation
            "mu": [0],
            "var": [1],
            "n_features": [50],
            "n_samples": [10_000],

            # constant params - weights
            "gamma": [1],
            "scale": [1],

            # permutation params
            "metric": [roc_auc_score],  # "negative_mean_squared_error" for regression, "roc_auc_score" for classification
            "model_params": [
                {
                    "objective": "binary",  # "regression" for regression, "binary" for classification
                    "learning_rate": 0.01,
                    "n_estimators": 100,
                    "random_state": 42
                }
            ],
            "n_repeats_permutations": [5],

            # changeable params
            "max_correlation": [0.95, 0.9, 0.8, 0.7, 0.6, 0.5],
            "noise_magnitude_max": np.arange(1, 5, 1),
            "seed": list(range(num_seeds)),
        }
    )
    experiments_grid = list(experiments_grid)

    # run experiments
    results = []
    for grid in tqdm(experiments_grid):
        results.append(run_experiment(grid))

    # save
    results = pd.DataFrame(results)
    results.to_csv(results_save_path, index=False)


if __name__ == "__main__":
    main(num_seeds=50, results_save_path="./data/experiment_results_no_relearn.csv")
