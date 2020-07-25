from typing import Dict, Callable, Tuple
import collections

import numpy as np
import pandas as pd
import sklearn

from .utils import rank_array


def calculate_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    scoring_function: Callable = sklearn.metrics.roc_auc_score,
    n_repeats: int = 3,
    seed: int = 42,
) -> Tuple[any, float, Dict[str, float], np.array]:
    """
    Example of permutation importance calculation (regression).
    :param model: sklearn model, or any model with `fit` and `predict` methods
    :param X: input features
    :param y: input target
    :param scoring_function: function to use for scoring, should output single float value
    :param n_repeats: how many times make permutation
    :param seed: random state for experiment reproducibility
    :return:
    """
    # step 1 - train model
    model.fit(X, y)

    # step 2 - make predictions for train data and score (higher score - better)
    y_hat_no_shuffle = model.predict(X)
    score = scoring_function(*(y, y_hat_no_shuffle))

    # step 3 - calculate permutation importance
    features = X.columns
    items = [(key, 0) for key in features]
    importances = collections.OrderedDict(items)

    for n in range(n_repeats):
        for col in features:
            # copy data to avoid using previously shuffled versions
            X_temp = X.copy()

            # shuffle feature_i values
            X_temp[col] = X[col].sample(X.shape[0], replace=True, random_state=seed+n).values

            # make prediction for shuffled dataset
            y_hat = model.predict(X_temp)

            # calculate score
            score_permuted = scoring_function(*(y, y_hat))

            # calculate delta score
            # better model <-> higher score
            # lower the delta -> more important the feature
            delta_score = score_permuted - score

            # save result
            importances[col] += delta_score / n_repeats

    importances_values = np.array(list(importances.values()))
    importance_ranks = rank_array(importances_values)
    return model, score, importances, importance_ranks
