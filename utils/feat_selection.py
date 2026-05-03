import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score


def permutation_importance_df(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
    predict_fn=None,
) -> pd.DataFrame:
    """
    Compute permutation feature importance and return a tidy DataFrame.

    Parameters
    ----------
    model : fitted sklearn estimator
    X : pd.DataFrame of shape (n_samples, n_features)
    y : pd.Series of shape (n_samples,)
    n_repeats : int
        Number of times each feature is permuted.
    random_state : int

    Returns
    -------
    df_imp : pd.DataFrame
        Columns: 'feature', 'importance_mean', 'importance_std'.
        Sorted by importance_mean descending.
    """
    if predict_fn is None:
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring='r2',
        )
        importances_mean = result.importances_mean
        importances_std = result.importances_std
    else:
        rng = np.random.default_rng(random_state)
        baseline_score = r2_score(y, predict_fn(model, X))
        importances = []

        for feature in X.columns:
            feature_importances = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                X_permuted[feature] = rng.permutation(X_permuted[feature].to_numpy())
                score = r2_score(y, predict_fn(model, X_permuted))
                feature_importances.append(baseline_score - score)
            importances.append(feature_importances)

        importances = np.asarray(importances)
        importances_mean = importances.mean(axis=1)
        importances_std = importances.std(axis=1)

    df_imp = pd.DataFrame({
        'feature':          X.columns,
        'importance_mean':  importances_mean,
        'importance_std':   importances_std,
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
    return df_imp
