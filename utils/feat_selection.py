import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_df(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
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
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring='r2',
    )
    df_imp = pd.DataFrame({
        'feature':          X.columns,
        'importance_mean':  result.importances_mean,
        'importance_std':   result.importances_std,
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
    return df_imp


