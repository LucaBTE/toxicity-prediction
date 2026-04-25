import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
import numpy as np

def spearman_corr_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the pairwise Spearman correlation matrix for all columns of X.

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
        Feature matrix. All columns must be numeric.

    Returns
    -------
    corr : pd.DataFrame of shape (n_features, n_features)
        Symmetric matrix of Spearman correlation coefficients in [-1, 1].
    """
    corr, _ = spearmanr(X)
    return pd.DataFrame(corr, index=X.columns, columns=X.columns)


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


def forward_selection(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_features: int = 10,
    cv: int = 5,
) -> list:
    """
    Greedy forward feature selection using cross-validated R² as criterion.

    At each round the feature that maximises CV R² when added to the
    current selected set is chosen. Selection stops when max_features
    is reached.

    Parameters
    ----------
    X_train : pd.DataFrame of shape (n_samples, n_features)
    y_train : pd.Series of shape (n_samples,)
    max_features : int
        Maximum number of features to select.
    cv : int
        Number of cross-validation folds.
    random_state : int

    Returns
    -------
    selected : list of str
        Ordered list of selected feature names.
    scores : list of float
        CV R² score after adding each selected feature.
    """
    remaining = list(X_train.columns)
    selected  = []
    scores    = []

    while len(selected) < max_features and remaining:
        best_score   = -np.inf
        best_feature = None

        for feature in remaining:
            candidate = selected + [feature]
            cv_scores = cross_val_score(
                model, X_train[candidate], y_train,
                cv=cv, scoring='r2',
            )
            score = cv_scores.mean()
            if score > best_score:
                best_score   = score
                best_feature = feature

        selected.append(best_feature)
        remaining.remove(best_feature)
        scores.append(best_score)

    return selected, scores


def clustered_forward_selection(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    corr_matrix: pd.DataFrame,
    corr_threshold: float = 0.7,
    max_features: int = 10,
    cv: int = 5,
) -> list:
    """
    Clustered forward feature selection.

    Features are first grouped by correlation using a threshold on the
    absolute Spearman correlation matrix. At each round only one candidate
    per cluster is considered (the one not yet selected from that cluster).
    The candidate that maximises CV R² is added to the selected set and its
    cluster is locked — no further candidates are drawn from it until all
    other clusters have contributed a feature.

    Parameters
    ----------
    X_train : pd.DataFrame of shape (n_samples, n_features)
    y_train : pd.Series of shape (n_samples,)
    corr_matrix : pd.DataFrame
        Precomputed Spearman correlation matrix (from spearman_corr_matrix).
    corr_threshold : float
        Features with |correlation| >= corr_threshold are placed in the
        same cluster.
    max_features : int
        Maximum number of features to select.
    cv : int
        Number of cross-validation folds.
    random_state : int

    Returns
    -------
    selected : list of str
        Ordered list of selected feature names.
    scores : list of float
        CV R² score after adding each selected feature.
    cluster_labels : dict
        Mapping from feature name to cluster id.
    """
    features = list(X_train.columns)
    abs_corr = corr_matrix.abs()

    # Step 1: assign each feature to a cluster
    cluster_labels = {}
    cluster_id     = 0
    unassigned     = set(features)

    for feat in features:
        if feat not in unassigned:
            continue
        correlated = set(
            abs_corr.index[abs_corr[feat] >= corr_threshold].tolist()
        ) & unassigned
        for f in correlated:
            cluster_labels[f] = cluster_id
        unassigned -= correlated
        cluster_id += 1

    # Step 2: greedy forward selection with one candidate per cluster
    selected         = []
    scores           = []
    locked_clusters  = set()

    while len(selected) < max_features:
        available_clusters = [
            cid for cid in set(cluster_labels.values())
            if cid not in locked_clusters
        ]
        if not available_clusters:
            break

        # one candidate per available cluster
        candidates = []
        for cid in available_clusters:
            cluster_features = [
                f for f, c in cluster_labels.items()
                if c == cid and f not in selected
            ]
            if not cluster_features:
                continue

            best_cluster_score   = -np.inf
            best_cluster_feature = None
            for f in cluster_features:
                score    = cross_val_score(
                    model, X_train[[f]], y_train,
                    cv=cv, scoring='r2',
                ).mean()
                if score > best_cluster_score:
                    best_cluster_score   = score
                    best_cluster_feature = f
            if best_cluster_feature is not None:
                candidates.append(best_cluster_feature)

        if not candidates:
            break

        # evaluate each candidate in context and pick the best one
        best_score   = -np.inf
        best_feature = None

        for feature in candidates:
            candidate_set = selected + [feature]
            cv_score      = cross_val_score(
                model, X_train[candidate_set], y_train,
                cv=cv, scoring='r2',
            ).mean()
            if cv_score > best_score:
                best_score   = cv_score
                best_feature = feature

        if best_feature is None:
            break

        selected.append(best_feature)
        scores.append(best_score)
        locked_clusters.add(cluster_labels[best_feature])

    return selected, scores, cluster_labels