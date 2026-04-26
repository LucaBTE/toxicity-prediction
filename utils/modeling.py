from pathlib import Path
import os
from typing import Any

import joblib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TARGET_COLUMN = "Y"
SMILES_COLUMN = "canonical_smiles"
ID_COLUMNS = ("Drug_ID", "Drug", SMILES_COLUMN, "mol")


def load_split_frames(data_dir: str | Path = PROCESSED_DATA_DIR) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir).expanduser().resolve()
    return {
        split: pd.read_csv(data_dir / file_name)
        for split, file_name in {
            "train": "train.csv",
            "valid": "valid.csv",
            "test": "test.csv",
        }.items()
    }


def load_tabular_arrays(data_dir: str | Path = PROCESSED_DATA_DIR) -> tuple:
    frames = load_split_frames(data_dir)
    X = {}
    y = {}
    for split, frame in frames.items():
        y[split] = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").to_numpy()
        drop_cols = [TARGET_COLUMN, *[col for col in ID_COLUMNS if col in frame.columns]]
        X[split] = frame.drop(columns=drop_cols)

    train_columns = list(X["train"].columns)
    if train_columns != list(X["valid"].columns) or train_columns != list(X["test"].columns):
        raise ValueError("Train, validation and test feature columns must match")

    bool_cols = X["train"].select_dtypes(include=["bool"]).columns.tolist()
    for split in X:
        X[split][bool_cols] = X[split][bool_cols].astype("int8")

    numeric_cols = X["train"].select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X["train"].select_dtypes(include=["object", "category"]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_cols),
        ],
        remainder="drop",
    )

    X_train = preprocessor.fit_transform(X["train"])
    X_valid = preprocessor.transform(X["valid"])
    X_test = preprocessor.transform(X["test"])

    return (
        X_train,
        X_valid,
        X_test,
        y["train"],
        y["valid"],
        y["test"],
        preprocessor.get_feature_names_out(),
    )


def load_smiles_splits(data_dir: str | Path = PROCESSED_DATA_DIR) -> dict[str, pd.DataFrame]:
    frames = load_split_frames(data_dir)
    for split, frame in frames.items():
        frames[split] = frame[[SMILES_COLUMN, TARGET_COLUMN]].dropna().copy()
    return frames


def artifact_paths(
    base_dir: str | Path,
    run_name: str,
    model_suffix: str,
    include_importance: bool = False,
    include_metadata: bool = False,
) -> dict[str, Path]:
    base_dir = Path(base_dir)
    slug = run_name.lower().replace(" ", "_")
    paths = {
        "model": base_dir / "outcome" / "models" / f"{slug}_model{model_suffix}",
        "predictions": base_dir / "outcome" / "predictions" / f"{slug}_predictions.csv",
        "pred_vs_real": base_dir / "outcome" / f"{slug}_pred_vs_real.png",
    }
    if include_importance:
        paths["importance"] = base_dir / "outcome" / "feature_importance" / f"{slug}_importance.png"
    if include_metadata:
        paths["metadata"] = base_dir / "outcome" / "metadata" / f"{slug}_metadata.json"
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    pd.DataFrame({"real": y_true, "prediction": y_pred}).to_csv(path, index=False)


def plot_pred_vs_real(path: Path, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="k")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", linewidth=2)
    plt.xlabel("Real Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_importance(path: Path, importance: np.ndarray, feature_names: np.ndarray, title: str) -> None:
    order = np.argsort(np.abs(importance))[::-1][:20]
    plt.figure(figsize=(10, 7))
    plt.barh(range(len(order)), importance[order], edgecolor="black")
    plt.yticks(range(len(order)), [str(feature_names[i]) for i in order])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def model_importance(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_).ravel()
    if hasattr(model, "coef_"):
        return np.asarray(model.coef_).ravel()
    result = permutation_importance(
        model,
        X_test[:1000],
        y_test[:1000],
        n_repeats=5,
        random_state=42,
        scoring="r2",
    )
    return np.asarray(result.importances_mean).ravel()


def save_ml_run(
    name: str,
    model: Any,
    paths: dict[str, Path],
    X_test: np.ndarray,
    y_test: np.ndarray,
    predictions: np.ndarray,
    feature_names: np.ndarray,
) -> dict[str, float]:
    predictions = np.asarray(predictions).ravel()
    metrics = regression_metrics(y_test, predictions)
    print(f"[{name}] RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R2: {metrics['r2']:.4f}")

    joblib.dump(model, paths["model"])
    save_predictions(paths["predictions"], y_test, predictions)
    plot_pred_vs_real(paths["pred_vs_real"], y_test, predictions, f"{name}: Predicted vs Real")

    importance = model_importance(model, X_test, y_test)
    if importance is not None and "importance" in paths:
        plot_importance(paths["importance"], importance, feature_names, f"{name} Feature Importance")

    return metrics
