# Modeling Notebooks

This folder contains the modeling layer for the toxicity prediction project.
The target variable is `Y`, and the main molecular input column for deep
learning models is `canonical_smiles`.

## Structure

- `ml-models/`: classical machine-learning notebooks.
- `dl-models/`: DeepPurpose deep-learning notebooks.
- shared modeling code lives in `../utils/modeling.py`.

The old duplicated helpers from `notebooks/modeling`, `ml-models/pipeline.py`,
`ml-models/storage.py`, `dl-models/schema.py`, and `dl-models/storage.py` were
removed. All common logic now has one source of truth.

## Shared Utilities

`utils/modeling.py` provides only the code needed by the current notebooks:

- loading `train.csv`, `valid.csv`, and `test.csv`;
- preparing tabular ML arrays;
- preparing SMILES/target splits for DeepPurpose;
- creating artifact paths;
- computing regression metrics;
- saving predictions;
- plotting predicted vs real values;
- plotting feature importance when available.

## ML Models

The ML notebooks are standardized:

- `model_ridge.ipynb`
- `model_random_forest.ipynb`
- `model_svm.ipynb`
- `model_xgboost.ipynb`

Each notebook follows the same workflow:

1. load processed data through `utils.modeling`;
2. train the model;
3. predict on the test split;
4. save the trained model;
5. save test predictions;
6. save a predicted-vs-real plot;
7. save feature importance when possible.

Artifacts are written under `notebooks/ml-models/outcome/`.

## DL Models

The DL notebooks use DeepPurpose directly on SMILES:

- `model_deeppurpose_gnn.ipynb`: graph model with `DGL_GCN`;
- `model_deeppurpose_cnn.ipynb`: sequence model with `CNN`.

Each notebook saves:

- model artifacts;
- test predictions;
- predicted-vs-real plot;
- metadata with run id, encoding and metrics.

Artifacts are written under `notebooks/dl-models/outcome/`.

## Notes

The notebooks assume they are executed from their own folder, matching the
existing project convention. Heavy dependencies such as DeepPurpose, RDKit,
PyTorch and XGBoost should be installed through the project environment.
