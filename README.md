# Toxicity Prediction - Group 05 ML4DD

Machine Learning for Drug Design project on acute toxicity prediction using
the TDC `LD50_Zhu` task.

The goal is to predict continuous acute oral toxicity values (`LD50`) from
molecular structure. The project combines data cleaning, descriptor generation,
exploratory analysis, feature selection, classical machine learning models, and
model interpretability.

Task reference: <https://tdcommons.ai/single_pred_tasks/tox/>

## Authors

- [Abate Luca](https://github.com/LucaBTE)
- [Bernacchia Alessia](https://github.com/AlessiaBernacchia)
- [Pioda Tommaso](https://github.com/Thetommigun432)
- [Villani Giacomo](https://github.com/DownToTheGround)

## Repository Structure

```text
.
├── docs/                       # LaTeX report and report figures
├── environment_setup/          # Conda environment definition and setup notes
├── notebooks/
│   ├── data_collection/        # TDC import and descriptor generation
│   ├── data_cleaning/          # SMILES validation, duplicate checks, cleaning
│   ├── data_exploration/       # Basic and advanced EDA
│   ├── feature_selection/      # Spearman, Random Forest, LASSO, PFI selection
│   └── ml-models/              # Ridge, SVM, Random Forest, XGBoost, ensemble
├── scripts/                    # Utility scripts for rerunning notebooks
├── utils/                      # Shared Python helpers
└── setup.py                    # Local package installation
```

Datasets, model artifacts, and notebook outputs are intentionally not
tracked when they are large or reproducible.

## Environment Setup

Use the Conda environment provided in the repository. From the project root:

```bash
conda env create -f environment_setup/environment.yml
conda activate tox_prediction
python -m ipykernel install --user --name tox_prediction --display-name "Python (Tox_Project)"
python -m pip install -e .
```

The editable install makes the local `utils` package available inside the
notebooks. If imports behave strangely after changing helper code, restart the
Jupyter kernel.

To update an existing environment after `environment.yml` changes:

```bash
conda env update -n tox_prediction --file environment_setup/environment.yml --prune
```

## How To Run The Whole Project

Run the notebooks from the repository root using the `tox_prediction` environment
and the `Python (Tox_Project)` Jupyter kernel.

The complete pipeline order is:

1. `notebooks/data_collection/00_import_data.ipynb`
2. `notebooks/data_cleaning/01_cleaning.ipynb`
3. `notebooks/data_exploration/01_basic_exploration.ipynb`
4. `notebooks/data_exploration/02_advanced_exploration.ipynb`
5. `notebooks/feature_selection/feature_selection.ipynb`
6. `notebooks/ml-models/model_ridge.ipynb`
7. `notebooks/ml-models/model_random_forest.ipynb`
8. `notebooks/ml-models/model_svm.ipynb`
9. `notebooks/ml-models/model_xgboost.ipynb`
10. `notebooks/ml-models/model_weighted_ensemble.ipynb`
11. `notebooks/ml-models/interpratability/model_interpretability.ipynb`

The data collection notebook creates the raw/processed descriptor tables used by
later stages. Cleaning and feature selection export intermediate CSV files under
`data/`, which is ignored by git. The model notebooks create `outcome/` folders
with predictions, plots, metadata, and serialized models.

## Rerunning Model Outcomes

After data preparation and feature selection have been run once, the modeling
notebooks can be rerun from the command line:

```bash
python scripts/rerun_outcomes.py --scope ml -y
```

To rerun all supported modeling notebooks, including deep-learning notebooks if
present in the checkout:

```bash
python scripts/rerun_outcomes.py --scope all -y
```

Useful options:

```bash
python scripts/rerun_outcomes.py --dry-run
python scripts/rerun_outcomes.py --scope ml --skip-delete
python scripts/rerun_outcomes.py --scope ml --skip-interpretability
```

The script removes existing `notebooks/**/outcome` folders by default, then
executes the selected notebooks in order with `jupyter nbconvert --execute`.

## Report

The final report source is:

```text
docs/report.tex
```

To rebuild the PDF:

```bash
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
```

LaTeX auxiliary files such as `.aux`, `.log`, `.fls`, and `.synctex.gz` are
ignored by git. The report figures are stored in `docs/src/`.

## Notes

- Some notebooks may take a long time because descriptors, Optuna tuning, SHAP,
  and model refits are computationally expensive.
- GPU acceleration is attempted where supported, but the classical ML notebooks
  are designed to fall back to CPU execution.
- The held-out test split is used only for final evaluation; tuning uses internal
  training splits and validation monitoring.
- If a notebook cannot find generated data, rerun the earlier notebooks in the
  pipeline order above.
