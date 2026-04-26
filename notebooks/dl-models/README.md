# DL Models

Deep-learning experiments live next to the existing tabular ML notebooks.
Shared data loading, metrics and storage helpers live in `utils/modeling.py`.

Storage follows the same artifact convention used by `notebooks/ml-models`:

- `outcome/models/`: trained model weights
- `outcome/predictions/`: inference outputs
- `outcome/metadata/`: run metadata and metrics
- `outcome/*_pred_vs_real.png`: prediction diagnostic plots
