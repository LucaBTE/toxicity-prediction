# Setup Environment
## How to setup the environment
To set up the environment for this project, we have created a Conda environment with all the necessary dependencies. Below are the steps to create and activate the environment:
1. **Install Conda**: If you don't have Conda installed, you can download and install it from the [official Anaconda website](https://www.anaconda.com/products/distribution).

2. **Navigate to the Project Directory**: Open your terminal or Anaconda Prompt and navigate to the root directory of the project.

3. **Create the Conda Environment**: Run the following command to create a new Conda environment named `toxicity-prediction` with Python 3.10:
   ```bash
   conda env create -f ./environment_setup/environment.yml
   ```

4. **Activate the Environment**: After the environment is created, activate it using the following command:
   ```bash
    conda activate tox_prediction
    ```
5. **Register the Environment as a Jupyter Kernel**: To use this environment in Jupyter Notebooks, you need to register it as a kernel. Run the following command:
    ```bash
    python -m ipykernel install --user --name tox_prediction --display-name "Python (Tox_Project)"
    ```

## Key Adjustments Made
- **Python 3.10 Selection**: Selected 3.10 to ensure stability for DeepPurpose and descriptastorus, which rely on specific C++ wrappers and older numpy headers.

- **Dependency Locking**: Fixed setuptools to 69.5.1 to avoid installation conflicts common with legacy chemistry packages.

- **Hybrid Installation**: Combined Conda (for heavy binaries like RDKit and PyTorch) with Pip (for specialized ML4DD libraries) to ensure all tools communicate correctly.

- **Explainability Ready**: Pre-installed exmol and skunk so you can generate counterfactuals and visual explanations immediately after training.


> To update the environment with new dependencies, simply add them to the `environment.yml` file and run `conda env update -n tox_prediction --file ./environment_setup/environment.yml --prune` to apply the changes.