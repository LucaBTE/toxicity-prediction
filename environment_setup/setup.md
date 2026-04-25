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

6. **Install Project**: To enable internal module imports and manage local utilities effectively, install the project using the `setup.py` file:
- if you are a developer and want to work on the codebase, use editable mode:
   ```bash
   python -m pip install -e .
   ```
- if you just want to use the project without modifying the code, use regular installation:
   ```bash
    python -m pip install .
    ```

7. **Launch Jupyter Notebook**: You can now launch Jupyter Notebook to start working on the project.

## Key Adjustments Made
- **Python 3.10 Selection**: Selected 3.10 to ensure stability for DeepPurpose and descriptastorus, which rely on specific C++ wrappers and older numpy headers.

- **Dependency Locking**: Fixed setuptools to 69.5.1 to avoid installation conflicts common with legacy chemistry packages.

- **Hybrid Installation**: Combined Conda (for heavy binaries like RDKit and PyTorch) with Pip (for specialized ML4DD libraries) to ensure all tools communicate correctly.

- **Explainability Ready**: Pre-installed exmol and skunk so you can generate counterfactuals and visual explanations immediately after training.

- **Editable Package Setup**: Integrated a setup.py file to treat the project directory as a local package. This allows you to import custom functions (e.g., from descriptors.ipynb) across different notebooks without modifying sys.path.

> To update the environment with new dependencies, simply add them to the `environment.yml` file and run `conda env update -n tox_prediction --file ./environment_setup/environment.yml --prune` to apply the changes.

> In theory each time you modify the codebase, you should be able to see the changes reflected in the notebooks without needing to reinstall the package, as long as you are running the notebooks within the activated `tox_prediction` environment. But in practice sometimes you might need to restart the kernel or re-import the modules to see the changes, especially if you are modifying functions that have already been imported in the notebook, because the IDE tends to use the chached data.

> When you create a new `.py` with functions to import in other notebooks, make sure to add an `__init__.py` file in the same directory to ensure it is recognized as a package. This way you can import the functions using standard Python import statements without needing to modify sys.path.