import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from rdkit.Chem.SaltRemover import SaltRemover
import missingno as msno
from rdkit import Chem
# import display of notebooks
from IPython.display import display


def plot_samples_distribution(train, valid, test, save_dir=None):
    """
    Plots the sample distribution across Train, Validation, and Test sets.
    Displays absolute counts and percentages.
    """
    # 1. Prepare data
    split_names = ['Train', 'Validation', 'Test']
    split_counts = [len(train), len(valid), len(test)] 

    total_samples = sum(split_counts)
    percentages = [(count / total_samples) * 100 for count in split_counts]

    # 2. Setup Plot
    plt.figure(figsize=(8, 6))
    colors = ['#4C72B0', '#F39C12', '#C44E52']  # Professional color palette
    bars = plt.bar(split_names, split_counts, color=colors, edgecolor='black', alpha=0.8)

    # 3. Add Labels (Count and Percentage)
    for bar, count, percent in zip(bars, split_counts, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + (max(split_counts) * 0.02), 
                f'{count}\n({percent:.1f}%)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 4. Formatting
    plt.title('Sample Distribution across Dataset Splits', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xlabel('Dataset Split', fontsize=12)
    plt.ylim(0, max(split_counts) * 1.2)  # Extra vertical space for labels
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 5. Handle Saving
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        file_path = save_dir / 'split_distribution.png'
        plt.savefig(file_path, dpi=300)
        print(f"✓ Plot saved to: {file_path}")

def add_molecule_column(df):
    """
    Adds a 'mol' column to the DataFrame by parsing the 'smiles' column using RDKit.
    Handles invalid SMILES by setting 'mol' to NaN and printing a warning.
    """
    def parse_smiles(smiles):
        try:
            return Chem.MolFromSmiles(smiles)
        except Exception as e:
            print(f"Error parsing SMILES: {smiles} - {e}")
            return None

    df['mol'] = df['canonical_smiles'].apply(parse_smiles)
    return df

def handle_invalid_smiles(df):
    """
    Checks for invalid SMILES entries where RDKit failed to parse and reports them.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    
    # Check for molecules that RDKit failed to parse
    invalid_mols = df[df['mol'].isna()]
    print(f"Number of invalid SMILES: {len(invalid_mols)}")

    if len(invalid_mols) > 0:
        print("Invalid SMILES:")
        print(invalid_mols[['smiles', 'Y']])
        df = df.dropna(subset=['mol'])  # Remove invalid molecules

    return df

def check_missing_data(df):
    """
    Visualizes missing values in the dataset.
    Shows at least the first 10 columns, plus any others that have missing data.
    """
    # 1. Identify columns with missing values
    cols_with_missing = df.columns[df.isnull().any()].tolist()
    
    # 2. Ensure we have at least 10 columns for the visualization
    # We take the first 10 columns of the dataframe as a baseline
    base_cols = df.columns[:10].tolist()
    
    # 3. Combine them using a set to avoid duplicates and convert back to list
    # This ensures we see the first 10 AND any column that is actually missing data
    cols_to_show = list(dict.fromkeys(base_cols + cols_with_missing))
    
    print(f"\tVisualizing {len(cols_to_show)} columns.")
    print(f"\tMissing values summary:\n{df[cols_with_missing].isnull().sum()}")
    
    # 4. Generate the matrix plot
    # We use the subset of columns defined above
    msno.matrix(df[cols_to_show])
    plt.title("Missing Data Matrix (Subset Selection)", fontsize=16)
    plt.show()

    # msno.heatmap(df)
    # plt.show()
    
    return df

def check_missing_values(df):
    """
    Identifies and prints only the columns that contain missing (NaN) values.
    """
    # 1. Calculate the sum of missing values for all columns
    missing_counts = df.isna().sum()
    
    # 2. Filter to keep only columns where the count is greater than 0
    only_missing = missing_counts[missing_counts > 0]
    
    if only_missing.empty:
        print("No missing values found in any column.")
    else:
        print("Columns with missing values:")
        print(only_missing)

def check_missing_values_in(df, column_name='Y'):
    """
    Checks for missing values in a specific column and prints the count.
    """
    missing_counts = df[column_name].isna().sum()
    print(f"Missing values in column '{column_name}':")
    print(missing_counts)

def check_mol_target_inconsistencies(df, smiles_col='canonical_smiles', target_col='Y', show_lines=True):
    """ 
    Checks for inconsistencies between SMILES strings and their associated toxicity labels.
    """
    
    # Group by SMILES and check for unique toxicity labels
    inconsistencies = df.groupby(smiles_col)[target_col].nunique()
    
    # Filter for cases where there are multiple unique toxicity labels for the same SMILES
    inconsistent_smiles = inconsistencies[inconsistencies > 1].index.tolist()
    
    if inconsistent_smiles:
        print(f"Found {len(inconsistent_smiles)} inconsistent SMILES with multiple toxicity labels")
        for smi in inconsistent_smiles:
            if show_lines:
                print(f"SMILES: {smi}")
                display(df[df[smiles_col] == smi][[target_col]])
    else:
        print("No inconsistencies found between SMILES and toxicity labels.")

def strip_salts_and_report(df, smiles_col='canonical_smiles'):
    """
    Strips salts from molecules and reports how many structures were modified.
    """
    remover = SaltRemover()
    
    # 1. Identify which molecules currently have salts
    # We check if the SMILES string contains a '.' (indicating multiple components)
    had_salts_mask = df[smiles_col].str.contains(r'\.', na=False)
    num_with_salts = had_salts_mask.sum()
    
    # 2. Apply salt stripping
    # dontRemoveEverything=True prevents returning an empty object if the input is only a salt
    df['mol'] = df['mol'].apply(lambda x: remover.StripMol(x, dontRemoveEverything=True))
    
    # 3. Update the SMILES strings to reflect the neutral parent structure
    df[smiles_col] = df['mol'].apply(Chem.MolToSmiles)
    
    # 4. Report findings
    print(f"\tFound and stripped salts in {num_with_salts} molecules.")
    if num_with_salts > 0:
        # Show an example of a stripped molecule (Optional)
        example_idx = df[had_salts_mask].index[0] if num_with_salts > 0 else None
        if example_idx is not None:
             print(f"\tExample stripped: Index {example_idx}")
             
    return df


def remove_duplicates(df):
    """
    Removes duplicate molecules based on the SMILES column and reports how many were removed.
    """
    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)
    
    num_removed = initial_count - final_count
    print(f"\tRemoved {num_removed} duplicate molecules.")
    
    return df