from setuptools import setup, find_packages

setup(
    name="toxicity_prediction",
    version="0.1.0",
    description="A project for predicting chemical toxicity (LD50) using RDKit, Mordred, and DeepPurpose.",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # Core Data Science
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "tqdm",
        
        # Machine Learning & Chemistry
        "scikit-learn",
        "joblib",
        "pytdc",
        "missingno",
        
        # Note: Heavy binary dependencies like 'rdkit' and 'pytorch' 
        # are best managed via Conda as per your environment.yml
    ],
)