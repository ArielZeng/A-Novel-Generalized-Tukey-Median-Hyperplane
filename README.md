# A-Novel-Generalized-Tukey-Median-Hyperplane

# Odd- and Even-Dimensional Tukey Median Hyperplane Regression

This repository provides the implementation and experimental code for the OTH (odd-dimensional) and ETH (even-dimensional) regression models proposed in the accompanying paper.

The code is intended to reproduce the experimental results reported in the paper and is organized according to the dimensionality of the regression problem and experimental settings.

---

## Directory Structure

The repository is organized as follows:

- `data/`  
  Contains all datasets used in the experiments, provided in CSV format.

- `experiments/3d/`  
  Scripts for three-dimensional regression experiments using the proposed OTH model.

- `experiments/4d/`  
  Scripts for four-dimensional regression experiments using the proposed ETH model.

- `experiments/outlier/`  
  Scripts for robustness experiments under contaminated settings with injected outliers.

---

## Datasets

The experiments are conducted on several real-world benchmark datasets, including:

- Abalone
- Boston Housing
- Energy Efficiency
- Fish Market

All datasets are provided in CSV format under the `data/` directory.  
The original sources of the datasets are publicly available and are cited in the paper.

---

## Experimental Scripts

Each script loads the corresponding dataset from the `data/` directory, fits the proposed regression model, and reports performance metrics such as MAE and R².

- **3D experiments** (`experiments/3d/`):  
  Evaluate the proposed OTH model in three-dimensional regression settings.

- **4D experiments** (`experiments/4d/`):  
  Evaluate the proposed ETH model in four-dimensional regression settings.

- **Outlier experiments** (`experiments/outlier/`):  
  Evaluate robustness under contaminated settings with artificial outliers.


Reproducibility

All experiments are deterministic and can be reproduced by directly running the provided scripts.

Relation to the Paper

This repository accompanies the paper:

A Novel Generalized Tukey Median Hyperplane: Odd- and Even-Dimensional Formulations with Enhanced Outlier Robustness

The scripts correspond to the experimental results reported in the paper.

License

This project is released for academic and research purposes.
