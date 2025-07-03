# Principal-Observable-Analysis

Principal Observable Analysis (POA) is a novel method for vectorization, dimension reduction, and visualization of metric measure spaces using the principle of maximizing variance (analogous to PCA) over 1-Lipschitz scalar fields. POA was recently introduced in the paper [Observable Covariance and Principal Observable Analysis](https://arxiv.org/abs/2506.04003) by Ece Karacam, Washington Mio, and Osman Berat Okutan.

This repository contains the code for POA and the notebooks to generate the figures in the paper. 

## Repository structure
- **Notebooks/** – Jupyter notebooks, one per figure.
- **poa_utils.py** – Main Python file with POA functions.
- **environment.yml** / **requirements.txt** – Environment setup.

## Getting Started

To run the notebooks and code in this repository, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/eckrcm/Principal-Observable-Analysis.git
cd Principal-Observable-Analysis
```

### 2. (Recommended) Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate poa-env
```

### 3. (Alternative) Using pip and requirements.txt

If you prefer pip/venv:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Start Jupyter Notebook or Lab

```bash
jupyter notebook
```
or
```bash
jupyter lab
```
Open the desired notebook from the `notebooks/` folder in your browser.

## Notes

- Make sure to select the correct Python interpreter/environment in VS Code or Jupyter.
- If you are running notebooks from the `Notebooks/` folder and **not** from the project root, add the project root to your Python path before importing `poa_utils`:
  ```python
  import sys
  sys.path.append('..')
  import poa_utils
- If you modify `poa_utils.py` while working in a notebook and have already imported it in the same session, reload it using:
  ```python
  import importlib
  import poa_utils
  importlib.reload(poa_utils)
  ```
- If you get `ModuleNotFoundError` for any package, install it with `conda install <package>` or `pip install <package>`.

## Troubleshooting

- For `dccp` issues, check your Python version or use a conda environment.
- For more help, see [dccp GitHub Issues](https://github.com/cvxgrp/dccp/issues).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
