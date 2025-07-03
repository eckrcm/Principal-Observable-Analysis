# Principal-Observable-Analysis

Principal Observable Analysis (POA) is a novel method for vectorization, dimension reduction, and visualization of metric measure spaces using the principle of maximizing variance (analogous to PCA) over 1-Lipschitz scalar fields. 

POA was recently introduced in the paper [Observable Covariance and Principal Observable Analysis](https://arxiv.org/abs/2506.04003) 
by Ece Karacam, Washington Mio, and Osman Berat Okutan.

This repository contains the code for POA and the notebooks to generate the figures in the paper. 

### Repository structure
  - **Notebooks/** – Jupyter notebooks, one per figure.
  - **poa_utils.py** – Main Python file with POA functions.
  - **environment.yml** / **requirements.txt** – Environment setup.
  

To run the notebooks and code in this repository, follow these steps:

1. Clone the repository  
```bash
git clone https://github.com/eckrcm/Principal-Observable-Analysis.git
cd Principal-Observable-Analysis
```
2. (Recommended) Create and activate the conda environment
```bash
conda env create -f environment.yml
conda activate poa-env
```
3. Start Jupyter Notebook or Lab
```bash
jupyter notebook
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
