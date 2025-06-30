This work was conducted at Université Paris-Saclay, CentraleSupélec, ENS Paris-Saclay, CNRS, LMPS – Laboratoire de Mécanique Paris-Saclay, Gif-sur-Yvette, France.

# Physics-Augmented Neural Networks for 3D unsupervised learning

This repository contains code and data for developing and using Physics-Augmented Neural Networks (PANN) aimed at modeling isotropic hyperelastic material behavior. It integrates TensorFlow neural network models with finite element analysis (FEA) and experimental displacement data obtained from DVC.

Code and data used in:

M. Bourdyot, M. Compans, R. Langlois, B. Smaniotto, E. Baranger, C. Jailin, **3D learning of a hyperelastic behavior with experimental data**, submitted to publication. 


[(NH residuals forces)](https://cjailin.github.io/html_outputs/3D_PANN_learning/NH_residual_forces.html)

[(PANN residuals forces)](https://cjailin.github.io/html_outputs/3D_PANN_learning/PANN_residual_forces.html)


## Repository Structure
This repo is organized with 2 independent tutorials and a common set of basic methods (PANN_lib).
- `PANN_lib/`: Core Python libraries for FEA functions, PANN, and traditional models, training, and visualization.
- `DVC_data/*`: CSV files (`coordinates_scan1.csv`, `connectivity_scan1.csv`, `displacements_scan*.csv`) with the experimental datasets.
- `3D_learning_core/`: Main scripts for model training and validation.

## Requirements
  - python >= 3.8
  - tensorflow >= 2.0
  - pandas
  - pyvista
  - numpy
  - matplotlib
  - seaborn
  - tensorboard (optional for monitoring)
  - gmsh (optional)

## Installation
Create a virtual environment and install the required libraries:
```bash
pip install tensorflow matplotlib numpy pandas pyvista seaborn tensorboard
```
or
```bash
conda install tensorflow matplotlib numpy pandas pyvista seaborn tensorboard
```

## Getting Started

```bash
cd tuto_exp_EUCLID_3D
python Main_exp_EUCLID.py
```

---

Feel free to contribute or open issues!
