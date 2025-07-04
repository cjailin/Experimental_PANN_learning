# Learning a hyperelastic constitutive model from 3D experimental data

This work was conducted at Université Paris-Saclay, CentraleSupélec, ENS Paris-Saclay, CNRS, LMPS – Laboratoire de Mécanique Paris-Saclay, Gif-sur-Yvette, France.

![Static Badge](https://img.shields.io/badge/Python-blue)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-GPL--3.0-lightgrey.svg)](LICENSE.md)


This repository contains code and data for developing and using Physics-Augmented Neural Networks (PANN) aimed at modeling isotropic hyperelastic material behavior. It integrates TensorFlow neural network models with finite element analysis (FEA) and experimental displacement data obtained from DVC.

Code and data used in:

M. Bourdyot, M. Compans, R. Langlois, B. Smaniotto, E. Baranger, C. Jailin, **Learning a hyperelastic constitutive model from 3D experimental data**, submitted to publication. 


[(NH residuals forces)](https://cjailin.github.io/html_outputs/3D_PANN_learning/NH_residual_forces.html)

[(PANN residuals forces)](https://cjailin.github.io/html_outputs/3D_PANN_learning/PANN_residual_forces.html)


## Repository Structure
This repo is organized with 2 independent tutorials and a common set of basic methods (PANN_lib).
``` bash
├───3D_learning_core
│   └───Main_exp_EUCLID.py
├───DVC_data
│   ├───connectivity_scan1.csv
│   └───coordinates_scan1.csv
│   └───displacements_scan1-1.csv
│   └───displacements_scan2-1.csv
│   └───displacements_scan3-1.csv
│   └───displacements_scan4-1.csv
├───PANN_lib
│   ├───FEA_fun.py
│   ├───FEA_mesh.py
│   └───NN_3D_models.py
│   ├───Pvista_plots.py
│   ├───training_utils.py
│   └───utils.py
```

- [`3D_training_core/`](/3D_training_core/): Main scripts for model training and validation.
- [`DVC_data/`](/DVC_data/): CSV files (`coordinates_scan1.csv`, `connectivity_scan1.csv`, `displacements_scan*.csv`) with the experimental datasets.
- [`PANN_lib/`](/PANN_lib/): Core Python libraries for FEA functions, PANN, and traditional models, training, and visualization.


## Requirements
  - python >= 3.8
  - tensorflow >= 2.0
  - pandas
  - pyvista
  - numpy
  - matplotlib
  - seaborn
  - tensorboard (optional for monitoring)

## Installation
Create a [virtual environment](https://docs.python.org/3/library/venv.html) and install the required libraries:
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

Feel free to contribute, start a Github discussion, or open issues!
