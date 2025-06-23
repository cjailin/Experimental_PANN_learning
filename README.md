# Physics-Augmented Neural Networks for 3D unsupervised learning

This repository contains code and data for developing and using Physics-Augmented Neural Networks (PANN) aimed at modeling isotropic material behavior under deformation. It integrates TensorFlow neural network models with finite element analysis (FEA) and experimental deformation data.

Code and data used in:

M. Bourdyot, M. Compans, R. Langlois, B. Smaniotto, E. Baranger, C. Jailin, **3D learning of a hyperelastic behavior with experimental data**, submitted to publication. 

## Repository Structure
This repo is organized with 2 independent tutorials and a common set of basic methods (PANN_lib).
- `PANN_lib/`: Core Python libraries for FEA functions, PANN, and traditional models, training, and visualization.
- `DVC_data/*`: CSV files (`coordinates_scan1.csv`, `connectivity_scan1.csv`, `displacements_scan*.csv`) with the experimental datasets.
- `3D_learning_core/`: Main scripts for model training and validation.

## Requirements
Requirements may vary on the tutorial. General requirements are:
  - python >= 3.8
  - tensorflow >= 2.0
  - pandas
  - pyVista
  - numPy
  - matplotlib
  - tensorboard (optional for vizu)
  - gmsh (optional)

## Installation
Create a virtual environement and install the required libraries:
```bash
pip install tensorflow matplotlib pandas pyvista tensorboard
```
or
```bash
conda install tensorflow matplotlib pandas pyvista tensorboard
```

## Getting Started

```bash
cd tuto_exp_EUCLID_3D
python Main_exp_EUCLID.py
```

---

Feel free to contribute or open issues!
