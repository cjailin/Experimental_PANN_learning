# Physics-Augmented Neural Networks for 3D unsupervised learning

This repository contains code and data for developing and using Physics-Augmented Neural Networks (PANN) aimed at modeling isotropic material behavior under deformation. It integrates TensorFlow neural network models with finite element analysis (FEA) and experimental deformation data.

## Repository Structure
This repo is organized with 2 independent tutorials and a common set of basic methods (PANN_lib).
- `PANN_lib/`: Core Python libraries for FEA functions, PANN, and traditional models, training, and visualization.
- `tuto_exp/`: Experimental deformation data and scripts for model training and validation.

## Requirements
Requirements may vary on the tutorial. General requirements are:
  - python >= 3.8
  - tensorFlow >= 2.0
  - pandas
  - pyVista
  - numPy
  - matplotlib
  - (optional) gmsh

## Installation
```bash
pip install tensorflow numpy pandas pyvista matplotlib
```
or
```bash
conda install tensorflow numpy pandas pyvista matplotlib
```

## Getting Started

```bash
cd tuto_exp_EUCLID_3D
python Main_exp_EUCLID.py
```

---

Feel free to contribute or open issues!
