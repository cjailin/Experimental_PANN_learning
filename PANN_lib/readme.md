
# Physics-Augmented Neural Networks Library

Core functionalities supporting finite element analysis, mesh manipulation, neural network models, and visualization.

## Contents
- `FEA_fun.py`: FEA methods for gradient computation and deformation analysis.
- `FEA_mesh.py`: Mesh generation methods using gmsh.
- `NN_3D_model.py`: TensorFlow implementation of constitutive models
    - PANN model
    - Linear Elastic model
    - Neo Hookean model
    - invariant computation (I,J,I2 and also K1, K2, K3)
- `Pvista_plots.py`: Visualization methods using PyVista.

## Features
- Precomputation of shape function gradients and deformation tensors
- TensorFlow-based neural network architectures (ICNN, LE and NH models)
- Visualization of 3D meshes and simulation results

## Usage
```python
from PANN_lib.FEA_mesh import generate_mesh
from PANN_lib.NN_3D_model import main_PANN_3D
```

