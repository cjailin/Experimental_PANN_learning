
# Physics-Augmented Neural Networks library

Core functionalities supporting finite element analysis, mesh manipulation, neural network models, and visualization.

## Contents
- `FEA_fun.py`: FEA methods for gradient computation and deformation analysis.
- `NN_3D_model.py`: TensorFlow implementation of constitutive models
    - PANN model
    - Saint-Venant-Kirchoff
    - Neo Hookean model
    - invariant computation (I,I2,I3,-2J)
- `Pvista_plots.py`: Visualization methods using PyVista.
- `training_utils.py`: main trainng function
- `utils.py`: general utility methods
 
- `FEA_mesh.py`: (optional) Mesh generation methods using gmsh.

## Features
- Precomputation of shape function gradients and deformation tensors
- TensorFlow-based neural network architectures (ICNN, LE and NH models)
- Visualization of 3D meshes and simulation results

## Usage
```python
from PANN_lib.FEA_mesh import generate_mesh
from PANN_lib.NN_3D_model import main_PANN_3D
```

