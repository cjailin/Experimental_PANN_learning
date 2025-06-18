# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:10:29 2025

@author: Clement Jailin

Université Paris-Saclay, CentraleSupélec, ENS Paris-Saclay, CNRS, 
LMPS—Laboratoire de Mécanique Paris-Saclay, 
Gif-sur-Yvette,France
"""
import tensorflow as tf
import numpy as np
import matplotlib as mpl

# Default font sizes for Matplotlib plots.
mpl.rcParams['axes.labelsize'] = 14   
mpl.rcParams['axes.titlesize'] = 16   
mpl.rcParams['xtick.labelsize'] = 14  
mpl.rcParams['ytick.labelsize'] = 14  
mpl.rcParams['legend.fontsize'] = 14  
mpl.rcParams['font.size'] = 14  

tf.keras.backend.set_floatx('float64')


class Mesh:
    """
    Finite Element Mesh object with geometry, connectivity, and boundary detection.
    """
    def __init__(self, coordinates, connectivity):
        self.coordinates = coordinates
        self.connectivity = connectivity
        self.coordx, self.coordy, self.coordz = self.coordinates.T

        self.coords_tf = tf.constant(self.coordinates, dtype=tf.float64)
        self.connect_tf = tf.constant(self.connectivity, dtype=tf.int32)

        self.grad_N_all, self.volumes = precompute_gradients(self.coords_tf, self.connect_tf)

        # Identify boundary and internal nodes
        self.z_min = np.min(self.coordz)
        self.z_max = np.max(self.coordz)
        self.bottom_nodes = np.abs(self.coordz - self.z_min) < 0.15
        self.top_nodes = np.abs(self.coordz - self.z_max) < 0.15
        self.free_nodes_indices = np.where(~(self.bottom_nodes | self.top_nodes))[0]
        self.top_nodes_indices = np.where(self.top_nodes)[0]
        self.bottom_nodes_indices = np.where(self.bottom_nodes)[0]
        self.num_nodes = coordinates.shape[0]
    def __repr__(self):
        return (
            f"Mesh(num_nodes={self.num_nodes}, \n"
            f"     num_elements={self.connectivity.shape[0]}, \n"
            f"     top_nodes={len(self.top_nodes_indices)}, \n"
            f"     bottom_nodes={len(self.bottom_nodes_indices)}, \n"
            f"     free_nodes={len(self.free_nodes_indices)})"
        )

def compute_grad_N_and_volume(elem_coords):
    """
    Compute gradient and volume of tetrahedral elements.

    Parameters
    ----------
    elem_coords : Tensor
        Coordinates of the tetrahedron nodes.

    Returns
    -------
    grad_N : Tensor
        Gradient of shape functions.
    volume : Tensor
        Volume of the tetrahedron.
    """
    ones = tf.ones((4, 1), dtype=tf.float64)
    D = tf.concat([ones, elem_coords], axis=1)  # (4,4)
    D_inv = tf.linalg.inv(D)
    grad_N = D_inv[1:, :]  # (3,4)

    v1 = elem_coords[1] - elem_coords[0]
    v2 = elem_coords[2] - elem_coords[0]
    v3 = elem_coords[3] - elem_coords[0]
    volume = tf.abs(tf.linalg.det(tf.stack([v1, v2, v3], axis=1))) / 6.0

    return grad_N, volume


def precompute_gradients(coords, T4):
    """
    Precompute gradients and volumes for all tetrahedral elements.

    Parameters
    ----------
    coords : Tensor
        Node coordinates array with shape (num_nodes, 3).
    T4 : Tensor
        Element connectivity array with shape (num_elements, 4).

    Returns
    -------
    grad_N_all : Tensor
        Gradient of shape functions for all elements.
    volumes : Tensor
        Volumes for all elements.
    """
    elem_coords = tf.gather(coords, T4)  # (num_elements, 4, 3)
    grad_N_all, volumes = tf.vectorized_map(compute_grad_N_and_volume, elem_coords)
    
    return grad_N_all, volumes


def compute_elem_gradient(inputs):
    """
    Compute the deformation gradient tensor for a single element.

    Parameters
    ----------
    inputs : tuple
        Tuple containing element displacements (elem_U) and shape function gradients (grad_N).

    Returns
    -------
    deformation_tensor : Tensor
        Deformation gradient tensor with shape (3, 3).
    """
    elem_U, grad_N = inputs
    grad_U = tf.matmul(elem_U, grad_N, transpose_a=True, transpose_b=True)  # (3,3)
    deformation_tensor = grad_U + tf.eye(3, dtype=tf.float64)
    return deformation_tensor


def compute_deformation_tensor_from_U(U, mesh):
    """
    Compute deformation tensors from displacement fields.

    Parameters
    ----------
    U : Tensor
        Displacement field tensor with shape (num_nodes, 3).
    mesh : Mesh class

    Returns
    -------
    deformation_tensors : Tensor
        Deformation gradient tensors for all elements with shape (num_elements, 3, 3).
    """
    elem_U = tf.gather(tf.transpose(U), mesh.connect_tf)  # (num_elements, 4, 3)
    deformation_tensors = tf.vectorized_map(compute_elem_gradient, (elem_U, mesh.grad_N_all))  # (num_elements, 3, 3)

    return deformation_tensors


def compute_element_nodal_forces(inputs):
    """
    Compute nodal forces for a single element based on stress.

    Parameters
    ----------
    inputs : tuple
        Tuple containing gradients of shape functions (grad_N), element stress tensor (stress_e), and volume (vol).

    Returns
    -------
    nodal_forces : Tensor
        Element nodal forces tensor with shape (4, 3).
    """
    grad_N, stress_e, vol = inputs
    nodal_forces = -vol * tf.matmul(grad_N, stress_e, transpose_a=True)  # (4,3)
    return nodal_forces  # (4,3)


def compute_global_nodal_forces(y_pred, mesh):
    """
    Compute global nodal forces from predicted stress tensors.

    Parameters
    ----------
    y_pred : Tensor
        Predicted stress tensors for elements.
    mesh: Mesh class

    Returns
    -------
    global_nodal_forces : Tensor
        Global nodal forces tensor with shape (num_nodes, 3).
    """
    stress = tf.cast(y_pred, dtype=tf.float64)
    
    elem_nodal_forces = tf.vectorized_map(
        compute_element_nodal_forces,
        (mesh.grad_N_all, stress, mesh.volumes)
    )  # (num_elements, 4, 3)

    global_nodal_forces = tf.zeros((mesh.num_nodes, 3), dtype=tf.float64)

    indices = tf.reshape(mesh.connect_tf, [-1])
    forces = tf.reshape(elem_nodal_forces, [-1, 3])

    global_nodal_forces = tf.tensor_scatter_nd_add(
        global_nodal_forces,
        indices=tf.expand_dims(indices, axis=1),
        updates=forces
    )

    return global_nodal_forces



