# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 17:16:35 2025

@author: Clement Jailin

Université Paris-Saclay, CentraleSupélec, ENS Paris-Saclay, CNRS, 
LMPS—Laboratoire de Mécanique Paris-Saclay, 
Gif-sur-Yvette,France
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Pvista_plots import (
    plot_single_mesh_result,
    )
from FEA_fun import (
    compute_global_nodal_forces,
    )


def load_displacement(filename, voxel_size):
    """
    Load and scale displacement data from CSV.

    Parameters
    ----------
    filename : str, voxel_size : float

    Returns
    -------
    np.ndarray
    """
    displacement = pd.read_csv(filename, header=None).values[:, :3]
    return displacement * voxel_size


def matrix_log(C):
    """
    Compute a tensor log
    """
    eigenvalues, eigenvectors = tf.linalg.eigh(C)
    log_eigenvalues = tf.math.log(eigenvalues)
    log_diag = tf.linalg.diag(log_eigenvalues)
    log_C = tf.matmul(eigenvectors, tf.matmul(log_diag, tf.linalg.matrix_transpose(eigenvectors)))
    
    return log_C

def compute_K_inv(x):
    """
    Compute invariants of the right Cauchy-Green deformation tensor from F.
    Returns a tensor of shape (..., 3): [I, J, -J].
    
    see: https://hal.science/hal-04292137v1/file/RCT_Costecalde_Verron_revised.pdf
    """
    x2 = tf.cast(x,dtype=tf.float32)
    
    # Compute the right Cauchy-Green tensor and Hencky strain tensor
    C = tf.linalg.matmul(x2,tf.linalg.matrix_transpose(x2))
    H = 0.5 * matrix_log(C)
    
    trace_H = tf.linalg.trace(H)              
    trace_H = tf.reshape(trace_H, (-1, 1, 1))    
    
    # Compute the deviatoric part of H
    devH =  H - (1/3) * trace_H * tf.eye(3, dtype=H.dtype)
    
    K1 = tf.linalg.trace(H) 
    K2 = tf.sqrt(tf.reduce_sum(tf.square(devH), axis=[-2, -1]))
    K3 = 3 * tf.sqrt(6.0) / (K2**3) * tf.linalg.det(devH)
    
    return tf.stack([K1, K2, K3], axis=-1)



def divergence_loss_PANN_3D(y_pred, mesh, f_app):
    """
    Compute divergence loss for finite element analysis.
    
    Parameters
    ----------
    y_pred : Tensor - Predicted stress tensors (num_elements, stress_components, stress_components).
    mesh   : Mesh class
    f_app  : np - applied force.
    
    Returns
    -------
    tf.Tensor: Scalar divergence loss.
    tuple:     (total_loss, bc_error, mean_error)
    """
    global_nodal_forces = compute_global_nodal_forces(tf.transpose(y_pred,perm=[0,2,1]), mesh)
    global_nodal_forces_int = tf.gather(global_nodal_forces, mesh.free_nodes_indices)

    Top_force    = tf.gather(global_nodal_forces, mesh.top_nodes_indices)
    Bottom_force = tf.gather(global_nodal_forces, mesh.bottom_nodes_indices)
    
    # Compute the loss (external & internal)
    loss1 = tf.reduce_sum( tf.square(global_nodal_forces_int) )
    loss2 = (tf.reduce_sum(Top_force[:,2])+abs(f_app))**2
    loss3 = (tf.reduce_sum(Bottom_force[:,2])-abs(f_app))**2
    ltot  = loss1+loss2+loss3
    
    loss_tot_save = int(ltot.numpy())
    bc_save = np.round(((np.sqrt(loss2)+np.sqrt(loss3))/2),2)
    me_save = np.round(np.sqrt(loss1/len(global_nodal_forces_int)),2)
    
    return ltot, (loss_tot_save,bc_save,me_save)

def plot_loss_surface(loss_grid, E_range, nu_range, title, cmap_limits):
    """
    Plot log10-scaled loss surface over (E, ν) parameter grid.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(np.log10(loss_grid),
               extent=[nu_range[0], nu_range[-1], E_range[-1], E_range[0]],
               aspect='auto', cmap='viridis', vmin=cmap_limits[0], vmax=cmap_limits[1])
    plt.colorbar(label='Log(Loss)')
    plt.xlabel(r'$\nu$')
    plt.ylabel('$E$ (MPa)')
    plt.show()


def grid_search_model(model_fn, mesh, E_range, nu_range, dataset, label, cmap_limits=(3.5, 6)):
    """
    Grid search over (E, ν).

    Parameters
    ----------
    model_fn : callable, mesh : Mesh, E_range : list, nu_range : list,
    dataset : dict, label : str, cmap_limits : tuple

    Returns
    -------
    np.ndarray (loss grid)
    """
    NN = len(E_range)
    Loss_grid = np.zeros((NN, NN))

    for iE, E in enumerate(E_range):
        for inu, nu in enumerate(nu_range):
            model = model_fn(E, nu)
            P = model(dataset['deformation'])[1]
            loss, _ = divergence_loss_PANN_3D(P, mesh, dataset['force'])
            Loss_grid[iE, inu] = loss.numpy()

    plot_loss_surface(Loss_grid, E_range, nu_range, label, cmap_limits)
    return Loss_grid


def find_optimal_parameters(loss_grid, E_range, nu_range):
    """
    Find minimum-loss parameters.

    Parameters
    ----------
    loss_grid : np.ndarray, E_range : array, nu_range : array

    Returns
    -------
    float (loss), float (E), float (ν)
    """
    idx_min = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
    return loss_grid[idx_min], E_range[idx_min[0]], nu_range[idx_min[1]]


def evaluate_model(model_fn, mesh, E, nu, dataset, title):
    """
    Evaluate model and return loss, prediction.

    Parameters
    ----------
    model_fn : callable, mesh : Mesh, E : float, nu : float,
    dataset : dict, title : str

    Returns
    -------
    float (loss), tf.Tensor (stress)
    """
    model = model_fn(E, nu)
    P = model(dataset['deformation'])[1]
    loss, metrics = divergence_loss_PANN_3D(P, mesh, dataset['force'])
    return loss.numpy(), P


def plot_residual_forces(P, mesh, title):
    """
    Plot residual nodal forces.

    Parameters
    ----------
    P : tf.Tensor, mesh : Mesh, title : str
    """
    forces = compute_global_nodal_forces(tf.transpose(P, perm=[0, 2, 1]), mesh)
    field  = -forces # This is simply a convention
    plot_single_mesh_result(mesh, field, title=title, cmap="hot", arrows=field)


def plot_invariant_pairplot(dataset_DVC, title=None):
    """
    Create a lower-triangle pairplot of (K1, K2, K3) invariants from a DVC dataset.

    Parameters
    ----------
    dataset_DVC : dict
        Dictionary where each entry is a loading step containing a 'K' key with
        shape (N, 3) invariant tensors as tf.Tensor.
    title : str, optional
        Title to display above the plot.
    """
    # Build concatenated DataFrame
    dfs = []
    for step, data in dataset_DVC.items():
        df = pd.DataFrame(data['K'].numpy(), columns=['$K_1$', '$K_2$', '$K_3$'])
        df['Group'] = f'Loading {step}'
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    # Create pairplot
    g = sns.pairplot(df_all, hue='Group', palette='Set1', plot_kws={'alpha': 0.3, 's': 10})

    # Remove upper triangle for clarity
    for i in range(3):
        for j in range(i + 1, 3):
            ax = g.axes[i, j]
            if ax is not None:
                ax.remove()

    # Format and add grid
    for row in g.axes:
        for ax in row:
            if ax is not None:
                ax.grid(True, which='both', linestyle=':', linewidth=0.5)

    if title:
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    plt.show()
