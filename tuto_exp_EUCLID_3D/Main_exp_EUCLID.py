# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 15:10:41 2025

@author: Clement Jailin

Université Paris-Saclay, CentraleSupélec, ENS Paris-Saclay, CNRS, 
LMPS—Laboratoire de Mécanique Paris-Saclay, 
Gif-sur-Yvette,France
"""

import sys
sys.path.append(r'..\\PANN_lib')

import numpy as np
import tensorflow as tf
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib as mpl
      
# Default font sizes for Matplotlib plots.
mpl.rcParams['axes.labelsize'] = 14   
mpl.rcParams['axes.titlesize'] = 16   
mpl.rcParams['xtick.labelsize'] = 14  
mpl.rcParams['ytick.labelsize'] = 14  
mpl.rcParams['legend.fontsize'] = 14  
mpl.rcParams['font.size'] = 14  


from FEA_fun import (
    precompute_gradients,
    compute_deformation_tensor_from_U,
    compute_global_nodal_forces
    )

from NN_3D_models import (
    main_PANN_3D,
    linear_model,
    NH_model,
    compute_K_inv
    )

from Pvista_plots import (
    plot_mesh_results,
    plot_single_mesh_result,
    initialize_plot,
    update_plot
    )
    

#%%
'''
Load and pre-process data
'''

voxel_size = 30e-3 # micrometer scale

# Load connectivity coordinates and apply scaling
connectivity = pd.read_csv('DVC_data/connectivity_scan1.csv', header=None).values
coordinates  = pd.read_csv('DVC_data/coordinates_scan1.csv', header=None).values * voxel_size
coordx, coordy, coordz = coordinates.T
Nnodes = coordinates.shape[0]

# Function to load displacement data and apply scaling
def load_displacement(filename):
    displacement = (pd.read_csv(filename, header=None).values)[:,:3] * voxel_size
    return displacement

# Load displacement datasets
depl_1 = load_displacement('DVC_data/displacements_scan1-1.csv')
depl_2 = load_displacement('DVC_data/displacements_scan2-1.csv')
depl_3 = load_displacement('DVC_data/displacements_scan3-1.csv')
depl_4 = load_displacement('DVC_data/displacements_scan4-1.csv')

f_app_1 = 280  
f_app_2 = 590   
f_app_3 = 863   
f_app_4 = 1110  

# BC identification
z_min, z_max = np.min(coordz), np.max(coordz)
bottom_nodes = abs(coordz- z_min)<0.15
top_nodes    = abs(coordz - z_max)<0.15

int_nodes = list(np.where(bottom_nodes+top_nodes==0)[0])
                 
# Load mesh
coords, connect = coordinates,connectivity
coords_tf  = tf.constant(coords, dtype=tf.float64)
connect_tf = tf.constant(connect, dtype=tf.int32)
num_nodes = tf.shape(coords_tf)[0]

# Compute gradient and volumes
grad_N_all, volumes = precompute_gradients(coords_tf, connect_tf)

free_nodes_indices = int_nodes
top_nodes_indices = np.where(top_nodes==1)[0]
bottom_nodes_indices = np.where(bottom_nodes==1)[0]

F_TPU_1 = compute_deformation_tensor_from_U(depl_1.T, connect_tf, grad_N_all)
F_TPU_2 = compute_deformation_tensor_from_U(depl_2.T, connect_tf, grad_N_all)
F_TPU_3 = compute_deformation_tensor_from_U(depl_3.T, connect_tf, grad_N_all)
F_TPU_4 = compute_deformation_tensor_from_U(depl_4.T, connect_tf, grad_N_all)


# Initialize the PANN model
model = main_PANN_3D()

#%%
dataset_DVC = {}
dataset_DVC['1']={}
dataset_DVC['2']={}
dataset_DVC['3']={}
dataset_DVC['4']={}

dataset_DVC['1']['force']=f_app_1
dataset_DVC['2']['force']=f_app_2
dataset_DVC['3']['force']=f_app_3
dataset_DVC['4']['force']=f_app_4

dataset_DVC['1']['deformation']=F_TPU_1
dataset_DVC['2']['deformation']=F_TPU_2
dataset_DVC['3']['deformation']=F_TPU_3
dataset_DVC['4']['deformation']=F_TPU_4

dataset_DVC['1']['displacement']=depl_1
dataset_DVC['2']['displacement']=depl_2
dataset_DVC['3']['displacement']=depl_3
dataset_DVC['4']['displacement']=depl_4

train_num = '4'
valid_num = '3'

K_1=compute_K_inv(F_TPU_1)
K_2=compute_K_inv(F_TPU_2)
K_3=compute_K_inv(F_TPU_3)
K_4=compute_K_inv(F_TPU_4)

plt.figure()
plt.scatter(K_1[:,1],K_1[:,2],s=0.1)
plt.scatter(K_2[:,1],K_2[:,2],s=0.1)
plt.scatter(K_3[:,1],K_3[:,2],s=0.1)
plt.scatter(K_4[:,1],K_4[:,2],s=0.1)
plt.xlabel('invariant K2')
plt.ylabel('invariant K3')
plt.show()
#%%


def divergence_loss_PANN_3D(y_pred, T4_tf, grad_N_all, volumes, num_nodes,free_nodes_indices,f_app):
    """
    Compute divergence loss for finite element analysis.
    
    Parameters
    ----------
    y_pred : Tensor - Predicted stress tensors (num_elements, stress_components, stress_components).
    T4_tf : Tensor - Element connectivity.
    grad_N_all : Tensor - Gradients of shape functions for each element.
    volumes : Tensor - Element volumes.
    num_nodes : int - Total number of nodes in the mesh.
    free_nodes_indices : ndarray - Indices of nodes with free degrees of freedom.
    
    Returns
    -------
    loss : Tensor - Scalar loss representing divergence.
    """
    global_nodal_forces = compute_global_nodal_forces(tf.transpose(y_pred,perm=[0,2,1]), T4_tf, grad_N_all, volumes, num_nodes)
    global_nodal_forces_int = tf.gather(global_nodal_forces, free_nodes_indices)

    Top_force = tf.gather(global_nodal_forces, top_nodes_indices)
    Bottom_force = tf.gather(global_nodal_forces, bottom_nodes_indices)
       
    
    # Compute the loss (external & internal)
    loss1 = tf.reduce_sum( tf.square(global_nodal_forces_int) )
    loss2 = (tf.reduce_sum(Top_force[:,2])+abs(f_app))**2
    loss3 = (tf.reduce_sum(Bottom_force[:,2])-abs(f_app))**2
    ltot  = loss1+loss2+loss3
    
    loss_tot_save = int(ltot.numpy())
    bc_save = np.round(((np.sqrt(loss2)+np.sqrt(loss3))/2),2)
    me_save = np.round(np.sqrt(loss3/len(global_nodal_forces_int)),2)
    
    return ltot, (loss_tot_save,bc_save,me_save)



#%% 
# '''
# Linear elastic material
# '''
# # Size of the grid
# NN=20
# E_values   = np.linspace(4, 6, NN)
# nu_values  = np.linspace(0., 0.3, NN)

# Loss_saved = np.zeros((NN,NN))
# for itE,E in enumerate(E_values):
#     for itnu,nu  in enumerate(nu_values):
#         LE_model_ = linear_model(E,nu)
#         P_LE = LE_model_(dataset_DVC[train_num]['deformation'])[1]  
#         loss_donnees_elastiques_h = divergence_loss_PANN_3D(P_LE, connect_tf, grad_N_all, volumes, num_nodes,free_nodes_indices,dataset_DVC[train_num]['force'])
#         loss_donnees_elastiques_h = loss_donnees_elastiques_h.numpy()
#         Loss_saved[itE,itnu] = loss_donnees_elastiques_h
  

# plt.figure(figsize=(8, 6))
# plt.imshow(np.log(Loss_saved),
#            extent=[nu_values[0], nu_values[-1], E_values[-1], E_values[0]],
#            aspect='auto', cmap='viridis')
# plt.colorbar(label='Log(Loss)')
# plt.xlabel('nu (Poisson ratio)')
# plt.ylabel('E (Young modulus)')
# plt.show()


# def find_min_loss(Loss_saved, E_values, nu_values):
#     """
#     Finds the minimum loss value from a 2D array and retrieves the corresponding
#     parameter values from provided arrays of parameters (E_values and nu_values).
#     """
#     idx_min = np.unravel_index(np.argmin(Loss_saved), Loss_saved.shape)
#     min_loss = Loss_saved[idx_min]
#     E_min = E_values[idx_min[0]]
#     nu_min = nu_values[idx_min[1]]
#     return min_loss, E_min, nu_min

# min_loss_LE, E_min_LE, nu_min_LE = find_min_loss(Loss_saved, E_values, nu_values)

# LE_model_ = linear_model(E_min_LE, nu_min_LE)
# P_LE_valid = LE_model_(dataset_DVC[valid_num]['deformation'])[1] 
# min_loss_valid_LE = divergence_loss_PANN_3D(P_LE_valid, connect_tf, grad_N_all, volumes, num_nodes,free_nodes_indices,dataset_DVC[valid_num]['force'])

# print(f"EL model -- Minimum Loss train: {min_loss_LE:.2f} N², E: {E_min_LE:.2f} MPa, nu: {nu_min_LE:.3f}")
# print(f"EL model -- Minimum Loss valid: {min_loss_valid_LE:.2f} N²")

# P_LE = LE_model_(dataset_DVC[train_num]['deformation'])[1] 
# global_nodal_forces_LE = compute_global_nodal_forces(tf.transpose(P_LE,perm=[0,2,1]), connect_tf, grad_N_all, volumes, num_nodes)

# # plot results
# field  = -global_nodal_forces_LE  
# arrows = -global_nodal_forces_LE  
# plot_single_mesh_result(coordinates, connect, field, title="Linear Elastic residual forces", cmap="hot", arrows=arrows,op=0.1)

# #%% 
# '''
# NH material
# '''
# # Size of the grid
# NN=20
# E_values   = np.linspace(5, 15, NN)
# nu_values  = np.linspace(0.15, 0.45, NN)

# Loss_saved = np.zeros((NN,NN))
# for itE,E in enumerate(E_values):
#     for itnu,nu  in enumerate(nu_values):
#         NH_model_ = NH_model(E,nu)
#         P_NH = NH_model_(dataset_DVC[train_num]['deformation'])[1] 
#         loss_NH = divergence_loss_PANN_3D(P_NH, connect_tf, grad_N_all, volumes, num_nodes,free_nodes_indices,dataset_DVC[train_num]['force'])
#         loss_NH = loss_NH.numpy()
#         Loss_saved[itE,itnu] = loss_NH
  

# plt.figure(figsize=(8, 6))
# plt.imshow(np.log(Loss_saved),
#            extent=[nu_values[0], nu_values[-1], E_values[-1], E_values[0]],
#            aspect='auto', cmap='viridis')
# plt.colorbar(label='Log(Loss)')
# plt.xlabel('nu (Poisson ratio)')
# plt.ylabel('E (Young modulus)')
# plt.show()


# def find_min_loss(Loss_saved, E_values, nu_values):
#     """
#     Finds the minimum loss value from a 2D array and retrieves the corresponding
#     parameter values from provided arrays of parameters (E_values and nu_values).
#     """
#     idx_min = np.unravel_index(np.argmin(Loss_saved), Loss_saved.shape)
#     min_loss = Loss_saved[idx_min]
#     E_min = E_values[idx_min[0]]
#     nu_min = nu_values[idx_min[1]]
#     return min_loss, E_min, nu_min

# min_loss_train_NH, E_min_NH, nu_min_NH = find_min_loss(Loss_saved, E_values, nu_values)


# NH_model_ = NH_model(E_min_NH, nu_min_NH)
# P_NH_valid = NH_model_(dataset_DVC[valid_num]['deformation'])[1]
# min_loss_valid_NH = divergence_loss_PANN_3D(P_NH_valid, connect_tf, grad_N_all, volumes, num_nodes,free_nodes_indices,dataset_DVC[valid_num]['force'])

# print(f"NH model -- Minimum Loss train: {min_loss_train_NH:.2f} N², E: {E_min_NH:.2f} MPa, nu: {nu_min_NH:.3f}")
# print(f"NH model -- Minimum Loss valid: {min_loss_valid_NH:.2f} N²")


# P_NH = NH_model_(dataset_DVC[train_num]['deformation'])[1]
# global_nodal_forces_NH = compute_global_nodal_forces(tf.transpose(P_NH,perm=[0,2,1]), connect_tf, grad_N_all, volumes, num_nodes)

# # plot results
# field  = -global_nodal_forces_NH  
# arrows = -global_nodal_forces_NH  
# plot_single_mesh_result(coordinates, connect, field, title="NH residual forces", cmap="hot", arrows=arrows)


#%%
# dataset definition:

name = 'DVC4'
dataset_EUCLID       = {}
dataset_EUCLID[name] = {}
dataset_EUCLID[name]['deformation'] = dataset_DVC[train_num]['deformation']
dataset_EUCLID[name]['force']       = dataset_DVC[train_num]['force']

L_loss = []
L_loss_val = []




def train_model(model, train_dataset, epochs, optimizer):
    """
    Train the PINN model over multiple epochs using custom divergence loss.

    Args:
        model (tf.keras.Model): The PANN model to be trained.
        train_dataset (dict): Training data.
        epochs (int): Number of epochs to train.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer used for training.
    Returns:
        list: Loss values over epochs.
    """
    
    plotter, grid, arrows_actor, mesh_actor = initialize_plot(
        coordinates, connect, 
        initial_field=np.zeros((num_nodes,3)) ,
        title="Forces - PANN model", 
        cmap="hot",
        initial_arrows=np.zeros((num_nodes,3 )))
    
    losses=[]
    val_losses=[]
    for epoch in range(epochs):
        
        # print('Epoch - '+str(epoch))
        loss_step = []
        for dataset_name in dataset_EUCLID:
            train_dataset      = dataset_EUCLID[dataset_name]['deformation']
            Force              = dataset_EUCLID[name]['force']
            
            with tf.GradientTape() as tape:
                y_pred = model(train_dataset, training=True)[1]
                loss_value, metrics_train = divergence_loss_PANN_3D(y_pred, connect_tf, grad_N_all, volumes, num_nodes,free_nodes_indices,Force)
                loss_value = tf.cast(loss_value,dtype=tf.float64)
                
            # Gradient computation
            grads = tape.gradient(loss_value, model.trainable_weights)
            
            # Apply gradients (update model weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
            loss_step.append(loss_value)

        loss_epoch = np.mean(loss_step)
        losses.append(loss_epoch) 
        
        # Print every xx epochs
        if epoch % 10==0:
            
            
            new_field = -np.array(compute_global_nodal_forces(tf.transpose(y_pred,perm=[0,2,1]), connect_tf, grad_N_all, volumes, num_nodes))
            arrows_actor = update_plot(
                plotter, grid, mesh_actor, new_field, "Forces - PANN model",
                new_arrows=new_field, arrows_actor=arrows_actor)
            
            
            y_pred_val = model(dataset_DVC[valid_num]['deformation'], training=True)[1]
            loss_val, metrics_val   = divergence_loss_PANN_3D(y_pred_val, connect_tf, grad_N_all, volumes, num_nodes,free_nodes_indices,dataset_DVC[valid_num]['force'])
            val_losses.append(loss_val.numpy())
            
                        
            lt, bc_t, me_t = metrics_train   # unpack your train metrics
            lv, bc_v, me_v = metrics_val     # unpack your valid metrics
            
            if epoch % 200==0:
                print("Epoch || Loss train |  BC train [N] | ME_int train [N] || "
                      " Loss valid | BC valid [N] | ME_int valid [N]")
            print(f"{epoch:5d} || "
                  f"{lt:10.1f} | {bc_t:13.2f} | {me_t:16.3f} || "
                  f"{lv:11.1f} | {bc_v:12.2f} | {me_v:15.3f}")

             
    return losses, val_losses


# Define optimizer
boundaries = [1000,   2500,  3000, ]
values     = [0.05,   0.02,  0.01,  0.005]
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


# Train the model
max_epoch = 4000
losses = train_model(model, dataset_EUCLID, epochs=max_epoch, optimizer=optimizer)

# # Save model
import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y%m%d_%H%M")
model_save_path = 'model_saved_'+formatted_time
model.save('saved/'+str(model_save_path))
print(f'Model saved to: saved_models/{model_save_path}')


plt.figure()
plt.semilogy(np.array(range(0,max_epoch,1)),losses[0],label='training - step '+str(train_num) )
plt.semilogy(np.array(range(0,max_epoch,10)),losses[1],label='training - step '+str(valid_num))
plt.xlabel('Epochs')
plt.ylabel('EUCLID loss')
plt.legend()
plt.show()



#%%

print(f"EL model -- Minimum Loss train: {min_loss_LE:.2f} N², E: {E_min_LE:.2f} MPa, nu: {nu_min_LE:.3f}")
print(f"EL model -- Minimum Loss valid: {min_loss_valid_LE:.2f} N²")
print('----')
print(f"NH model -- Minimum Loss train: {min_loss_train_NH:.2f} N², E: {E_min_NH:.2f} MPa, nu: {nu_min_NH:.3f}")
print(f"NH model -- Minimum Loss valid: {min_loss_valid_NH:.2f} N²")
print('----')
print(f"PANN     -- Minimum Loss train: {losses[0][-1]:.2f} N²")
print(f"PANN     -- Minimum Loss valid: {losses[1][-1]:.2f} N²")


#%%
# PLot results PANN

F_plot = dataset_DVC[train_num]['deformation']

W_pred,P_pred  = model.predict(F_plot)
global_nodal_forces = compute_global_nodal_forces(tf.transpose(P_pred,perm=[0,2,1]), connect_tf, grad_N_all, volumes, num_nodes)

# plot results
field  = -global_nodal_forces  
arrows = -global_nodal_forces  
plot_single_mesh_result(coordinates, connect, field, title="PANN residual forces", cmap="hot", arrows=arrows)


fields = [dataset_DVC[train_num]['displacement'], 
          F_plot[:,2,2], 
          global_nodal_forces, 
          P_pred[:,2,2]]

titles = ["Displacements", "Deformation", "Forces", "Stresses"]
arrows = [depl_4, None, -global_nodal_forces, None]
cmaps = ["coolwarm", "viridis", "plasma", "inferno"]

plot_mesh_results(coordinates, connect, fields, titles, arrows=arrows,cmaps=cmaps)


#%%
# Plot comparison PANN vs LE

fields = [global_nodal_forces_LE, 
          global_nodal_forces_LE-global_nodal_forces, 
          global_nodal_forces, 
          abs(global_nodal_forces_LE-global_nodal_forces)]

titles = ["Forces LE", "Force difference", "Forces PANN", "Force difference"]
arrows = [-global_nodal_forces_LE, 
          None, 
          -global_nodal_forces, 
          -global_nodal_forces_LE+global_nodal_forces]
cmaps = ["hot", "seismic", "hot", "seismic"]
ops   = [0.5,1,0.5,0.1]

plot_mesh_results(coordinates, connect, fields, titles, arrows=arrows,cmaps=cmaps, ops=ops)


