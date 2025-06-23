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
import matplotlib.pyplot as plt
import matplotlib as mpl

# Default font sizes for Matplotlib plots.
mpl.rcParams['axes.labelsize'] = 14   
mpl.rcParams['axes.titlesize'] = 16   
mpl.rcParams['xtick.labelsize'] = 14  
mpl.rcParams['ytick.labelsize'] = 14  
mpl.rcParams['legend.fontsize'] = 14  
mpl.rcParams['font.size'] = 14  

from training_utils import train_model

from FEA_fun import (
    compute_deformation_tensor_from_U,
    compute_global_nodal_forces,
    Mesh,
    )

from NN_3D_models import (
    main_PANN_3D,
    linear_model,
    NH_model,
    )

from Pvista_plots import (
    plot_mesh_results,
    plot_single_mesh_result,
    )

from utils import (
    compute_K_inv,
    load_displacement,
    divergence_loss_PANN_3D,
    grid_search_model,
    find_optimal_parameters,
    evaluate_model,
    plot_residual_forces,
    plot_invariant_pairplot,
    )

#%%



VOXEL_SIZE    = 30e-3 # micrometer scale
forces        = [280,590,863,1110] # Loading forces
GRID_SAMPLING = 15

TRAIN_NUM = '4'
VALID_NUM = '3'


#%%
'''
Load and pre-process data
'''

# Load connectivity coordinates and apply scaling
connectivity = pd.read_csv('../DVC_data/connectivity_scan1.csv', header=None).values
coordinates  = pd.read_csv('../DVC_data/coordinates_scan1.csv' , header=None).values * VOXEL_SIZE
mesh         = Mesh(coordinates, connectivity)

dataset_DVC = {}
for loading_step in ['1','2','3','4']:
    dataset_DVC[loading_step]={}
    dataset_DVC[loading_step]['force'] = forces[int(loading_step)-1]
    dataset_DVC[loading_step]['displacement'] = load_displacement('../DVC_data/displacements_scan'+loading_step+'-1.csv',VOXEL_SIZE)
    dataset_DVC[loading_step]['deformation']  = compute_deformation_tensor_from_U(dataset_DVC[loading_step]['displacement'].T, mesh)
    dataset_DVC[loading_step]['K']  = compute_K_inv(dataset_DVC[loading_step]['deformation'])

plot_invariant_pairplot(dataset_DVC, title="Invariant Space per Loading Step")

#%%
'''
Linear elastic model
'''
print('Evaluate Saint-Venant-Kirchoff model...')
E_range_LE  = np.linspace(4, 6, GRID_SAMPLING)
nu_range_LE = np.linspace(0.0, 0.2, GRID_SAMPLING)
loss_LE_grid = grid_search_model(linear_model, mesh, E_range_LE, nu_range_LE, dataset_DVC[TRAIN_NUM], label="Saint-Venant-Kirchoff", cmap_limits=(3.5, 6))
min_loss_LE, E_opt_LE, nu_opt_LE = find_optimal_parameters(loss_LE_grid, E_range_LE, nu_range_LE)
LE_model_ = NH_model(E_opt_LE, nu_opt_LE)
print(f"LE model -- Train Loss: {min_loss_LE:.2f} N², E: {E_opt_LE:.2f} MPa, nu: {nu_opt_LE:.3f}")

for valid_num in ['3','2','1']:
    loss_valid_LE, P_valid_LE = evaluate_model(linear_model,mesh, E_opt_LE, nu_opt_LE, dataset_DVC[valid_num], "LE valid "+valid_num)
    print(f"LE model -- Valid loss {valid_num}: {loss_valid_LE:.2f} N²")

P_train_LE = LE_model_(dataset_DVC[TRAIN_NUM]['deformation'])[1]
plot_residual_forces(P_train_LE, mesh, "Saint-Venant-Kirchoff residual forces")


#%% 
'''
NH model
'''
print('Evaluate Neo-Hookean model...')
E_range_NH = np.linspace(9, 11, GRID_SAMPLING)
nu_range_NH = np.linspace(0.25, 0.45, GRID_SAMPLING)

loss_NH_grid = grid_search_model(NH_model, mesh, E_range_NH, nu_range_NH, dataset_DVC[TRAIN_NUM], "Neo-Hookean", cmap_limits=(3.5, 5.5))
min_loss_NH, E_opt_NH, nu_opt_NH = find_optimal_parameters(loss_NH_grid, E_range_NH, nu_range_NH)
NH_model_ = NH_model(E_opt_NH, nu_opt_NH)
print(f"NH model -- Train Loss: {min_loss_NH:.2f} N², E: {E_opt_NH:.2f} MPa, nu: {nu_opt_NH:.3f}")

for valid_num in ['3','2','1']:
    loss_valid_NH, P_valid_NH = evaluate_model(NH_model,mesh, E_opt_NH, nu_opt_NH, dataset_DVC[valid_num], "NH valid" + valid_num)
    print(f"NH model -- Valid loss {valid_num}: {loss_valid_NH:.2f} N²")

P_train_NH = NH_model_(dataset_DVC[TRAIN_NUM]['deformation'])[1]
plot_residual_forces(P_train_NH, mesh, "NH residual forces")

#%%
'''
PANN model
'''
# Initialize the PANN model
model = main_PANN_3D(n=8,layer_num=2)

# Define optimizer
boundaries = [1000,   2500,  3000,       ]
values     = [0.02,   0.01,  0.005,  0.005]
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Train the model
max_epoch = 10000
losses = train_model(model, 
                     dataset_DVC, 
                     train_names=[TRAIN_NUM],
                     valid_name=VALID_NUM, 
                     epochs=max_epoch, 
                     optimizer=optimizer, 
                     mesh=mesh)

# # Save model
import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y%m%d_%H%M")
model_save_path = 'model_saved_'+formatted_time
model.save('saved/'+str(model_save_path))
print(f'Model saved to: saved_models/{model_save_path}')


plt.figure()
plt.semilogy(np.array(range(0,max_epoch,1)),losses[0],label='training - step '+str(TRAIN_NUM) )
plt.semilogy(np.linspace(0,max_epoch,len(losses[1])),losses[1],label='Evaluation - step '+str(VALID_NUM))
plt.xlabel('Epochs')
plt.ylabel('EUCLID loss')
plt.legend()
plt.grid()
plt.show()


#%%
print("Name         || Loss error |  BC error [N] | ME_int error [N] || Num neg. W ||")
print('----------------')
for VALID_NUM in ['4','3','2','1']:

    W_LE,P_LE = LE_model_(dataset_DVC[VALID_NUM]['deformation'])
    loss_val,metrics_LE   = divergence_loss_PANN_3D(P_LE, mesh, dataset_DVC[VALID_NUM]['force'])
    lt, bc_t, me_t = metrics_LE   
    print(f"{str('LE   - step'+VALID_NUM)} || "
          f"{lt:10.1f} | {bc_t:13.2f} | {me_t:16.3f} || {int(sum(W_LE.numpy()<0)):10.0f} ||")
    
    W_NH,P_NH = NH_model_(dataset_DVC[VALID_NUM]['deformation'])
    loss_val,metrics_NH   = divergence_loss_PANN_3D(P_NH, mesh, dataset_DVC[VALID_NUM]['force'])
    lt, bc_t, me_t = metrics_NH   
    print(f"{str('NH   - step'+VALID_NUM)} || "
          f"{lt:10.1f} | {bc_t:13.2f} | {me_t:16.3f} || {int(sum(W_NH.numpy()<0)):10.0f} ||")
    
    W_NN,P_NN = model(dataset_DVC[VALID_NUM]['deformation'], training=True)
    loss_val,metrics_NN   = divergence_loss_PANN_3D(P_NN, mesh, dataset_DVC[VALID_NUM]['force'])
    lt, bc_t, me_t = metrics_NN   
    print(f"{str('PANN - step'+VALID_NUM)} || "
          f"{lt:10.1f} | {bc_t:13.2f} | {me_t:16.3f} || {int(sum(W_NN.numpy()<0)):10.0f} ||")
    print('----------------')


#%%
# Plot results PANN
plot_choice = '4'

W_pred,P_pred  = model.predict(dataset_DVC[plot_choice]['deformation'])
global_nodal_forces = compute_global_nodal_forces(tf.transpose(P_pred,perm=[0,2,1]), mesh)

field  = -global_nodal_forces  
arrows = -global_nodal_forces  
plot_single_mesh_result(mesh, field, title="PANN_residual_forces", cmap="hot", arrows=arrows, op=0.3, save=True)








