# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 17:05:54 2025

@author: Clement Jailin

Université Paris-Saclay, CentraleSupélec, ENS Paris-Saclay, CNRS, 
LMPS—Laboratoire de Mécanique Paris-Saclay, 
Gif-sur-Yvette,France
"""
import numpy as np
import tensorflow as tf
import datetime
from tensorflow.keras.callbacks import TensorBoard
import os

from FEA_fun import (
    compute_global_nodal_forces,
    )
from utils import (
    divergence_loss_PANN_3D,
    )
from Pvista_plots import (
    initialize_plot,
    update_plot
    )


def train_model(model, dataset_DVC, train_names, valid_name, epochs, optimizer, mesh):
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
    
    print("-------------------------------------------------------------------------------------------------------")
    print('start training')
    print(f"To view TensorBoard, run:\ntensorboard --logdir={os.path.abspath('logs/fit')}")
    print("-------------------------------------------------------------------------------------------------------")

    plotter, grid, arrows_actor, mesh_actor = initialize_plot(
        mesh.coordinates, mesh.connectivity, 
        initial_field=np.zeros((mesh.num_nodes,3)) ,
        title="Forces - PANN model", 
        cmap="hot",
        initial_arrows=np.zeros((mesh.num_nodes,3 )))

    # Create a logs directory with timestamp
    log_dir = f"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)

    losses=[]
    val_losses=[]
    for epoch in range(epochs):
        loss_step = []
        for dataset_name in train_names:
            with tf.GradientTape() as tape:
                y_pred = model(dataset_DVC[dataset_name]['deformation'], training=True)[1]
                loss_value, metrics_train = divergence_loss_PANN_3D(y_pred,mesh,dataset_DVC[dataset_name]['force'])
                loss_value = tf.cast(loss_value,dtype=tf.float64)
            # Gradient 
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
            loss_step.append(loss_value)

        loss_epoch = np.mean(loss_step)
        losses.append(loss_epoch) 

        # tensorboard output
        with summary_writer.as_default():
            tf.summary.scalar('Loss/train', loss_epoch, step=epoch)
    
        # Print every xx epochs
        if epoch % 100==0:
            
            new_field = -np.array(compute_global_nodal_forces(tf.transpose(y_pred,perm=[0,2,1]),mesh))
            arrows_actor = update_plot(
                plotter, grid, mesh_actor, new_field, "Forces - PANN model",
                new_arrows=new_field, arrows_actor=arrows_actor)
            
            y_pred_val = model(dataset_DVC[valid_name]['deformation'], training=True)[1]
            loss_val, metrics_val   = divergence_loss_PANN_3D(y_pred_val,mesh,dataset_DVC[valid_name]['force'])
            val_losses.append(loss_val.numpy())
            
            lt, bc_t, me_t = metrics_train   
            lv, bc_v, me_v = metrics_val     
            
            if epoch % 1000==0:
                print("Epoch || Loss train | BC train [N] | ME_int train [N] || Loss valid | BC valid [N] | ME_int valid [N]")
                print("-------------------------------------------------------------------------------------------------------")
            print(f"{epoch:5d} || "
                  f"{lt:10.1f} | {bc_t:12.2f} | {me_t:16.3f} || "
                  f"{lv:10.1f} | {bc_v:12.2f} | {me_v:15.3f}")
            
            with summary_writer.as_default():
                tf.summary.scalar('Loss/valid', loss_val, step=epoch)
                tf.summary.scalar('Metrics/BC_valid', bc_v, step=epoch)
                tf.summary.scalar('Metrics/ME_valid', me_v, step=epoch)

    print("-------------------------------------------------------------------------------------------------------")
    print('end training')
    print(f"To view TensorBoard, run:\ntensorboard --logdir={os.path.abspath('logs/fit')}")
    print("-------------------------------------------------------------------------------------------------------")
    
    return losses, val_losses
