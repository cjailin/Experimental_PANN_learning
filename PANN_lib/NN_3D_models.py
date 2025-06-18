# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:00:33 2024

@author: Clement Jailin

Université Paris-Saclay, CentraleSupélec, ENS Paris-Saclay, CNRS, 
LMPS—Laboratoire de Mécanique Paris-Saclay, 
Gif-sur-Yvette,France


Three main models are implemented, P=M(F):
    - PANN with standard architecture:  main_PANN_3D()
    - Traditional elastic linear model: P_elastic(E,nu) 
    - Traditional NH model:             NH_model(E,nu)

"""
import tensorflow as tf
from tensorflow.keras import initializers, layers
from tensorflow.keras.constraints import NonNeg
import numpy as np


#%%

def compute_inv_3D_isotropic(x):
    """
    Compute the 3D isotropic invariants of the right Cauchy-Green deformation tensor.
    
    Args:
        x (tf.Tensor): Input deformation gradient tensor of shape (3, 3).
    Returns:
        tf.Tensor: A tensor containing a set of invariants based on the input deformation gradient.
    """
    
    x = tf.cast(x, tf.float64)
    
    # Compute the right Cauchy-Green tensor C = F^T * F
    C = tf.linalg.matmul(tf.linalg.matrix_transpose(x), x)
    I1 = tf.linalg.trace(C)
    I2 = 0.5*(tf.math.square(tf.linalg.trace(C))-tf.linalg.trace(tf.linalg.matmul(C,C)))
    I3 = tf.linalg.det(C)
    J  = tf.linalg.det(x)
    
    return tf.stack([I1, I2, I3, -2*J], axis=-1)
    

# Function to compute growth energy (optional)
def compute_growth(x):
    """
    Compute growth energy term based on the determinant of the deformation gradient.
    
    Args:
        x (tf.Tensor): Input deformation gradient tensor of shape (3, 3).
    Returns:
        tf.Tensor: A scalar tensor representing the growth energy.
    """
    J  = tf.linalg.det(x)
    return tf.math.square(J+1/J-2)


class inv_comp(layers.Layer):
    
    
    def __init__(self, n=8, layer_num=2, **kwargs):
        """
        Args:
            num_experiments (int): Total number of experiments.
            embedding_dim (int): Dimension of the experiment embedding vector.
            n (int): Number of neurons per hidden layer.
            layer_num (int): Number of hidden layers.
            lambda_disent (float): Weight for the disentanglement loss.
        """
   
        super(inv_comp, self).__init__(**kwargs)

        self.invariant_layer = layers.Lambda(compute_inv_3D_isotropic, name="invariants")

        self.hidden_layers = []
        positive_init = tf.keras.initializers.RandomUniform(minval=1e-3, maxval=0.1, seed=42)
        
        for i in range(layer_num):
            self.hidden_layers.append(
                layers.Dense(
                    units=n,
                    activation='softplus',
                    kernel_initializer=positive_init,
                    kernel_constraint=NonNeg(),
                    name=f"hidden_{i}",
                )
            )
            

        self.final_layer = layers.Dense(
            units=1,
            activation='softplus',
            kernel_initializer=positive_init,
            # kernel_constraint=NonNeg(),
            name="final_dense",
        )
        

    def call(self, inputs):
        """
        Forward pass.
        Args:
            inputs: A tuple (F, exp_id) where:
                - F is a tensor of shape (batch, 3, 3) representing the deformation gradient.
                - exp_id is a tensor of shape (batch,) containing the experiment index.
        Returns:
            Energy difference (scalar per sample) relative to the identity deformation.
        """
        F = inputs

        # Compute invariant features from the deformation gradient.
        inv_features = self.invariant_layer(F)
        for layer in self.hidden_layers:
            inv_features = layer(inv_features)

        energy = self.final_layer(inv_features)

        # ---- Subtract identity energy ----
        batch_size = tf.shape(F)[0]
        
        identity_F = tf.eye(3, dtype=F.dtype)[tf.newaxis, ...]
        identity_inv = self.invariant_layer(identity_F)
        with tf.GradientTape() as tape:
            tape.watch(identity_inv)
            identity_features = tf.identity(identity_inv)
            for layer in self.hidden_layers:
                identity_features = layer(identity_features)
            energy_identity = self.final_layer(identity_features) 
        
        H = tape.gradient(energy_identity, identity_inv)

        nu = 2. * H * tf.stack([1.*tf.ones_like(H[:,0]), 
                                2.*tf.ones_like(H[:,1]), 
                                1.*tf.ones_like(H[:,2]), 
                               -1.*tf.ones_like(H[:,3])
                                ], axis=-1)
        nu_red = tf.reduce_sum(nu , axis=[1], keepdims=True)
        nu_red_tiled = tf.tile(nu_red, [batch_size, 1])
        
        J = tf.linalg.det(F)[:, tf.newaxis]
        correction = nu_red_tiled * (J - 1.0)
        
        return energy - energy_identity - correction + (J+1/J-2)**2
 
      
class DerivativeLayer(layers.Layer):
    def __init__(self, model):
        super(DerivativeLayer, self).__init__()
        self.model = model

    def call(self, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            predictions = self.model(inputs)
        return tape.gradient(predictions, inputs)
     
        
def main_PANN_3D(**kwargs):
    """
    Create the 3D PANN model for computing energy potentials and stress tensors.
    
    Returns:
        tf.keras.Model: The PANN model with energy and derivative (stress) outputs.
    """
    # Define input shape for the deformation gradient (3x3 matrix)
    xs = tf.keras.Input(shape=(3,3), dtype=tf.float64)
    
    # Create the invariant computation and energy potential layer (ICNN)
    icnn_layer = inv_comp(**kwargs)
    ys         = icnn_layer(xs)
    
    # Add the derivative layer (computes stress tensor)
    derivative_layer = DerivativeLayer(icnn_layer) 
    derivatives      = derivative_layer(xs) 
    
    # Connect inputs and outputs to create the model
    model = tf.keras.Model(inputs = [xs], outputs = [ys,derivatives])
    return model




#%%

def Stress_elastic(F_, E_, nu_):
    """
    Compute the stress: P or sigma for linear elasticity.
    Args:
        F_ (np.ndarray): Deformation gradient tensor, shape (n, 3, 3).
    Returns:
        np.ndarray: Stress tensor.
    """
    
    mu_nh     = E_ / (2 * (1 + nu_))
    lambda_nh = E_ * nu_ / ((1 + nu_) * (1 - 2* nu_))
    
    F_array = np.array(F_)  
    sym_grad = 0.5 * (np.matmul(np.transpose(F_array, axes=(0, 2, 1)),F_array)  - np.eye(3))

    sigma_elas = 2 * mu_nh * sym_grad + lambda_nh * np.trace(sym_grad, axis1=1, axis2=2)[:, np.newaxis, np.newaxis] * np.eye(3)
    
    # If it is the PK tensor that is needed:
    # P_elas = F_@P_elas
    
    return np.stack(sigma_elas,0)


#%%

class Compute_Linear_Potentials(layers.Layer):

    def __init__(self, E, nu):
        super(Compute_Linear_Potentials, self).__init__()
        self.mu_nh = E / (2 * (1 + nu))  # Shear modulus (mu)
        self.lambda_nh = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lamé's first parameter (lambda)

    def call(self, F):
        """
        Compute the elastic potential for linear elasticity.
        Args:
            x (tf.Tensor): Input tensor of deformation gradients, shape (N, 3, 3).
        Returns:
            tf.Tensor: Elastic potential, shape (N,).
        """
        # Calculate the strain tensor E = 1/2 (F^T * F - I)
        E_tensor = 0.5 * (tf.matmul(F, F, transpose_a=True) - tf.eye(3, dtype=tf.float64))
        
        # Compute the energy potential:
        trace_E = tf.linalg.trace(E_tensor)  
        
        W = self.mu_nh *tf.linalg.trace(E_tensor ** 2) + \
            (0.5*self.lambda_nh) * trace_E ** 2
        
        return W
    
def linear_model(E, nu):
    
    xs = tf.keras.Input(shape=(3, 3), dtype=tf.float64)

    potential_layer = Compute_Linear_Potentials(E, nu)
    ys = potential_layer(xs)

    derivative_layer = DerivativeLayer(potential_layer)
    PKS = derivative_layer(xs)

    model_gene = tf.keras.Model(inputs=[xs], outputs=[ys, PKS])

    return model_gene

#%%

class Compute_NH_Potentials(layers.Layer):

    def __init__(self, E, nu):
        super(Compute_NH_Potentials, self).__init__()
        self.mu_nh = E / (2 * (1 + nu))
        self.lambda_nh = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.kappa = E / (3 * (1 - 2 * nu))
        
    def call(self, x):
        inv = compute_inv_3D_isotropic(x)
        I   = inv[:, 0]
        J   = tf.sqrt(inv[:, 2])

        W = self.mu_nh / 2 * (I - 3) - self.mu_nh * tf.math.log(J) + \
            self.lambda_nh / 2 * tf.math.square(tf.math.log(J))
            
        # W = self.mu_nh / 2 * (I - 3) - self.mu_nh * tf.math.log(J) + \
        #     self.lambda_nh / 2 * tf.math.square(J-1)
            
        # W = self.mu_nh / 2 * (I - 3) - self.mu_nh * tf.math.log(J) + \
        #     self.kappa / 2 * tf.math.square(J-1)

        return W

def NH_model(E, nu):
    xs = tf.keras.Input(shape=(3, 3), dtype=tf.float64)

    potential_layer = Compute_NH_Potentials(E, nu)
    ys = potential_layer(xs)

    derivative_layer = DerivativeLayer(potential_layer)
    PKS = derivative_layer(xs)

    model_gene = tf.keras.Model(inputs=[xs], outputs=[ys, PKS])

    return model_gene






