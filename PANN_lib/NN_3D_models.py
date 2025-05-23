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
from tensorflow.keras import layers,regularizers,initializers
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras import layers
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
    
    I  = tf.linalg.trace(C)
    I2 = 0.5*(tf.math.square(tf.linalg.trace(C))-tf.linalg.trace(tf.linalg.matmul(C,C)))
    J  = tf.linalg.det(x)
    
    # Return multiple invariants as a stack
    # return tf.stack([I, J, -J, I2, 
    #                  tf.math.square(J-1), 
    #                  tf.math.log(J), I**2], axis=-1)
    return tf.stack([I, J, -J, I2], axis=-1)
    

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
    
    
    def __init__(self, n=5, layer_num=5, **kwargs):
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
        for i in range(layer_num):
            self.hidden_layers.append(
                layers.Dense(
                    units=n,
                    activation='softplus',
                    kernel_constraint=NonNeg(),
                    name=f"hidden_{i}",
                    kernel_initializer=initializers.GlorotUniform(seed=42),
                )
            )

        self.final_layer = layers.Dense(
            units=1,
            activation='softplus',
            kernel_constraint=NonNeg(),
            name="final_dense",
            kernel_initializer=initializers.GlorotUniform(seed=42),
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
        
        with tf.GradientTape() as tape:
            identity_F = tf.tile(tf.expand_dims(tf.eye(3, dtype=F.dtype), axis=0), [batch_size, 1, 1])
            tape.watch(identity_F)
            # Manually compute energy at identity (same as identity_energy)
            identity_features = self.invariant_layer(identity_F)
            for layer in self.hidden_layers:
                identity_features = layer(identity_features)
            energy_identity = self.final_layer(identity_features) # this is pure W(I) + tf.expand_dims((tf.linalg.det(identity_F) + 1.0 / tf.linalg.det(identity_F) - 2.0) ** 2, axis=-1) 
        
        H = -tape.gradient(energy_identity, identity_F)
        # # Compute FᵀF - I
        C = tf.matmul(tf.transpose(F, perm=[0, 2, 1]), F)
        E = (C - tf.eye(3, batch_shape=[batch_size], dtype=F.dtype))/2  # (batch, 3, 3)
        # Frobenius inner product H:C
        correction = tf.reduce_sum(H * E, axis=[1, 2], keepdims=True)  # (batch, 1)
        correction = tf.squeeze(correction, axis=-1)
        
        return energy - energy_identity + correction
 
      
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
        # Calculate Lamé coefficients
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
        
        potential = self.mu_nh * tf.reduce_sum(E_tensor ** 2, axis=[1, 2]) + \
                    (0.5*self.lambda_nh) * trace_E ** 2
        
        return potential
    
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

    def call(self, x):
        inv = compute_inv_3D_isotropic(x)
        I = inv[:, 0]
        J = inv[:, 1]

        W = self.mu_nh / 2 * (I - 3) - self.mu_nh * tf.math.log(J) + \
            self.lambda_nh / 2 * tf.math.square(tf.math.log(J))

        return W

def NH_model(E, nu):
    xs = tf.keras.Input(shape=(3, 3), dtype=tf.float64)

    potential_layer = Compute_NH_Potentials(E, nu)
    ys = potential_layer(xs)

    derivative_layer = DerivativeLayer(potential_layer)
    PKS = derivative_layer(xs)

    model_gene = tf.keras.Model(inputs=[xs], outputs=[ys, PKS])

    return model_gene

#%%
'''
Additional invariant computation methods
'''

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
    # Ensure proper type casting
    x2 = tf.cast(x,dtype=tf.float32)
    
    # Compute the right Cauchy-Green tensor: C = F^T * F
    C = tf.linalg.matmul(x2,tf.linalg.matrix_transpose(x2))
    
    # Compute H as Hencky strain tensor
    H = 0.5 * matrix_log(C)
    
    trace_H = tf.linalg.trace(H)              
    trace_H = tf.reshape(trace_H, (-1, 1, 1))    
    
    # Compute the deviatoric part of H
    devH =  H - (1/3) * trace_H * tf.eye(3, dtype=H.dtype)
    
    K1 = tf.linalg.trace(H) 
    K2 = tf.sqrt(tf.reduce_sum(tf.square(devH), axis=[-2, -1]))
    K3 = 3 * tf.sqrt(6.0) / (K2**3) * tf.linalg.det(devH)
    
    return tf.stack([K1, K2, K3], axis=-1)



