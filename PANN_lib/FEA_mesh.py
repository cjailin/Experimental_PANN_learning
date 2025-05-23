# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:12:24 2025

@author: Clement Jailin

Université Paris-Saclay, CentraleSupélec, ENS Paris-Saclay, CNRS, 
LMPS—Laboratoire de Mécanique Paris-Saclay, 
Gif-sur-Yvette,France
"""

import gmsh
import numpy as np

# Mesh Generation
def generate_mesh(L=1.0, W=1.0, H=2.0, R=0.3, mesh_size=0.1):
    """
    Generate a mesh using gmsh for a block with a cylindrical hole.

    Parameters
    ----------
    L : float
        Length of the block (X-direction).
    W : float
        Width of the block (Y-direction).
    H : float
        Height of the block (Z-direction).
    R : float
        Radius of the cylindrical hole.
    mesh_size : float
        Mesh element size.

    Returns
    -------
    coords : ndarray
        Node coordinates (num_nodes, 3).
    connectivity : ndarray
        Element connectivity (num_elements, 4).
    """
    gmsh.initialize()
    gmsh.model.add("block_with_hole")

    box = gmsh.model.occ.addBox(-L/2, -W/2, -H/2, L, W, H)
    cyl = gmsh.model.occ.addCylinder(-L, 0, 0, 2*L, 0, 0, R)
    gmsh.model.occ.cut([(3, box)], [(3, cyl)], removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.mesh.generate(3)

    _, nodes_coords, _ = gmsh.model.mesh.getNodes()
    coords = nodes_coords.reshape(-1, 3)

    _, elements = gmsh.model.mesh.getElementsByType(4)
    connectivity = np.array(elements).reshape(-1, 4) - 1

    num_nodes = coords.shape[0]
    num_elements = connectivity.shape[0]
    print('-----------------')
    print(f"Mesh generated with {num_nodes} nodes and {num_elements} elements.")

    gmsh.finalize()

    return coords, connectivity