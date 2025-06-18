# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 10:39:37 2025

@author: Clement Jailin

Université Paris-Saclay, CentraleSupélec, ENS Paris-Saclay, CNRS, 
LMPS—Laboratoire de Mécanique Paris-Saclay, 
Gif-sur-Yvette,France
"""

import pyvista as pv
import numpy as np

def plot_mesh_results(coordinates, connect, fields, titles, cmaps=None, arrows=None, ops=None):
    """
    Plots a mesh with 4 different fields using PyVista. Automatically detects if the data
    is cell-based or point-based.

    Parameters
    ----------
    coordinates : ndarray
        Array of node coordinates, shape (num_nodes, 3).
    connect : ndarray
        Connectivity array defining elements (tetrahedra), shape (num_elements, 4).
    fields : list of ndarray
        List of 4 arrays containing the data to plot. Each can be either point data
        (num_nodes,) or cell data (num_elements,).
    titles : list of str
        List of 4 titles for each subplot.
    cmaps : list of str, optional
        List of colormaps for each subplot. Defaults to standard colormaps.
    arrows : list of tuples, optional
        Each tuple should be (positions, vectors), positions and vectors are ndarrays
        used for plotting arrows. If None, no arrows plotted for that subplot.

    Returns
    -------
    None
    
    
    ______________________________________________
    EXAMPLE of USE

    fields = [U_plot, 
              U_plot[:,2], 
              global_nodal_forces, 
              P_pred[:,2,2]]

    titles = ["Displacements", "Displacements2", "Forces", "Stresses"]
    arrows = [U_plot, None, -global_nodal_forces, None]

    plot_mesh_results(coordinates, connect, fields, titles, arrows=arrows)
    ______________________________________________
    
    """
    # Default colormaps if none provided
    if cmaps is None:
        cmaps = ["coolwarm", "viridis", "plasma", "inferno"]
    if ops is None:
        ops = [1,1,1,1]

    # Build cells array: [4, node0, node1, node2, node3] for each tetrahedron
    cells = np.hstack([np.insert(c, 0, 4) for c in connect]).astype(np.int64)
    cell_types = np.full(len(connect), 10, dtype=np.uint8)  # type 10 for tetrahedron

    # Create the base grid (no deformation)
    grid = pv.UnstructuredGrid(cells, cell_types, coordinates)

    # Setup plotter
    plotter = pv.Plotter(shape=(2, 2), window_size=(1200, 800))

    for idx, (field, title, cmap, op) in enumerate(zip(fields, titles, cmaps, ops)):
        plotter.subplot(idx // 2, idx % 2)
        plotter.add_text(title, font_size=10)
        field = np.array(field)
        
        mesh = grid.copy()

        # Determine if field is point data or cell data
        if field.shape[0] == coordinates.shape[0]:
            mesh.point_data[title] = field
            plotter.add_mesh(mesh, scalars=title, cmap=cmap, show_edges=True, opacity=op)
        elif field.shape[0] == connect.shape[0]:
            mesh.cell_data[title] = field
            plotter.add_mesh(mesh, scalars=title, cmap=cmap, show_edges=True, opacity=op)
        else:
            raise ValueError(f"Field '{title}' has incompatible shape.")

        # Plot arrows if provided
        if arrows[idx] is not None:
            plotter.add_arrows(coordinates, np.array(arrows[idx]), mag=1.0, color="red")
            
    plotter.link_views()
    plotter.show(interactive_update=True)



#%%


def plot_single_mesh_result(mesh, field, title='', cmap="plasma", arrows=None, op=1, save=False):
    """
    Plots a mesh with a single field using PyVista. Automatically detects if the data
    is cell-based or point-based. Optionally plots arrows representing vectors.

    Parameters
    ----------
    coordinates : ndarray
        Array of node coordinates, shape (num_nodes, 3).
    connect : ndarray
        Connectivity array defining elements (tetrahedra), shape (num_elements, 4).
    field : ndarray
        Array containing the data to plot, either point data (num_nodes,) or cell data (num_elements,).
    title : str
        Title for the plot.
    cmap : str, optional
        Colormap for the plot. Default is "plasma".
    arrows : ndarray, optional
        Array of vectors for plotting arrows at the node coordinates. If None, no arrows plotted.

    Returns
    -------
    None
    
    ______________________________________________
    EXAMPLE of USE

    field = P_pred[:,2,2]  # Example field data
    arrows = global_nodal_forces  # Example arrows (forces)

    plot_single_mesh_result(mesh, field, title="Stress Prediction", cmap="hot", arrows=arrows)
    ______________________________________________
    
    """
    coordinates,connect = mesh.coordinates, mesh.connectivity
    
    cells = np.hstack([np.insert(c, 0, 4) for c in connect]).astype(np.int64)
    cell_types = np.full(len(connect), 10, dtype=np.uint8)  # tetra

    grid = pv.UnstructuredGrid(cells, cell_types, coordinates)

    # Set scalar field
    if field.shape[0] == coordinates.shape[0]:
        grid.point_data[title] = field
    elif field.shape[0] == connect.shape[0]:
        grid.cell_data[title] = field
    else:
        raise ValueError("Field size does not match node or element count.")

    plotter = pv.Plotter(window_size=(600, 600))
    plotter.add_text(title, font_size=12)
    plotter.add_mesh(grid, scalars=title, cmap=cmap, show_edges=True, opacity=op)

    if arrows is not None:
        plotter.add_arrows(coordinates, np.array(arrows), mag=1.0, color="red")
        plotter.add_arrows(coordinates, np.array(arrows), mag=1.0, color="red")

    if save:
        plotter.show(interactive_update=True)
        plotter.export_html(f"{title}.html")
        print(f"Interactive HTML saved to: {title}.html")
    else:
        plotter.show(interactive_update=True)

#%%

"""
PLOT and UPDATE

______________________________________________
EXAMPLE of USE

# # Initialize plot once
plotter, grid, arrows_actor, mesh_actor = initialize_plot(
    coordinates, connect, 
    initial_field=np.array(P_pred)[:,2,2]),
    title="Stress", 
    cmap="plasma",
    initial_arrows=np.array(global_nodal_forces)
)

# Loop where fields are updated
for iteration in range(10):
    new_field = np.array(P_pred[:,2,2])
    new_arrows = np.array(global_nodal_forces)*iteration

    arrows_actor = update_plot(
        plotter, grid, mesh_actor, new_field, "Stress",
        new_arrows=new_arrows, arrows_actor=arrows_actor
    )
______________________________________________

"""


def initialize_plot(coordinates, connect, initial_field, title, cmap="plasma", initial_arrows=None):
    cells = np.hstack([np.insert(c, 0, 4) for c in connect]).astype(np.int64)
    cell_types = np.full(len(connect), 10, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, coordinates)

    plotter = pv.Plotter(window_size=(600, 600))
    plotter.add_text(title, font_size=12)

    if initial_field.shape[0] == coordinates.shape[0]:
        grid.point_data[title] = initial_field
        mesh_actor = plotter.add_mesh(grid, scalars=title, cmap=cmap, show_edges=True)
    elif initial_field.shape[0] == connect.shape[0]:
        grid.cell_data[title] = initial_field
        mesh_actor = plotter.add_mesh(grid, scalars=title, cmap=cmap, show_edges=True)
    else:
        raise ValueError("Initial field has incompatible shape.")

    arrows_actor = None
    if initial_arrows is not None:
        arrows_actor = plotter.add_arrows(coordinates, initial_arrows, mag=1.0, color="red")

    plotter.show(interactive_update=True)

    return plotter, grid, arrows_actor, mesh_actor


def update_plot(plotter, grid, mesh_actor, new_field, title, new_arrows=None, arrows_actor=None):
    # Update field data in-place
    if new_field.shape[0] == grid.number_of_points:
        grid.point_data[title][:] = new_field
    elif new_field.shape[0] == grid.number_of_cells:
        grid.cell_data[title][:] = new_field
    else:
        raise ValueError("New field has incompatible shape.")

    # Update arrows if needed
    if arrows_actor:
        plotter.remove_actor(arrows_actor)
        arrows_actor = None

    if new_arrows is not None:
        arrows_actor = plotter.add_arrows(grid.points, new_arrows, mag=1.0, color="red")

    # Refresh rendering
    plotter.render()

    return arrows_actor



