import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def plot_topology(Mesh, ax, x, BF, port=None, poly_handle=None, line_handles=None):
    """
    Plots or updates the optimized mesh topology.

    Parameters:
    - Mesh: dict with 'nodes', 'connectivityList', 'edges'
    - ax: matplotlib 3D axis
    - x: array of design values
    - BF: dict with 'data'
    - port: optional index of edge to highlight
    - poly_handle: optional Poly3DCollection for updating
    - line_handles: optional list of line handles to update

    Returns:
    - poly_handle: for faces
    - line_handles: list of line plot objects
    """
    nodes = Mesh['nodes']
    faces = Mesh['connectivityList'].astype(int) - 1
    edges = Mesh['edges']

    x = x.detach().cpu().numpy()
    N = len(x)

    grayscale = (255 - 0.08 * (255 - 0)) / 255
    face_colors = np.tile(grayscale, (len(faces), 3))
    triangles = [nodes[face] for face in faces]
    poly_handle = Poly3DCollection(triangles, facecolors=face_colors, edgecolor='k', linewidth=0.3, alpha=0)
    ax.add_collection3d(poly_handle)

    # --- Plot or replot edges
    for i in range(N):
        edge_index = BF['data'][i, 2].astype(int) - 1
        edge_nodes = edges[edge_index].astype(int) - 1
        edge_coords = nodes[edge_nodes]
        rbg_color = x[i]
        line, = ax.plot(edge_coords[:,0], edge_coords[:,1], edge_coords[:,2],
                        color=rbg_color, linewidth=2)

    # --- Highlight port
    if port is not None:
        port_edge = BF['data'][port, 2].astype(int) - 1
        port_nodes = edges[port_edge].astype(int) - 1
        port_coords = nodes[port_nodes]
        port_line, = ax.plot(port_coords[:,0], port_coords[:,1], port_coords[:,2],
                             color='r', linewidth=3)

    # --- Configure axes once (not inside loop)
    ax.set_box_aspect([2, 1, 0.2])
    ax.set_xlim([np.min(nodes[:,0]), np.max(nodes[:,0])])
    ax.set_ylim([np.min(nodes[:,1]), np.max(nodes[:,1])])
    ax.axis('off')
    ax.view_init(elev=90, azim=-90)
    return


def get_rgb_values(g, port):
    gray_values = 1 - g  # range 0–1
    rgb_values = gray_values.repeat(1, 3)  # shape: (len(g), 3)

    if 0 <= port < len(g):
        rgb_values[port] = torch.tensor([1.0, 0.0, 0.0])  # red override

    return rgb_values



def get_pp_from_port(data, port: int):
    """
    Get the center point of the edge corresponding to the given port index.
    GAS: dict containing mesh and BF data
    port: index of the port (zero-based)
    Returns: center point of the edge as a numpy array
    """
    TEC = data.Mesh.triangleEdgeCenters
    BF = data.BF.data[:, 2].astype(int) - 1
    BFCenters = TEC[BF, :]
    # Same mask as before
    mask = (((BFCenters[:, 0] == 0) & (BFCenters[:, 1] >= 0)) |
            ((BFCenters[:, 1] == 0) & (BFCenters[:, 0] >= 0)))
    BFCentersf = BFCenters[mask, :]
    filtered_indices = np.where(mask)[0]

    # Find index in filtered array that corresponds to the given port
    idx_in_filtered = np.where(filtered_indices == port)[0]
    if idx_in_filtered.size == 0:
        raise ValueError(f"Port index {port} is not in the filtered list of edge centers.")
    
    # Return the center point corresponding to that filtered index
    pp = BFCentersf[idx_in_filtered[0]]
    return pp

def get_p_from_pxpy(maxy, pxpy: np.ndarray) -> float:
    """
    Get the p value from pxpy coordinates.
    maxy: maximum y-coordinate of the mesh
    pxpy: numpy array with px and py coordinates
    Returns: p value
    """
    px, py = pxpy[0], pxpy[1]
    if py > 0:
        p = maxy - py
    else:
        p = px + maxy
    return p