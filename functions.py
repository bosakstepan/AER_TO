import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch

def plot_topology(Mesh, x, BF, port=None):
    """
    Plots the optimized mesh topology.

    Parameters:
    - Mesh: dict with keys 'nodes' (Nx3) and 'connectivityList' (Tx3)
    - x: (T,) or (T,1) array of design values in [0,1]
    - BF: dict with key 'data', each row has edge index in column 2 (MATLAB's 3rd col)
    - port: index of the basis function to highlight

    Returns:
    - handle: Poly3DCollection object
    """

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    nodes = Mesh['nodes']
    faces = Mesh['connectivityList'].astype(int) - 1  # Convert to zero-based index

    # --- Prepare triangle colors (using grayscale mapping)
    N = len(x)
    x = x.detach().cpu().numpy()
    grayscale = (255 - 0.08 * (255 - 0)) / 255  # fixed grayscale as in MATLAB code
    face_colors = np.tile(grayscale, (len(faces), 3))

    # --- Create triangles
    triangles = [nodes[face] for face in faces]
    handle = Poly3DCollection(triangles, facecolors=face_colors, edgecolor='k', linewidth=0.3, alpha=0)
    ax.add_collection3d(handle)

    # --- Set view and limits
    ax.view_init(elev=90, azim=-90)
    ax.set_box_aspect([2,1,0.2])
    ax.set_xlim([np.min(nodes[:,0]), np.max(nodes[:,0])])
    ax.set_ylim([np.min(nodes[:,1]), np.max(nodes[:,1])])
    ax.axis('off')
    # --- Plot edges with color x[i]
    for i in range(N):
        edge_index = BF['data'][i, 2].astype(int) - 1 # Convert to zero-based index
        edge_nodes = Mesh['edges'][edge_index].astype(int) - 1
        edge_coords = nodes[edge_nodes]
        rbg_color = x[i]
        ax.plot(edge_coords[:,0], edge_coords[:,1], edge_coords[:,2], color=rbg_color, linewidth=2)

    # --- Highlight port
    if port is not None:
        port_edge = BF['data'][port, 2].astype(int) - 1
        port_nodes = Mesh['edges'][port_edge].astype(int) - 1
        port_coords = nodes[port_nodes]
        ax.plot(port_coords[:,0], port_coords[:,1], port_coords[:,2], color='r', linewidth=3)

    plt.tight_layout()
    plt.show()

    return handle

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