import numpy as np
from inner_reconstruction_method import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import eigvals

def compute_rhos(
    mesh, gradients, ntd_AD, ntd_A0, alpha
):
    """
    Compute inclusion indices by processing mesh partitions sequentially.
    """
    tdim = mesh.topology.dim
    connectivity = mesh.topology.connectivity(tdim, 0)

    # Owned cells only
    num_elements = mesh.topology.index_map(tdim).size_local

    N = np.shape(ntd_A0)[0]
    rhos = []

    if N > 16:
        frechet_mode_sample_idx = 8
    else:
        frechet_mode_sample_idx = 0

    # print(f"Sampling Frechet quantity at mode index {frechet_mode_sample_idx}")

    for elem in range(num_elements):
        verts = mesh.geometry.x[connectivity.links(elem)][:, :2]
        v0, v1, v2 = verts
        area = 0.5 * abs(
            (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
        )

        Dfrechet = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                grad_dot = np.dot(gradients[i][elem], gradients[j][elem])
                Dfrechet[i, j] -= grad_dot * area

                if i != j:
                    Dfrechet[j, i] = Dfrechet[i, j]

        evs = eigvals(- ntd_AD + ntd_A0 + alpha*np.eye(N),-Dfrechet)
        revs = np.real(evs)
        rhos.append(np.min(revs[revs>0]))

    return rhos


def reconstruction3(
    mesh,
    gradients,
    ntd_AD,
    ntd_A0,
    alpha
):
    # list of element indices
    tdim = mesh.topology.dim
    num_elements = mesh.topology.index_map(tdim).size_local
    inclusion_counter = np.zeros(num_elements, dtype=int)


    rhos = compute_rhos(
        mesh, gradients, ntd_AD, ntd_A0, alpha
    )

    return rhos


color_positions = [
    (0.0, "white"),  # 0% - white
    (0.15, "yellow"),  # 15% - yellow
    (1.0, "magenta"),  # 100% - magenta
]
custom_cmap = LinearSegmentedColormap.from_list("positioned_cmap", color_positions)


def plot_method_3(mesh, rhos, colormap=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the entire mesh
    topology = mesh.topology
    tdim = topology.dim
    cells = topology.connectivity(tdim, 0).array.reshape(-1, tdim + 1)
    vertices = mesh.geometry.x

    # Create a colormap
    if colormap is None:
        cmap = custom_cmap
    elif isinstance(colormap, str):
        cmap = cm.get_cmap(colormap)
    else:
        cmap = colormap

    inclusion_indexes = np.arange(len(rhos))
    # inclusion_indexes = (np.where(inclusion_counter > 0)[0]).tolist()
    norm_counter = rhos / np.max(rhos)

    for elem in inclusion_indexes:
        highlight_cell = cells[elem]
        norm_value = norm_counter[elem]
        color = cmap(norm_value)
        poly_highlight = plt.Polygon(
            vertices[highlight_cell][:, :2], fill=True, color=color, alpha=1.0
        )
        ax.add_patch(poly_highlight)
        poly = plt.Polygon(
            vertices[highlight_cell][:, :2],
            fill=None,
            edgecolor="gray",
            alpha=0.3,
            linewidth=0.2,
        )
        ax.add_patch(poly)

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=0, vmax=np.max(rhos))
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Probing constant", rotation=270, labelpad=15)

    # Set plot properties
    ax.set_aspect("equal")
    ax.autoscale_view()

    # set axis limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    ax.set_aspect("equal")

    return fig, ax
