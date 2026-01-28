import numpy as np
from inner_reconstruction_method import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap


def compute_reconstruction_test_values_partial(
    mesh, gradients, ntd_AD, ntd_A0, rho, element_list, pos_def=True
):
    """
    Compute inclusion indices by processing mesh partitions sequentially.
    """
    tdim = mesh.topology.dim
    connectivity = mesh.topology.connectivity(tdim, 0)

    # Owned cells only
    num_elements = mesh.topology.index_map(tdim).size_local

    N = np.shape(ntd_A0)[0]
    min_eig_indices = []
    min_eig_values = []
    frechet_quantity_values = []

    if N > 16:
        frechet_mode_sample_idx = 8
    else:
        frechet_mode_sample_idx = 0

    # print(f"Sampling Frechet quantity at mode index {frechet_mode_sample_idx}")

    for elem in element_list:
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

        if pos_def:
            check = ntd_A0 + rho * Dfrechet - ntd_AD
        else:
            check = ntd_AD - (ntd_A0 + rho * Dfrechet)

        check = (check + check.T) / 2  # ensure symmetry
        min_eig = np.min(np.real(np.linalg.eigvals(check)))

        idx = round(N / 2 + frechet_mode_sample_idx)
        frechet_quantity = np.abs(Dfrechet[idx, idx])

        min_eig_values.append(min_eig)
        frechet_quantity_values.append(frechet_quantity)
        min_eig_indices.append(elem)

    min_eig_values = np.asarray(min_eig_values, dtype=float)
    frechet_quantity_values = np.asarray(frechet_quantity_values, dtype=float)

    eigenvalue_dict = dict(zip(min_eig_indices, min_eig_values))
    frechet_quantity_dict = dict(zip(min_eig_indices, frechet_quantity_values))

    return eigenvalue_dict, frechet_quantity_dict


def reconstruction2(
    mesh,
    gradients,
    ntd_AD,
    ntd_A0,
    alpha,
    rho0,
    rho_step,
    max_it=100,
    minimum_inclusions=1,
    print_output=True,
    pos_def=True,
):
    # list of element indices
    tdim = mesh.topology.dim
    num_elements = mesh.topology.index_map(tdim).size_local
    active_elements = list(range(num_elements))
    inclusion_counter = np.zeros(num_elements, dtype=int)

    rho = rho0
    done = False

    for it in range(max_it):
        if print_output:
            if it % 10 == 0:
                print(f"Iteration {it}, active_elements={len(active_elements)}")

        ev_dict, _ = compute_reconstruction_test_values_partial(
            mesh, gradients, ntd_AD, ntd_A0, rho, active_elements, pos_def=pos_def
        )

        inclusion_indices = reconstruct_inclusions(ev_dict, alpha)
        if len(inclusion_indices) < minimum_inclusions:
            print(f"Total iterations: {it + 1}")
            done = True
            break

        # update counter
        inclusion_counter[inclusion_indices] += 1

        # update active elements
        active_elements = inclusion_indices

        # update rho
        rho += rho_step

    if not done:
        print("Warning: Maximum iterations reached without convergence.")

    return inclusion_counter


color_positions = [
    (0.0, "white"),  # 0% - white
    (0.15, "yellow"),  # 15% - yellow
    (1.0, "magenta"),  # 100% - magenta
]
custom_cmap = LinearSegmentedColormap.from_list("positioned_cmap", color_positions)


def plot_method_2(mesh, inclusion_counter, colormap=None):
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

    inclusion_indexes = np.arange(len(inclusion_counter))
    # inclusion_indexes = (np.where(inclusion_counter > 0)[0]).tolist()
    norm_counter = inclusion_counter / np.max(inclusion_counter)

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
        cmap=cmap, norm=plt.Normalize(vmin=0, vmax=np.max(inclusion_counter))
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Count", rotation=270, labelpad=15)

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
