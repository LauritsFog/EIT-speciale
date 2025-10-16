import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def plot_conductivity_components(
    conductivity_func,
    title="Conductivity Tensor Components",
    n_points=500,
):
    """
    Plot the conductivity tensor components (a11, a12, a22)
    for a given conductivity function such as my_conductivity_AD(x, y, shape=...).

    Parameters
    ----------
    conductivity_func : callable
        Function taking (x, y, shape=...) and returning (a11, a12, a22)
    shape_choice : str
        Which shape to visualize ("ellipse", "triangle", "U", ...)
    title : str
        Plot title
    domain_radius : float
        Radius of the circular domain (default = 1.0)
    n_points : int
        Grid resolution
    """

    domain_radius = 1.0

    # --- Grid setup
    xi = np.linspace(-domain_radius, domain_radius, n_points)
    yi = np.linspace(-domain_radius, domain_radius, n_points)
    X, Y = np.meshgrid(xi, yi)

    a11_grid = np.full_like(X, np.nan)
    a12_grid = np.full_like(X, np.nan)
    a22_grid = np.full_like(X, np.nan)

    # --- Compute conductivity tensors
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_val, y_val = X[i, j], Y[i, j]
            if np.sqrt(x_val**2 + y_val**2) <= domain_radius:
                a11, a12, a22 = conductivity_func(x_val, y_val)
                a11_grid[i, j] = a11
                a12_grid[i, j] = a12
                a22_grid[i, j] = a22

    # --- Plot setup
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    components = [
        (a11_grid, r"$k_{11}$", "viridis"),
        (a12_grid, r"$k_{12}$", "viridis"),
        (a22_grid, r"$k_{22}$", "viridis"),
    ]

    for ax, (grid, label, cmap) in zip(axes, components):
        im = ax.imshow(
            grid,
            extent=[-domain_radius, domain_radius, -domain_radius, domain_radius],
            origin="lower",
            cmap=cmap,
            aspect="equal",
        )
        ax.contour(X, Y, grid, levels=10, colors="white", linewidths=0.5, alpha=0.7)
        ax.set_title(f"{label} Component")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.add_patch(
            plt.Circle(
                (0, 0),
                domain_radius,
                fill=False,
                color="black",
                linestyle="-",
                linewidth=1,
            )
        )
        plt.colorbar(im, ax=ax, label=label)

    fig.suptitle(f"{title}", fontsize=16)
    plt.tight_layout()
    return fig, axes


def plot_highlighted_elements(mesh, element_indices):
    """
    Plot 2D mesh with highlighted elements using matplotlib

    Parameters:
    -----------
    mesh : dolfinx.mesh.Mesh
        The mesh
    element_indices : list or int
        List of element indices to highlight, or single index
    """
    # Convert single index to list for uniform handling
    if isinstance(element_indices, int):
        element_indices = [element_indices]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the entire mesh
    topology = mesh.topology
    tdim = topology.dim
    cells = topology.connectivity(tdim, 0).array.reshape(-1, tdim + 1)
    vertices = mesh.geometry.x

    # Plot all cells (background mesh)
    for cell in cells:
        poly = plt.Polygon(
            vertices[cell][:, :2], fill=None, edgecolor="black", alpha=0.3
        )
        ax.add_patch(poly)

    # Highlight the specified elements in red
    for element_index in element_indices:
        if element_index < len(cells):
            highlight_cell = cells[element_index]
            poly_highlight = plt.Polygon(
                vertices[highlight_cell][:, :2], fill=True, color="red", alpha=0.7
            )
            ax.add_patch(poly_highlight)
        else:
            print(
                f"Warning: Element index {element_index} out of range (0-{len(cells) - 1})"
            )

    # Set plot properties
    ax.set_aspect("equal")
    ax.autoscale_view()

    title = f"Mesh with {len(element_indices)} highlighted elements"
    if len(element_indices) == 1:
        title = f"Mesh with highlighted element {element_indices[0]}"
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    return fig, ax


def plot_test_matrix_test_matrix_eigenvalues(
    mesh, eigenvalue_dict, title="Minimum eigenvalue of test matrix for each element"
):
    """
    Plot ALL mesh elements colored by eigenvalue number

    Parameters:
    -----------
    mesh : dolfinx.mesh.Mesh
        The mesh
    eigenvalue_dict : dict
        Dictionary mapping element_index -> eigenvalue
    title : str, optional
        Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    topology = mesh.topology
    tdim = topology.dim
    cells = topology.connectivity(tdim, 0).array.reshape(-1, tdim + 1)
    vertices = mesh.geometry.x

    # Get all eigenvalue numbers for normalization
    all_eigenvalues = list(eigenvalue_dict.values())
    if all_eigenvalues:
        vmin = min(all_eigenvalues)
        vmax = max(all_eigenvalues)
        norm = plt.Normalize(vmin, vmax)
        cmap = plt.cm.viridis

        # Plot all cells with color based on eigenvalue number
        for element_index in range(len(cells)):
            if element_index in eigenvalue_dict:
                cond_num = eigenvalue_dict[element_index]
                color = cmap(norm(cond_num))
            else:
                # Elements not in eigenvalue_dict get a default color
                color = "lightgray"

            poly = plt.Polygon(
                vertices[cells[element_index]][:, :2],
                fill=True,
                color=color,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.add_patch(poly)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label("Eigenvalue", rotation=270, labelpad=15)
    else:
        print("No eigenvalues to plot")

    ax.set_aspect("equal")
    ax.autoscale_view()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    return fig, ax
