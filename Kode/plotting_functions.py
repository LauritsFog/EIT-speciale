import numpy as np
import matplotlib.pyplot as plt


def plot_conductivity_components(
    Omega_0,
    Omega_1,
    conductivity_Omega_0,
    conductivity_Omega_1,
    title="Conductivity Tensor Components",
):
    """
    Plot the three components of the anisotropic conductivity tensor.

    Parameters:
    -----------
    Omega_0 : function
        Function that takes coordinates (x, y) and returns boolean for subdomain 0
    Omega_1 : function
        Function that takes coordinates (x, y) and returns boolean for subdomain 1
    conductivity_Omega_0 : function
        Function that takes (x, y) and returns (k11, k12, k22) for subdomain 0
    conductivity_Omega_1 : function
        Function that takes (x, y) and returns (k11, k12, k22) for subdomain 1
    title : str
        Title for the plot
    """
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Create fine grid for plotting
    n_points = 500
    xi = np.linspace(-1, 1, n_points)
    yi = np.linspace(-1, 1, n_points)
    X, Y = np.meshgrid(xi, yi)

    # Initialize conductivity arrays
    k11_grid = np.full_like(X, np.nan)
    k12_grid = np.full_like(X, np.nan)
    k22_grid = np.full_like(X, np.nan)

    # Fill conductivity values based on the input functions
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_val = X[i, j]
            y_val = Y[i, j]
            r = np.sqrt(x_val**2 + y_val**2)

            if r <= 1.0:  # Only inside unit disk
                # Determine which subdomain the point belongs to
                if Omega_0(np.array([x_val, y_val])):
                    k11, k12, k22 = conductivity_Omega_0(x_val, y_val)
                elif Omega_1(np.array([x_val, y_val])):
                    k11, k12, k22 = conductivity_Omega_1(x_val, y_val)
                else:
                    k11, k12, k22 = 0, 0, 0

                k11_grid[i, j] = k11
                k12_grid[i, j] = k12
                k22_grid[i, j] = k22
            # Outside unit circle remains as NaN

    # Plot k11 component
    im1 = axes[0].imshow(
        k11_grid, extent=[-1, 1, -1, 1], origin="lower", cmap="viridis", aspect="equal"
    )
    axes[0].contour(
        X, Y, k11_grid, levels=10, colors="white", linewidths=0.5, alpha=0.7
    )
    axes[0].set_title(r"$k_{11}$ Component")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    # Add unit circle boundary
    outer_circle = plt.Circle(
        (0, 0), 1, fill=False, color="black", linestyle="-", linewidth=1
    )
    axes[0].add_patch(outer_circle)
    plt.colorbar(im1, ax=axes[0], label=r"$k_{11}$")

    # Plot k12 component
    im2 = axes[1].imshow(
        k12_grid, extent=[-1, 1, -1, 1], origin="lower", cmap="plasma", aspect="equal"
    )
    axes[1].contour(
        X, Y, k12_grid, levels=10, colors="white", linewidths=0.5, alpha=0.7
    )
    axes[1].set_title(r"$k_{12}$ Component")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    # Add unit circle boundary
    outer_circle = plt.Circle(
        (0, 0), 1, fill=False, color="black", linestyle="-", linewidth=1
    )
    axes[1].add_patch(outer_circle)
    plt.colorbar(im2, ax=axes[1], label=r"$k_{12}$")

    # Plot k22 component
    im3 = axes[2].imshow(
        k22_grid, extent=[-1, 1, -1, 1], origin="lower", cmap="inferno", aspect="equal"
    )
    axes[2].contour(
        X, Y, k22_grid, levels=10, colors="white", linewidths=0.5, alpha=0.7
    )
    axes[2].set_title(r"$k_{22}$ Component")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    # Add unit circle boundary
    outer_circle = plt.Circle(
        (0, 0), 1, fill=False, color="black", linestyle="-", linewidth=1
    )
    axes[2].add_patch(outer_circle)
    plt.colorbar(im3, ax=axes[2], label=r"$k_{22}$")

    fig.suptitle(title, fontsize=16)
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
    mesh, eigenvalue_dict, title="All Elements eigenvalue Number Heatmap"
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
