import numpy as np
import matplotlib.pyplot as plt


def plot_mesh_partitions(
    mesh, partitions, title="Mesh partitions", alpha=0.5, cmap_name="gist_ncar"
):
    """
    Plot 2D mesh partitions with distinct colors (supports up to hundreds of partitions).

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The mesh to visualize.
    partitions : list[list[int]]
        List of partitions, each containing cell indices.
    title : str, optional
        Plot title.
    alpha : float, optional
        Transparency of partition colors (0–1).
    cmap_name : str, optional
        Name of Matplotlib colormap (e.g. 'gist_ncar', 'nipy_spectral', 'turbo').
        Should be continuous to support many distinct colors.
    """
    topology = mesh.topology
    tdim = topology.dim

    # Ensure cell->vertex connectivity
    if topology.connectivity(tdim, 0) is None:
        topology.create_connectivity(tdim, 0)
    connectivity = topology.connectivity(tdim, 0)
    vertices = mesh.geometry.x

    num_cells = mesh.topology.index_map(tdim).size_local
    num_partitions = len(partitions)

    # Get a continuous colormap, then sample it at evenly spaced intervals
    base_cmap = plt.get_cmap(cmap_name)
    random_vals = np.random.rand(num_partitions)
    colors = base_cmap(random_vals)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each partition with a unique color
    for p_idx, part in enumerate(partitions):
        if not part:
            continue
        color = colors[p_idx]
        for elem in part:
            if 0 <= elem < num_cells:
                cell_vertices = connectivity.links(elem)
                poly = plt.Polygon(
                    vertices[cell_vertices][:, :2],
                    fill=True,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor="black",
                    linewidth=0.1,
                )
                ax.add_patch(poly)

    # Final formatting
    ax.set_aspect("equal")
    ax.autoscale_view()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{title} ({num_partitions} partitions)")
    plt.tight_layout()

    return fig, ax


def plot_ND_map(
    ND_matrix,
    frequencies=None,
    title="Neumann-to-Dirichlet (ND) Map",
    cmap="viridis",
):
    """
    Plot the Neumann-to-Dirichlet (ND) matrix as a heatmap with frequency labels.

    Parameters
    ----------
    ND_matrix : 2D array_like
        The ND matrix (e.g., from solver.ntd_matrix).
    frequencies : array_like, optional
        The basis frequencies (e.g., from solver.basis_frequencies).
        Used to label the axes in physical mode order.
    title : str, optional
        Title for the plot.
    cmap : str, optional
        Matplotlib colormap (default: 'coolwarm').
    log_scale : bool, optional
        If True, plot log10(|ND_matrix|).
    show_colorbar : bool, optional
        Whether to include a colorbar.
    symmetric : bool, optional
        If True, set symmetric color limits around zero.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The created figure and axis objects.
    """
    ND_matrix = np.array(ND_matrix, dtype=float)

    value_label = r"$\Lambda(A)_{ij}$"

    vmin, vmax = np.min(ND_matrix), np.max(ND_matrix)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        ND_matrix, cmap=cmap, origin="upper", aspect="equal", vmin=vmin, vmax=vmax
    )

    # Axis labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Input Mode", fontsize=12)
    ax.set_ylabel("Output Mode", fontsize=12)

    # Optional frequency labels
    if frequencies is not None:
        freqs = np.array(frequencies)
        ticks = np.arange(len(freqs))
        tick_labels = [
            f"cos({n}θ)" if i < len(freqs) // 2 else f"sin({n}θ)"
            for i, n in enumerate(freqs)
        ]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(tick_labels, fontsize=9)
    else:
        ax.set_xticks(np.arange(ND_matrix.shape[1]))
        ax.set_yticks(np.arange(ND_matrix.shape[0]))

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(value_label, fontsize=12)

    ax.grid(False)
    plt.tight_layout()
    return fig, ax


def plot_errors(errors, mesh_sizes=None, labels=None, title="Error Convergence"):
    """
    Plot FEM errors on a semilogy (log-linear) plot.

    Parameters
    ----------
    errors : array_like or 2D array_like
        List/array of L² (or other) error values.
        If 2D, each row/column represents a separate error curve.
    mesh_sizes : array_like, optional
        Mesh sizes (h-values) or degrees of freedom corresponding to each error.
        If None, uses indices 1..N on the x-axis.
    labels : list of str, optional
        Legend labels for each error curve.
    title : str, optional
        Title of the plot.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The created plot objects.
    """
    errors = np.atleast_2d(np.array(errors))
    n_curves, n_points = (
        errors.shape if errors.shape[0] < errors.shape[1] else errors.T.shape
    )

    # Handle mesh sizes
    if mesh_sizes is None:
        mesh_sizes = np.arange(1, n_points + 1)
    mesh_sizes = np.array(mesh_sizes)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot each error curve
    for i in range(errors.shape[0]):
        label = labels[i] if labels is not None and i < len(labels) else None
        ax.semilogy(mesh_sizes, errors[i], "o-", linewidth=2, markersize=6, label=label)
        ax.semilogy(
            mesh_sizes,
            (np.max(errors) * 1000 * mesh_sizes) ** 2,
            "k--",
            label="O(h²)" if i == 0 else None,
        )

    ax.set_xlabel("Mesh size (h)")
    ax.set_ylabel("Error (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.7)

    if labels is not None:
        ax.legend()

    plt.tight_layout()
    return fig, ax


def plot_conductivity_components(
    conductivity_func,
    title="Conductivity Tensor s",
    n_points=500,
):
    """
     Plot the conductivity tensor components (a11, a12, a22)
     for a given conductivity function such as my_conductivity_AD(x, y, shape=...).

     Parameters
    -------
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

    # Grid setup
    xi = np.linspace(-domain_radius, domain_radius, n_points)
    yi = np.linspace(-domain_radius, domain_radius, n_points)
    X, Y = np.meshgrid(xi, yi)

    a11_grid = np.full_like(X, np.nan)
    a12_grid = np.full_like(X, np.nan)
    a22_grid = np.full_like(X, np.nan)

    # Compute conductivity tensors
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_val, y_val = X[i, j], Y[i, j]
            if np.sqrt(x_val**2 + y_val**2) <= domain_radius:
                a11, a12, a22 = conductivity_func(x_val, y_val)
                a11_grid[i, j] = a11
                a12_grid[i, j] = a12
                a22_grid[i, j] = a22

    # Plot setup
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    components = [
        (a11_grid, r"$a_{11}$", "viridis"),
        (a12_grid, r"$a_{12}$", "viridis"),
        (a22_grid, r"$a_{22}$", "viridis"),
    ]

    for ax, (grid, label, cmap) in zip(axes, components):
        im = ax.imshow(
            grid,
            extent=[-domain_radius, domain_radius, -domain_radius, domain_radius],
            origin="lower",
            cmap=cmap,
            aspect="equal",
        )
        # ax.contour(X, Y, grid, levels=50, colors="white", linewidths=0.5, alpha=0.7)
        ax.set_title(f"{label}")
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


def plot_highlighted_elements_with_inclusions(
    mesh, element_indices=None, cell_tags=None
):
    """
    Plot 2D mesh with highlighted elements and optionally color inclusion regions.
    """
    if element_indices is None:
        element_indices = []
    elif isinstance(element_indices, int):
        element_indices = [element_indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    topology = mesh.topology
    tdim = topology.dim

    # Ensure connectivity
    if topology.connectivity(tdim, 0) is None:
        topology.create_connectivity(tdim, 0)

    connectivity = topology.connectivity(tdim, 0)
    vertices = mesh.geometry.x

    num_cells = mesh.topology.index_map(tdim).size_local

    # Plot all cells
    for cell_index in range(num_cells):
        cell_vertices = connectivity.links(cell_index)
        poly = plt.Polygon(
            vertices[cell_vertices][:, :2],
            fill=None,
            edgecolor="gray",
            alpha=0.1,
            linewidth=0.4,
        )
        ax.add_patch(poly)

    # Highlight elements
    for cell_index in element_indices:
        if 0 <= cell_index < num_cells:
            cell_vertices = connectivity.links(cell_index)
            poly = plt.Polygon(
                vertices[cell_vertices][:, :2],
                fill=True,
                color="red",
                alpha=0.4,
                edgecolor="black",
                linewidth=0,
            )
            ax.add_patch(poly)
        else:
            print(f"Element index {cell_index} out of range (0-{num_cells - 1})")

    # Color inclusion cells
    if cell_tags is not None:
        inclusion_cells = np.where(cell_tags.values == 1)[0]
        for c in inclusion_cells:
            cell_vertices = connectivity.links(int(c))
            poly = plt.Polygon(
                vertices[cell_vertices][:, :2],
                fill=True,
                color="blue",
                alpha=0.4,
                edgecolor="black",
                linewidth=0,
            )
            ax.add_patch(poly)

    ax.set_aspect("equal")
    ax.autoscale_view()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Mesh with highlighted inclusions")

    return fig, ax


def plot_highlighted_elements(mesh, element_indices, inclusions=None):
    """
     Plot 2D mesh with highlighted elements using matplotlib

     Parameters:
    --------
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

    if inclusions:
        # Create grid
        x = np.linspace(-1, 1, 500)
        y = np.linspace(-1, 1, 500)
        X, Y = np.meshgrid(x, y)

        for D in inclusions:
            # Evaluate f(x, y)
            Z = D.condition(X, Y)

            # Plot contour f(x, y) = 0
            ax.contour(X, Y, Z, levels=[0], colors="blue", linewidths=2)

    return fig, ax


def plot_test_matrix_eigenvalues(
    mesh, eigenvalue_dict, sigmoid_scaling=None, title="Smallest eigenvalue per element"
):
    all_vals = np.array(list(eigenvalue_dict.values()), dtype=float)
    cell_indices = np.array(list(eigenvalue_dict.keys()))

    if sigmoid_scaling is not None:
        all_vals = 1 / (1 + np.exp(-1 * sigmoid_scaling * all_vals))

    # Create a mapping from cell index to transformed value
    transformed_dict = dict(zip(cell_indices, all_vals))

    fig, ax = plt.subplots(figsize=(10, 8))
    topology = mesh.topology
    tdim = topology.dim
    connectivity = topology.connectivity(tdim, 0)
    vertices = mesh.geometry.x

    vmin, vmax = min(all_vals), max(all_vals)
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.cm.viridis

    num_cells = mesh.topology.index_map(tdim).size_local
    for cell_idx in range(num_cells):
        cell_vertices = connectivity.links(cell_idx)
        color = (
            cmap(norm(transformed_dict[cell_idx]))
            if cell_idx in transformed_dict
            else "lightgray"
        )
        poly = plt.Polygon(
            vertices[cell_vertices][:, :2],
            fill=True,
            color=color,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.add_patch(poly)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("Eigenvalue", rotation=270, labelpad=15)

    ax.set_aspect("equal")
    ax.autoscale_view()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    return fig, ax
