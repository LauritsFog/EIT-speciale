import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
    title="Conductivity Tensors",
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


def plot_highlighted_elements(
    mesh, element_indices, inclusions=None, color="red", alpha=0.7
):
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

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the entire mesh
    topology = mesh.topology
    tdim = topology.dim
    cells = topology.connectivity(tdim, 0).array.reshape(-1, tdim + 1)
    vertices = mesh.geometry.x

    # Plot all cells (background mesh)
    for cell in cells:
        poly = plt.Polygon(
            vertices[cell][:, :2],
            fill=None,
            edgecolor="black",
            alpha=0.3,
            linewidth=0.4,
        )
        ax.add_patch(poly)

    # Highlight the specified elements in red
    for element_index in element_indices:
        if element_index < len(cells):
            highlight_cell = cells[element_index]
            poly_highlight = plt.Polygon(
                vertices[highlight_cell][:, :2],
                fill=True,
                color=color,
                alpha=alpha,
                linewidth=0,
            )
            ax.add_patch(poly_highlight)
        else:
            print(
                f"Warning: Element index {element_index} out of range (0-{len(cells) - 1})"
            )

    # Set plot properties
    ax.set_aspect("equal")
    ax.autoscale_view()

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


def plot_highlighted_eigenvalues(mesh, element_indices, eigenvalue_dict):
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

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the entire mesh
    topology = mesh.topology
    tdim = topology.dim
    cells = topology.connectivity(tdim, 0).array.reshape(-1, tdim + 1)
    vertices = mesh.geometry.x

    # Plot all cells (background mesh)
    for cell in cells:
        poly = plt.Polygon(
            vertices[cell][:, :2], fill=None, edgecolor="gray", alpha=0.3, linewidth=0.2
        )
        ax.add_patch(poly)

    if isinstance(eigenvalue_dict, np.ndarray):
        vals = eigenvalue_dict
    else:
        vals = np.array(list(eigenvalue_dict.values()), dtype=float)
        I = np.argsort(np.array(list(eigenvalue_dict.keys())))
        vals = vals[I]
    vmin, vmax = np.min(vals[element_indices]), np.max(vals[element_indices])
    cmap = plt.cm.viridis

    # Highlight the specified elements in red
    for element_index in element_indices:
        if element_index < len(cells):
            highlight_cell = cells[element_index]
            value = vals[element_index]
            norm_value = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            color = cmap(norm_value)
            poly_highlight = plt.Polygon(
                vertices[highlight_cell][:, :2], fill=True, color=color, alpha=1
            )
            ax.add_patch(poly_highlight)
        else:
            print(
                f"Warning: Element index {element_index} out of range (0-{len(cells) - 1})"
            )

    # add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Eigenvalue", rotation=270, labelpad=15)

    # Set plot properties
    ax.set_aspect("equal")
    ax.autoscale_view()

    # set axis limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # title = f"Mesh with {len(element_indices)} highlighted elements"
    if len(element_indices) == 1:
        title = f"Mesh with highlighted element {element_indices[0]}"
    # plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    return fig, ax


def plot_inclusion_reconstruction_error(
    alphas, reconstruction_measures, alpha_optim=None, alpha_alt=None
):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, reconstruction_measures, "-", linewidth=2, markersize=6)

    if alpha_optim is not None:
        ax.axvline(
            x=alpha_optim,
            color="red",
            linestyle="--",
            label=f"Optimal alpha = {alpha_optim:.6e}",
        )
        ax.legend()

    if alpha_alt is not None:
        ax.axvline(
            x=alpha_alt,
            color="blue",
            linestyle="--",
            label=f"Sub-optimal alpha = {alpha_alt:.6e}",
        )
        ax.legend()

    ax.set_xlim(min(alphas), max(alphas))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.6e"))
    ax.tick_params(axis="x", labelrotation=45)

    ax.set_xlabel("Regularization parameter (alpha)")
    ax.set_ylabel("Reconstruction error")
    ax.grid(True, which="both", linestyle="--", alpha=0.7)

    plt.tight_layout()

    return fig, ax
