import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors


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
    using a REVERSED grayscale colormap (Min=White, Max=Black)
    and a shared colorbar.
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

    # --- 1. Compute Global Min/Max for Shared Scale ---
    all_values = np.array([a11_grid, a12_grid, a22_grid])
    v_min = np.nanmin(all_values)
    v_max = np.nanmax(all_values)

    # --- 2. Use constrained_layout=True ---
    # This automatically manages space for the colorbar so it doesn't overlap
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    components = [
        (a11_grid, r"$a_{11}$"),
        (a12_grid, r"$a_{12}$"),
        (a22_grid, r"$a_{22}$"),
    ]

    im = None

    for ax, (grid, label) in zip(axes, components):
        im = ax.imshow(
            grid,
            extent=[-domain_radius, domain_radius, -domain_radius, domain_radius],
            origin="lower",
            cmap="gray_r",
            vmin=v_min,
            vmax=v_max,
            aspect="equal",
        )
        ax.set_title(f"{label}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Black circle for visibility
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

    # --- 3. Add Colorbar ---
    # With constrained_layout, we don't need 'fraction' or 'pad' as much,
    # but 'shrink' helps keep the bar from being too tall.
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, location="right")
    cbar.set_label("Conductivity value")

    fig.suptitle(f"{title}", fontsize=16)

    return fig, axes


def plot_highlighted_elements(
    mesh, element_indices, values=None, color="red", alpha=0.7
):
    """
    Plot 2D mesh with highlighted elements using matplotlib.

    Parameters:
    --------
    mesh : dolfinx.mesh.Mesh
         The mesh
    element_indices : list or int
         List of element indices to highlight
    values : list or np.ndarray, optional
         Values corresponding to element_indices. If provided, elements are
         colored using the inverted magma colormap (low=white, high=dark).
    """
    # Convert single index to list for uniform handling
    if isinstance(element_indices, int):
        element_indices = [element_indices]
        if values is not None and isinstance(values, (int, float)):
            values = [values]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the entire mesh (background)
    topology = mesh.topology
    tdim = topology.dim
    cells = topology.connectivity(tdim, 0).array.reshape(-1, tdim + 1)
    vertices = mesh.geometry.x

    # Plot background mesh edges
    for cell in cells:
        poly = plt.Polygon(
            vertices[cell][:, :2],
            fill=None,
            edgecolor="gray",
            alpha=0.3,
            linewidth=0.2,
        )
        ax.add_patch(poly)

    # Setup Colormap if values are provided
    cmap = None
    norm = None

    if values is not None:
        if len(values) != len(element_indices):
            raise ValueError("Length of 'values' must match 'element_indices'")

        # 'magma_r' is the inverted magma map.
        # Standard magma: Low=Black, High=White.
        # Inverted magma (magma_r): Low=White, High=Black.
        cmap = plt.get_cmap("magma_r")
        norm = mcolors.Normalize(vmin=np.min(values), vmax=np.max(values))

    # Highlight the specified elements
    for i, element_index in enumerate(element_indices):
        if element_index < len(cells):
            highlight_cell = cells[element_index]

            # Determine color
            if values is not None:
                # Map value to color
                face_color = cmap(norm(values[i]))
            else:
                # Use static color
                face_color = color

            poly_highlight = plt.Polygon(
                vertices[highlight_cell][:, :2],
                fill=True,
                color=face_color,
                alpha=alpha,
                linewidth=0,
            )
            ax.add_patch(poly_highlight)
        else:
            print(
                f"Warning: Element index {element_index} out of range (0-{len(cells) - 1})"
            )

    # Add Colorbar if values were used
    if values is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Value")

    # Set plot properties
    ax.set_aspect("equal")
    ax.autoscale_view()

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

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
     eigenvalue_dict : dict or np.ndarray
         Dictionary mapping element indices to values, or array of values
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

    # Prepare values
    if isinstance(eigenvalue_dict, np.ndarray):
        vals = eigenvalue_dict
    else:
        vals = np.array(list(eigenvalue_dict.values()), dtype=float)
        # Assuming keys are sorted indices if passed as dict,
        # or aligned with mesh cells if passed as array
        I = np.argsort(np.array(list(eigenvalue_dict.keys())))
        vals = vals[I]

    # Determine range for normalization based on highlighted elements
    vmin, vmax = np.min(vals[element_indices]), np.max(vals[element_indices])

    # Use inverted magma colormap (Low=White, High=Black/Dark)
    cmap = plt.cm.magma_r

    # Highlight the specified elements
    for element_index in element_indices:
        if element_index < len(cells):
            highlight_cell = cells[element_index]
            value = vals[element_index]

            # Normalize value to 0-1 range
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

    # Plot all cells (background mesh)
    for cell in cells:
        poly = plt.Polygon(
            vertices[cell][:, :2], fill=None, edgecolor="gray", alpha=0.3, linewidth=0.2
        )
        ax.add_patch(poly)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Eigenvalue", rotation=270, labelpad=15)

    # Set plot properties
    ax.set_aspect("equal")
    ax.autoscale_view()

    # set axis limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    if len(element_indices) == 1:
        title = f"Mesh with highlighted element {element_indices[0]}"
        # plt.title(title)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

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
