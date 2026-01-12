# %%

from fem_solver import FemConductivitySolver, interpolate_solutions_to_new_mesh
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import ufl
from plotting_functions import (
    plot_conductivity_components,
)
from conductivity_functions import (
    conductivity_AD,
)
from inner_reconstruction_method import (
    compute_reconstruction_test_values,
    reconstruct_inclusions,
    add_noise,
    precompute_gradients,
)


# %%
def estimate_convergence_rate(h_values, error_values):
    """
    Fits the power law E = C * h^k to the data.

    Parameters:
    h_values (list or array): Mesh sizes (h)
    error_values (list or array): Corresponding errors

    Returns:
    C (float): The proportionality constant
    k (float): The convergence rate (slope in log-log plot)
    """
    h = np.array(h_values)
    e = np.array(error_values)

    # Remove any zero or negative values to avoid log errors
    mask = (h > 0) & (e > 0)
    h_clean = h[mask]
    e_clean = e[mask]

    if len(h_clean) < 2:
        return np.nan, np.nan

    # Linear regression in log-log space: ln(E) = k * ln(h) + ln(C)
    # polyfit returns [slope, intercept]
    k, ln_C = np.polyfit(np.log(h_clean), np.log(e_clean), 1)

    return np.exp(ln_C), k


def sample_radial_eigenvalues(
    mesh,
    frechet_eigenvalue_map,
    direction=(1.0, 0.0),
):
    tdim = mesh.topology.dim
    num_elements = mesh.topology.index_map(tdim).size_local
    connectivity = mesh.topology.connectivity(tdim, 0)
    x = mesh.geometry.x

    all_centroids = np.zeros((num_elements, 2))

    # 1. Compute Centroids
    for i in range(num_elements):
        vertex_indices = connectivity.links(i)
        vertex_coords = x[vertex_indices][:, :2]
        all_centroids[i] = np.mean(vertex_coords, axis=0)

    # 2. Define the line vector (normalized)
    u = np.array(direction)
    u = u / np.linalg.norm(u)

    # 3. Calculate perpendicular distance of every centroid to the line
    # For a line through origin with unit direction u, distance d = |c - (c·u)u|
    # Or using 2D cross product: d = |x*uy - y*ux|
    dist_to_line = np.abs(all_centroids[:, 0] * u[1] - all_centroids[:, 1] * u[0])

    # Calculate projection length along the line (radial distance)
    r_projections = all_centroids[:, 0] * u[0] + all_centroids[:, 1] * u[1]

    # 4. Selection Logic:
    # We pick elements whose centroids are very close to the line.
    # To handle varying mesh density, we use a tolerance related to the average mesh size
    # or a small fraction of the radius.
    line_tolerance = 0.02  # Adjust this: 2% of the radius

    # Also ensure the projection is positive (going in the right direction)
    mask = (dist_to_line < line_tolerance) & (r_projections > 0)

    indices_on_line = np.where(mask)[0]

    # 5. Sort by projection length (r)
    r_selected = r_projections[indices_on_line]
    sort_indices = np.argsort(r_selected)
    sorted_element_indices = indices_on_line[sort_indices]

    # 6. Extract values
    radial_distance = []
    frechet_quantity_raw = []

    for idx in sorted_element_indices:
        if idx in frechet_eigenvalue_map:
            radial_distance.append(r_projections[idx])
            frechet_quantity_raw.append(frechet_eigenvalue_map[idx])

    return np.array(radial_distance), np.array(frechet_quantity_raw)


def get_sine_eigenvalues(ND_matrix, max_freq):
    ND_matrix_sine = ND_matrix[max_freq:, max_freq:]

    ND_eigenvals_unsorted, ND_eigenvectors = np.linalg.eig(ND_matrix_sine)

    dominant_indices = np.argmax(np.abs(ND_eigenvectors), axis=0)

    canonical_sort_permutation = np.argsort(dominant_indices)

    ND_eigenvals = ND_eigenvals_unsorted[canonical_sort_permutation]

    return np.array(ND_eigenvals)


def polar_to_euclidean(polar_conductivities_AD, x, y):
    """
    Convert a diagonal polar conductivity tensor diag(a_r, a_theta)
    into its Cartesian (x, y) coordinate form, automatically computing theta.

    Parameters
    ----------
    a_r : float
        Radial conductivity component.
    a_theta : float
        Tangential (angular) conductivity component.
    x, y : float or array_like
        Cartesian coordinates.

    Returns
    -------
    A : Euclidean conducitivity tensor components (Axx, Axy, Ayy)
    """

    a_r, a_theta = polar_conductivities_AD

    # Compute polar angle
    theta = np.arctan2(y, x)

    # Compute cos/sin
    c = np.cos(theta)
    s = np.sin(theta)

    # Transform to Euclidean
    Axx = a_r * c**2 + a_theta * s**2
    Axy = (a_r - a_theta) * s * c
    Ayy = a_r * s**2 + a_theta * c**2

    return (Axx, Axy, Ayy)


def compute_solution_coefficients(r1, r2, n, polar_mats):
    """
    Compute coefficients [alpha1, alpha2, beta2, alpha3, beta3]
    for a 3-layer radially anisotropic disk with Neumann BC.

    Parameters
    a_thetaa_thetaa_theta-
    r1, r2 : float
        Radii separating layers (0 < r1 < r2 < 1)
    n : int
        Fourier mode index (n ≠ 0)
    polar_mats : list of 3 tuples
        Each tuple is the diagonal polar conductivity matrix for one layer:
        [(a_r1, a_theta1), (a_r2, a_theta2), (a_r3, a_theta3)]

    Returns
    a_thetaa_theta-
    coeffs_AD : ndarray
        [alpha_1, alpha_2, beta_2, alpha_3, beta_3]
    """
    m = abs(n)

    # Extract radial and tangential conductivities
    a_r = np.array([s[0] for s in polar_mats])
    a_theta = np.array([s[1] for s in polar_mats])

    # Anisotropy parameter
    kappa = np.sqrt(a_theta / a_r)

    # Precompute powers
    r1_k1 = r1 ** (kappa[0] * m)
    r1_k2 = r1 ** (kappa[1] * m)
    r2_k2 = r2 ** (kappa[1] * m)
    r2_k3 = r2 ** (kappa[2] * m)

    # Unknowns: [alpha1, alpha2, beta2, alpha3, beta3]
    A = np.zeros((5, 5))
    b = np.zeros(5)

    # a_theta Interface 1↔2 (continuity) a_theta
    A[0, 0] = r1_k1
    A[0, 1] = -r1_k2
    A[0, 2] = -(r1 ** (-kappa[1] * m))

    # a_theta Interface 1↔2 (flux continuity) a_theta
    A[1, 0] = a_r[0] * kappa[0] * r1 ** (kappa[0] * m - 1)
    A[1, 1] = -a_r[1] * kappa[1] * r1 ** (kappa[1] * m - 1)
    A[1, 2] = a_r[1] * kappa[1] * r1 ** (-kappa[1] * m - 1)

    # a_theta Interface 2↔3 (continuity) a_theta
    A[2, 1] = r2_k2
    A[2, 2] = r2 ** (-kappa[1] * m)
    A[2, 3] = -r2_k3
    A[2, 4] = -(r2 ** (-kappa[2] * m))

    # a_theta Interface 2↔3 (flux continuity) a_theta
    A[3, 1] = a_r[1] * kappa[1] * r2 ** (kappa[1] * m - 1)
    A[3, 2] = -a_r[1] * kappa[1] * r2 ** (-kappa[1] * m - 1)
    A[3, 3] = -a_r[2] * kappa[2] * r2 ** (kappa[2] * m - 1)
    A[3, 4] = a_r[2] * kappa[2] * r2 ** (-kappa[2] * m - 1)

    # a_theta Neumann BC a_theta
    A[4, 3] = a_r[2] * kappa[2] * m
    A[4, 4] = -a_r[2] * kappa[2] * m
    b[4] = 1.0

    coeffs_AD = np.linalg.solve(A, b)
    coeffs_AD = coeffs_AD.tolist()
    coeffs_AD = [
        (coeffs_AD[0], 0.0),
        (coeffs_AD[1], coeffs_AD[2]),
        (coeffs_AD[3], coeffs_AD[4]),
    ]
    return coeffs_AD


def exact_solution(x, y, n, radii, coeffs_AD, ks):
    """
    UFL version of the layered analytic solution:
    u_n(x,y) = v_n(r) * cos(n * theta)  (real-valued)

    Parameters
    ----------
    x, y : UFL expressions
        Symbolic coordinates (from ufl.SpatialCoordinate()).
    n : int
        Fourier mode number.
    radii : list or tuple of floats
        Layer boundaries [r1, r2, ...].
    coeffs_AD : list of tuples
        [(alpha1, beta1), (alpha2, beta2), (alpha3, beta3), ...].
    ks : list of floats
        Anisotropy exponents k_j for each layer.

    Returns
    -------
    u_expr : UFL expression
        Symbolic real-valued expression for u(x, y).
    """
    r = ufl.sqrt(x**2 + y**2)
    theta = ufl.atan2(y, x)
    m = abs(n)

    # Initialize v_expr
    v_expr = 0.0

    r1, r2 = radii

    # Layer 1
    cond = ufl.ge(r, r2)  # r >= r_2
    alpha_1, beta_1 = coeffs_AD[2]
    k_1 = ks[2]
    v_1 = alpha_1 * r ** (k_1 * m) + beta_1 * r ** (-k_1 * m)
    v_expr += ufl.conditional(cond, v_1, 0)

    # Layer 2
    cond = ufl.And(ufl.ge(r, r1), ufl.lt(r, r2))  # r_1 <= r < r_2
    alpha_2, beta_2 = coeffs_AD[1]
    k_2 = ks[1]
    v_2 = alpha_2 * r ** (k_2 * m) + beta_2 * r ** (-k_2 * m)
    v_expr += ufl.conditional(cond, v_2, 0)

    # Layer 3
    cond = ufl.lt(r, r1)  # r < r_1
    alpha_3, beta_3 = coeffs_AD[0]
    k_3 = ks[0]
    v_3 = alpha_3 * r ** (k_3 * m) + beta_3 * r ** (-k_3 * m)
    v_expr += ufl.conditional(cond, v_3, 0)

    # Real-valued field (cosine mode)
    u_expr = v_expr * ufl.cos(n * theta)
    return u_expr


### Conductivity

# Define geometry
r1, r2 = 0.3, 0.7

shape_choice = "two_ellipses"
shape_params = (0, 0, r1, r1, 0, 0, r2, r2)

# Each layer given in polar coordinates (a_r, a_theta)
# polar_conductivities_AD = [
#     (1.0, 1.0),  # 0 <= r < r1, layer 3
#     (1.0, 1.0),  # r1 <= r < r2, layer 2
#     (1.0, 1.0),  # r2 <= r <= 1, layer 1
# ]

polar_conductivities_AD = [
    (16.0, 12.0),  # 0 <= r < r1, layer 3
    (4.0, 8.0),  # r1 <= r < r2, layer 2
    (1.0, 1.0),  # r2 <= r <= 1, layer 1
]

polar_conductivities_A0 = [
    polar_conductivities_AD[2],
    polar_conductivities_AD[2],
    polar_conductivities_AD[2],
]

# Anisotropy factors κ_j = sqrt(a_theta^(j) / a_r^(j))
ks_AD = [
    np.sqrt(polar_conductivities_AD[0][1] / polar_conductivities_AD[0][0]),  # Layer 3
    np.sqrt(polar_conductivities_AD[1][1] / polar_conductivities_AD[1][0]),  # Layer 2
    np.sqrt(polar_conductivities_AD[2][1] / polar_conductivities_AD[2][0]),  # Layer 1
]

ks_A0 = [
    np.sqrt(polar_conductivities_A0[0][1] / polar_conductivities_A0[0][0]),  # Layer 3
    np.sqrt(polar_conductivities_A0[1][1] / polar_conductivities_A0[1][0]),  # Layer 2
    np.sqrt(polar_conductivities_A0[2][1] / polar_conductivities_A0[2][0]),  # Layer 1
]

k = ks_A0[2]  # Outer layer anisotropy

### Setings

noise_level = 0.0

max_freq = 25
modes = range(1, max_freq + 1)

# Eigenmode frequency
n = 3

h_array = [0.06]  # , 0.04, 0.02, 0.01, 0.005]
h_array = np.array(h_array)

recon_h = 0.061
recon_solver = FemConductivitySolver(mesh_characteristic_length=recon_h)
recon_mesh = recon_solver.mesh

eigenvalue_error_scaling = (h_array[0] * np.array(modes)) ** 2

# Frechet quantity decay (must be smaller that max_freq)
frechet_diag_mode = 8
r = np.linspace(0, 1, 100)

radial_frechet_decay = (h_array[-1] ** 2) * (
    np.abs(r) ** (k * 2 * frechet_diag_mode - 2)
)

# Check inner most layer (layer 3) first.
my_conductivity_A0 = lambda x, y: polar_to_euclidean(polar_conductivities_AD[2], x, y)
inclusion_conductivity_1 = lambda x, y: polar_to_euclidean(
    polar_conductivities_AD[0], x, y
)
inclusion_conductivity_2 = lambda x, y: polar_to_euclidean(
    polar_conductivities_AD[1], x, y
)


def my_conductivity_AD(x, y):
    return conductivity_AD(
        x,
        y,
        my_conductivity_A0,
        (inclusion_conductivity_1, inclusion_conductivity_2),
        shape_choice,
        shape_params,
    )


coeffs_AD = compute_solution_coefficients(r1, r2, n, polar_conductivities_AD)
coeffs_A0 = compute_solution_coefficients(r1, r2, n, polar_conductivities_A0)

# Visualization
fig, axes = plot_conductivity_components(my_conductivity_AD, title="")
plt.savefig("Conductivity_distribution_analytic_example.pdf")


def u_exact_AD(x, y):
    return exact_solution(x, y, n, [r1, r2], coeffs_AD, ks_AD)


def u_exact_A0(x, y):
    return exact_solution(x, y, n, [r1, r2], coeffs_A0, ks_A0)


errors_AD = np.zeros((len(h_array), 1))
errors_A0 = np.zeros((len(h_array), 1))

eigenval_errors_AD = np.zeros((max_freq, len(h_array)))
eigenval_errors_A0 = np.zeros((max_freq, len(h_array)))

radial_dist_lists = []
frechet_quantity_lists = []

for i, cl in enumerate(h_array):
    print(f"Characeristic length: {cl}")

    solver_AD = FemConductivitySolver(mesh_characteristic_length=cl)
    c, alpha, beta = solver_AD.set_conductivity(my_conductivity_AD, my_conductivity_A0)
    const = c * (alpha**2) / (beta**2)

    mesh = solver_AD.get_mesh()

    solver_A0 = FemConductivitySolver(mesh=mesh)
    solver_A0.set_conductivity(my_conductivity_A0, my_conductivity_A0)

    x = ufl.SpatialCoordinate(mesh)
    nc = ufl.cos(n * ufl.atan2(x[1], x[0]))

    solver_AD.assemble_system(neumann_cond=nc)
    solver_AD.solve_system()

    solver_A0.assemble_system(neumann_cond=nc)
    solver_A0.solve_system()

    error_AD = solver_AD.compute_error(u_exact_AD, norm_type="H1")
    errors_AD[i] = error_AD

    error_A0 = solver_A0.compute_error(u_exact_A0, norm_type="H1")
    errors_A0[i] = error_A0

    ntd_AD, n_freqs = solver_AD.compute_ntd_map(max_freq=max_freq)
    ntd_A0, n_freqs = solver_A0.compute_ntd_map(max_freq=max_freq)

    if noise_level > 0:
        ntd_AD = add_noise(ntd_AD, noise_level=noise_level)

        ntd_fem_eigenvalues_AD = get_sine_eigenvalues(ntd_AD, max_freq)

        ntd_A0 = add_noise(ntd_A0, noise_level=noise_level)

        ntd_fem_eigenvalues_A0 = get_sine_eigenvalues(ntd_A0, max_freq)
    else:
        ntd_fem_eigenvalues_AD = np.diag(ntd_AD[max_freq:, max_freq:])

        ntd_fem_eigenvalues_A0 = np.diag(ntd_A0[max_freq:, max_freq:])

    ntd_exact_eigenvalues_AD = []
    ntd_exact_eigenvalues_A0 = []

    for mode in modes:
        coeffs_temp = compute_solution_coefficients(
            r1, r2, mode, polar_conductivities_AD
        )
        (alpha1, beta1), (alpha2, beta2), (alpha3, beta3) = coeffs_temp

        ntd_exact_eigenvalues_AD.append(alpha3 + beta3)

        coeffs_temp = compute_solution_coefficients(
            r1, r2, mode, polar_conductivities_A0
        )
        (alpha1, beta1), (alpha2, beta2), (alpha3, beta3) = coeffs_temp

        ntd_exact_eigenvalues_A0.append(alpha3 + beta3)

    eigenval_errors_AD[:, i] = (
        ntd_exact_eigenvalues_AD - ntd_fem_eigenvalues_AD
    ) / np.abs(ntd_exact_eigenvalues_AD)

    eigenval_errors_A0[:, i] = (
        ntd_exact_eigenvalues_A0 - ntd_fem_eigenvalues_A0
    ) / np.abs(ntd_exact_eigenvalues_A0)

    recon_basis_solutions = interpolate_solutions_to_new_mesh(solver_A0, recon_solver)

    gradients = precompute_gradients(recon_basis_solutions, recon_mesh)

    # Call the function directly
    eigenvalue_dict, frechet_quantity_dict = compute_reconstruction_test_values(
        mesh=recon_mesh,
        gradients=gradients,
        ntd_AD=ntd_AD,
        ntd_A0=ntd_A0,
        rho=const,
    )

    radial_dist, frechet_quantity = sample_radial_eigenvalues(
        recon_mesh,
        frechet_quantity_dict.values(),
        direction=(1.0, 0.0),  # Sample along the positive x-axis
    )

    print(frechet_quantity)

    radial_dist_lists.append(radial_dist)
    frechet_quantity_lists.append(frechet_quantity)

    solver_AD.cleanup()
    solver_A0.cleanup()

### Plot FEM solution errors

c_AD, k_AD = estimate_convergence_rate(h_array, errors_AD.flatten())
c_A0, k_A0 = estimate_convergence_rate(h_array, errors_A0.flatten())

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(
    h_array,
    errors_AD,
    "o-",
    linewidth=2,
    markersize=6,
    label="$u_h^{A_D}$",
)
ax.semilogy(
    h_array,
    errors_A0,
    "o-",
    linewidth=2,
    markersize=6,
    label="$u_h^{A_0}$",
)
ax.semilogy(
    h_array,
    (c_AD * 2 * h_array) ** (k_AD),
    "k--",
    label=f"O($h^{{{round(k_AD, 1)}}}$)",
)
ax.semilogy(
    h_array,
    (c_A0 * 5 * h_array) ** (k_A0),
    "k:",
    label=f"O($h^{{{round(k_A0, 1)}}}$)",
)
ax.set_xlabel("Mesh size (h)")
ax.set_ylabel("$H^1$-error (log scale)")
ax.grid(True, which="both", linestyle="--", alpha=0.7)
ax.legend()
plt.savefig("FEM_solution_error_convergence.pdf")

### Plot smallest eigenvalue errors

min_eigval_errors_AD = np.min(np.abs(eigenval_errors_AD), axis=0)
min_eigval_errors_A0 = np.min(np.abs(eigenval_errors_A0), axis=0)

c_AD, k_AD = estimate_convergence_rate(h_array, min_eigval_errors_AD)
c_A0, k_A0 = estimate_convergence_rate(h_array, min_eigval_errors_A0)

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(
    h_array,
    min_eigval_errors_AD,
    "o-",
    linewidth=2,
    markersize=6,
    label="$u_h^{A_D}$",
)
ax.semilogy(
    h_array,
    min_eigval_errors_A0,
    "o-",
    linewidth=2,
    markersize=6,
    label="$u_h^{A_0}$",
)
ax.semilogy(
    h_array,
    (c_AD * 5 * h_array) ** (k_AD),
    "k--",
    label=f"O($h^{{{round(k_AD, 1)}}}$)",
)
ax.semilogy(
    h_array,
    (c_A0 * 10 * h_array) ** (k_A0),
    "k:",
    label=f"O($h^{{{round(k_A0, 1)}}}$)",
)
ax.set_xlabel("Mesh size (h)")
ax.set_ylabel("Absolute error (log scale)")
ax.grid(True, which="both", linestyle="--", alpha=0.7)
ax.legend()
plt.savefig("ND_map_smallest_eigenvalue_error_convergence.pdf")

### Plot eigenvalue error for each mode

idx_start = 8
c_AD, k_AD = estimate_convergence_rate(
    modes[idx_start:], np.abs(eigenval_errors_AD[idx_start:, 0])
)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

for i in range(len(h_array)):
    ax.semilogy(
        modes,
        np.abs(eigenval_errors_AD[:, i]),
        "o-",
        linewidth=2,
        markersize=4,
        label=f"Mesh size (h) = {h_array[i]}",
    )
ax.semilogy(
    modes,
    (c_AD * 100 * modes) ** (k_AD),
    "k--",
    linewidth=2,
    markersize=4,
    label=f"O($n^{{{round(k_AD, 1)}}}$)",
)
ax.set_xlabel("Eigenvalue index ($n$)")
ax.set_ylabel("Absolute relative error (Log Scale)")
ax.grid(True, which="both", ls="--")
ax.legend()

plt.savefig("ND_eigenvalue_relative_error_scaling_AD.pdf")

idx_start = 8
c_A0, k_A0 = estimate_convergence_rate(
    modes[idx_start:], np.abs(eigenval_errors_A0[idx_start:, 0])
)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

for i in range(len(h_array)):
    ax.semilogy(
        modes,
        np.abs(eigenval_errors_A0[:, i]),
        "o-",
        linewidth=2,
        markersize=4,
        label=f"Mesh size (h) = {h_array[i]}",
    )
ax.semilogy(
    modes,
    (c_A0 * 100 * modes) ** (k_A0),
    "k--",
    linewidth=2,
    markersize=4,
    label=f"O($n^{{{round(k_A0, 1)}}}$)",
)
ax.set_xlabel("Eigenvalue index ($n$)")
ax.set_ylabel("Absolute relative error (Log Scale)")
ax.grid(True, which="both", ls="--")
ax.legend()

plt.savefig("ND_eigenvalue_relative_error_scaling_A0.pdf")

### Plot signed eigenvalue error for each mode

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

for i in range(len(h_array)):
    ax.plot(
        modes,
        eigenval_errors_AD[:, i],
        "o-",
        linewidth=2,
        markersize=4,
        label=f"Mesh size (h) = {h_array[i]}",
    )

ax.set_xlabel("Eigenvalue index ($n$)")
ax.set_ylabel("Signed relative error")
ax.grid(True, which="both", ls="--")
ax.legend()

# plt.savefig("ND_eigenvalue_signed_relative_error.pdf")

### Plot frechet matrix eigenvalues

# plot_test_matrix_eigenvalues(
#     mesh, frechet_quantity, title="N'th diagonal element of the Frechet derivative"
# )

# plt.savefig("Frechet_derivative_plot.pdf")

### Plot ND map eigenvalue decay

sine_modes = np.arange(1, max_freq + 1)

decay_rate = (max(ntd_exact_eigenvalues_AD) + 1) / sine_modes

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.semilogy(
    sine_modes,
    ntd_fem_eigenvalues_AD,
    "-",
    label="Numerical eigenvaluess",
)
ax.semilogy(
    sine_modes,
    ntd_exact_eigenvalues_AD,
    "--",
    label="Exact eigenvalues",
)
ax.semilogy(
    sine_modes,
    decay_rate,
    "k--",
    label="O($1/n$)",
)
ax.set_xlabel("Eigenvalue index ($n$)")
ax.set_ylabel("Eigenvalue magnitude")
ax.grid(True, which="both", ls="--")
ax.legend()

plt.savefig("ND_eigenvalue_decay_AD.pdf")

### Plot Frechet eigenvalues along radial line for the finest mesh

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for i in range(len(h_array)):
    ax.semilogy(
        radial_dist_lists[i],
        np.abs(frechet_quantity_lists[i]),
        "-",
        label=f"Mesh size (h) = {h_array[i]}",
    )
ax.semilogy(
    r,
    radial_frechet_decay,
    "k--",
    label="O($r^{2kN-2}$)",
)
ax.set_xlabel("Radial distance from center ($r$)")
ax.set_ylabel("Absolute value of N'th diagonal element (log scale)")
ax.legend()
ax.grid(True)

plt.savefig("Frechet_diagonal_radial_decay.pdf")

plt.show()
