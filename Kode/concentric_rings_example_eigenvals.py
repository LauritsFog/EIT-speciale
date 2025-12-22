# %%
import dolfinx.mesh
from dolfinx.geometry import BoundingBoxTree, compute_colliding_cells
from fem_solver import FemConductivitySolver
import matplotlib.pyplot as plt
import numpy as np
import ufl
from plotting_functions import (
    plot_conductivity_components,
    plot_errors,
    plot_ND_map,
    plot_test_matrix_eigenvalues,
)
from conductivity_functions import (
    conductivity_AD,
)
from inner_reconstruction_method import add_noise
from inner_reconstruction_method import (
    find_inclusion_elements,
)
from scipy.optimize import curve_fit


# %%


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
    mask = (dist_to_line < line_tolerance) & (r_projections >= -0.01)

    indices_on_line = np.where(mask)[0]

    # 5. Sort by projection length (r)
    r_selected = r_projections[indices_on_line]
    sort_indices = np.argsort(r_selected)
    sorted_element_indices = indices_on_line[sort_indices]

    # 6. Extract values
    radial_distance = []
    frechet_eigenvalues_raw = []

    for idx in sorted_element_indices:
        if idx in frechet_eigenvalue_map:
            radial_distance.append(r_projections[idx])
            frechet_eigenvalues_raw.append(frechet_eigenvalue_map[idx])

    return np.array(radial_distance), np.array(frechet_eigenvalues_raw)


def get_sine_eigenvalues(ND_matrix, max_freq):
    ND_matrix_sine = ND_matrix[max_freq:, max_freq:]

    ND_eigenvals_unsorted, ND_eigenvectors = np.linalg.eig(ND_matrix_sine)

    dominant_indices = np.argmax(np.abs(ND_eigenvectors), axis=0)

    canonical_sort_permutation = np.argsort(dominant_indices)

    ND_eigenvals = ND_eigenvals_unsorted[canonical_sort_permutation]

    return np.array(ND_eigenvals)


def polar_to_euclidean(AD_conductivity, x, y):
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

    a_r, a_theta = AD_conductivity

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
    coeffs : ndarray
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

    coeffs = np.linalg.solve(A, b)
    coeffs = coeffs.tolist()
    coeffs = [(coeffs[0], 0.0), (coeffs[1], coeffs[2]), (coeffs[3], coeffs[4])]
    return coeffs


def exact_solution(x, y, n, radii, coeffs, ks):
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
    coeffs : list of tuples
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
    alpha_1, beta_1 = coeffs[2]
    k_1 = ks[2]
    v_1 = alpha_1 * r ** (k_1 * m) + beta_1 * r ** (-k_1 * m)
    v_expr += ufl.conditional(cond, v_1, 0)

    # Layer 2
    cond = ufl.And(ufl.ge(r, r1), ufl.lt(r, r2))  # r_1 <= r < r_2
    alpha_2, beta_2 = coeffs[1]
    k_2 = ks[1]
    v_2 = alpha_2 * r ** (k_2 * m) + beta_2 * r ** (-k_2 * m)
    v_expr += ufl.conditional(cond, v_2, 0)

    # Layer 3
    cond = ufl.lt(r, r1)  # r < r_1
    alpha_3, beta_3 = coeffs[0]
    k_3 = ks[0]
    v_3 = alpha_3 * r ** (k_3 * m) + beta_3 * r ** (-k_3 * m)
    v_expr += ufl.conditional(cond, v_3, 0)

    # Real-valued field (cosine mode)
    u_expr = v_expr * ufl.cos(n * theta)
    return u_expr


## Configuration

characteristic_length = 0.02

c_tol = 0
alpha_tol = 0
beta_tol = 0

# Define geometry
r1, r2 = 0.3, 0.7

# Eigenmode frequency
n = 3

shape_choice = "two_ellipses"
shape_params = (0, 0, r1, r1, 0, 0, r2, r2)

# Each layer given in polar coordinates (a_r, a_theta)

AD_conductivity = [
    (8.0, 8.0),  # 0 <= r < r1, layer 3
    (4.0, 4.0),  # r1 <= r < r2, layer 2
    (1.0, 1.0),  # r2 <= r <= 1, layer 1
]

# Anisotropy factors κ_j = sqrt(a_theta^(j) / a_r^(j))
ks = [
    np.sqrt(AD_conductivity[0][1] / AD_conductivity[0][0]),  # Layer 3
    np.sqrt(AD_conductivity[1][1] / AD_conductivity[1][0]),  # Layer 2
    np.sqrt(AD_conductivity[2][1] / AD_conductivity[2][0]),  # Layer 1
]

print("Anisotropy factors k_j: ")
print(ks)

# Check inner most layer (layer 3) first.
my_conductivity_A0 = lambda x, y: polar_to_euclidean(AD_conductivity[2], x, y)
inclusion_conductivity_1 = lambda x, y: polar_to_euclidean(AD_conductivity[0], x, y)
inclusion_conductivity_2 = lambda x, y: polar_to_euclidean(AD_conductivity[1], x, y)


def my_conductivity_AD(x, y):
    return conductivity_AD(
        x,
        y,
        my_conductivity_A0,
        (inclusion_conductivity_1, inclusion_conductivity_2),
        shape_choice,
        shape_params,
    )


coeffs = compute_solution_coefficients(r1, r2, n, AD_conductivity)
print("Solution coefficients: ")
print(coeffs[0])
print(coeffs[1])
print(coeffs[2])

(alpha1, beta1), (alpha2, beta2), (alpha3, beta3) = coeffs

# Visualization
fig, axes = plot_conductivity_components(
    my_conductivity_AD, title=f"{shape_choice} Conductivity Distribution"
)

solver_A0 = FemConductivitySolver(mesh_characteristic_length=characteristic_length)
solver_A0.set_conductivity(my_conductivity_A0, my_conductivity_A0)

mesh = solver_A0.mesh

solver_AD = FemConductivitySolver(mesh=mesh)
c, alpha, beta = solver_AD.set_conductivity(my_conductivity_AD, my_conductivity_A0)

max_freq = 5

const = c * (alpha**2) / (beta**2)

fig_components, axes_components = solver_AD.plot_conductivity_components()

## Solve with example Neumann condition

# x = ufl.SpatialCoordinate(solver_AD.get_mesh())

# nc = ufl.cos(n * ufl.atan2(x[1], x[0]))  # Neumann condition (flux) on boundary

# # Re-assemble only the RHS with new flux
# solver_AD.assemble_system(neumann_cond=nc)
# uAD = solver_AD.solve_system()


# def u_exact(x, y):
#     return exact_solution(x, y, n, [r1, r2], coeffs, ks)


# # solver_AD.plot_solution("FEM solution")
# solver_AD.plot_exact_solution(u_exact, "Exact solution")

# Plot entire boundary
# solver_AD.plot_boundary_solution_with_neumann_cond(
#     "Solution $u_f^A$ on boundary with Neumann condition"
# )

## Investigate eiganvalues

noise_level = 0.0

ntd_AD, n_freqs = solver_AD.compute_ntd_map(max_freq=max_freq)
ND_fem_eigenvals = np.diag(ntd_AD[max_freq:, max_freq:])

# ntd_AD_post_noise = add_noise(ntd_AD, noise_level)
# ND_fem_eigenvals = get_sine_eigenvalues(ntd_AD_post_noise, max_freq)

# ntd_AD_pre_noise, _ = solver_AD.compute_ntd_map(max_freq=max_freq, noise_level=noise_level)
# ND_fem_eigenvals = get_sine_eigenvalues(ntd_AD_pre_noise, max_freq)

ntd_A0, n_freqs = solver_A0.compute_ntd_map(max_freq=max_freq)

inclusion_indexes, test_matrix_eigenvalues, frechet_eigenvalues = (
    find_inclusion_elements(
        mesh=mesh,
        solutions_A0=solver_A0.basis_solutions,
        ntd_AD=ntd_AD,
        ntd_A0=ntd_A0,
        num_partitions=None,
        const=const,
        tol=0,
    )
)

radial_dist, frechet_norm = sample_radial_eigenvalues(
    mesh,
    frechet_eigenvalues,
    direction=(1.0, 0.0),  # Sample along the positive x-axis
)

k = max(ks)
C = np.max(np.abs(frechet_norm)) / np.max(
    np.min(np.abs(frechet_norm)) + np.abs(radial_dist) ** (k * max_freq)
)

radial_decay = C * np.abs(radial_dist) ** (4 * k * max_freq - 4) + np.min(
    np.abs(frechet_norm)
)

print(np.min(np.abs(frechet_norm)))

fig, ax = plt.subplots()
ax.plot(radial_dist, np.abs(frechet_norm), "r-", label="Numerical eigenvalues")
ax.plot(
    radial_dist,
    radial_decay,
    "k--",
    label="O($r^{kN}$) + $\epsilon$",
)
ax.set_xlabel("Radial Distance from Center")
ax.set_ylabel("Minimum absolute Frechet Eigenvalue")
ax.legend()
ax.set_title("Frechet Eigenvalues along Radius")
ax.grid(True)

plt.savefig("Frechet_norm_radial_decay.pdf")

plot_test_matrix_eigenvalues(mesh, frechet_eigenvalues)

ND_exact_eigenvals = []

sine_modes = np.arange(1, max_freq + 1)

for n in sine_modes:
    coeffs = compute_solution_coefficients(r1, r2, n, AD_conductivity)
    (alpha1, beta1), (alpha2, beta2), (alpha3, beta3) = coeffs

    ND_exact_eigenvals.append(alpha3 + beta3)

decay_rate = (max(ND_exact_eigenvals) + 1) / sine_modes

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.stem(
    sine_modes,
    ND_exact_eigenvals,
    basefmt=" ",
    linefmt="C0-",
    markerfmt="C0o",
    label="Exact eigenvaluess",
)
ax.stem(
    sine_modes,
    ND_fem_eigenvals,
    linefmt="C1--",
    markerfmt="C1x",
    basefmt=" ",
    label="Numerical eigenvaluess",
)
ax.plot(
    sine_modes,
    decay_rate,
    "k--",
    label="O($1/n$)",
)
ax.set_title("Exact and numerical eigenvalues of the ND map")
ax.set_xlabel("Eigenvalue index $n$")
ax.set_ylabel("Eigenvalue magnitude (Log Scale)")
ax.grid(True, which="both", ls="--")
ax.legend()
plt.yscale("log")
# plt.savefig("ND_eigenvalue_decay.pdf")

plt.show()

# %%
