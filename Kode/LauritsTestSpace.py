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
    h,
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
    line_tolerance = h * 0.5

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
        # if idx in frechet_eigenvalue_map:
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


def polar_to_euclidean(polar_conductivities_A, x, y):
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

    a_r, a_theta = polar_conductivities_A

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


def map_forward(x, y, s):
    """
    Maps (x, y) -> (r, theta) -> (r^s, theta) -> (x_new, y_new)
    """
    # 1. Cartesian to Polar
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Handle origin to avoid numerical errors
    if r < 1e-12:
        return 0.0, 0.0

    # 2. Apply Diffeomorphism (Forward)
    # R(r) = r^s
    r_new = r**s

    # 3. Polar to Cartesian (using the NEW radius)
    x_new = r_new * np.cos(theta)
    y_new = r_new * np.sin(theta)

    return x_new, y_new


### Conductivity

# Define geometry
r1, r2 = 0.3, 0.7

shape_choice = "two_ellipses"
shape_params = (0, 0, r1, r1, 0, 0, r2, r2)

# Each layer given in polar coordinates (a_r, a_theta)
polar_conductivities_A = [
    (16.0, 12.0),  # 0 <= r < r1, layer 3
    (4.0, 8.0),  # r1 <= r < r2, layer 2
    (1.0, 1.0),  # r2 <= r <= 1, layer 1
]

s = 2.0

polar_conductivities_Atilde = [
    (s * polar_conductivities_A[0][0], 1 / s * polar_conductivities_A[0][1]),
    (s * polar_conductivities_A[1][0], 1 / s * polar_conductivities_A[1][1]),
    (s * polar_conductivities_A[2][0], 1 / s * polar_conductivities_A[2][1]),
]

### Setings

cl = 0.05

max_freq = 5
modes = range(1, max_freq + 1)

# Check inner most layer (layer 3) first.
A0 = lambda x, y: polar_to_euclidean(polar_conductivities_A[2], x, y)
inclusion_conductivity_1_A = lambda x, y: polar_to_euclidean(
    polar_conductivities_A[0], x, y
)
inclusion_conductivity_2_A = lambda x, y: polar_to_euclidean(
    polar_conductivities_A[1], x, y
)

A0tilde = lambda x, y: polar_to_euclidean(polar_conductivities_Atilde[2], x, y)
inclusion_conductivity_1_Atilde = lambda x, y: polar_to_euclidean(
    polar_conductivities_Atilde[0], x, y
)
inclusion_conductivity_2_Atilde = lambda x, y: polar_to_euclidean(
    polar_conductivities_Atilde[1], x, y
)


def my_conductivity_A(x, y):
    return conductivity_AD(
        x,
        y,
        A0,
        (inclusion_conductivity_1_A, inclusion_conductivity_2_A),
        shape_choice,
        shape_params,
    )


def my_conductivity_Atilde(x, y):
    x, y = map_forward(x, y, 1.0 / s)
    return conductivity_AD(
        x,
        y,
        A0tilde,
        (inclusion_conductivity_1_Atilde, inclusion_conductivity_2_Atilde),
        shape_choice,
        shape_params,
    )


solver_A = FemConductivitySolver(mesh_characteristic_length=cl)
solver_A.set_conductivity(my_conductivity_A, A0)

mesh = solver_A.get_mesh()

solver_Atilde = FemConductivitySolver(mesh=mesh)
solver_Atilde.set_conductivity(my_conductivity_Atilde, A0tilde)

ntd_A, n_freqs = solver_A.compute_ntd_map(max_freq=max_freq)
ntd_Atilde, n_freqs = solver_Atilde.compute_ntd_map(max_freq=max_freq)

# Magnitude of Lambda(A) - Lambda(Atilde) eigenvalues
print(np.linalg.norm(ntd_A - ntd_Atilde, ord=2))

print(np.diag(ntd_A))
print(np.diag(ntd_Atilde))

fig, axes = plot_conductivity_components(my_conductivity_A, title="A")

fig, axes = plot_conductivity_components(my_conductivity_Atilde, title="")

plt.show()

# characteristic_length = 0.02
# recon_h = 0.05

# max_freq = 8

# # min_eig -> 1 / (1 + np.exp(-sigmoid_scaling * min_eig)). Set to None for no scaling
# sigmoid_scaling = 1e7

# # Tolerance in the inclusion test: min(eigenvalues) + tol > 0
# delta = 1e-6

# c_tol = 0
# alpha_tol = 0
# beta_tol = 0

# shape_choice = "ellipse"
# shape_params = ((0.2, 0.2), (0.3, 0.3))  # Center, axes

# # Define conductivities
# my_conductivity_Atilde = lambda x, y: (1, 0, 1)

# inclusion_conductivity_1 = lambda x, y: (5, 0, 2)


# def my_conductivity_AD(x, y):
#     return conductivity_AD(
#         x,
#         y,
#         my_conductivity_Atilde,
#         inclusion_conductivity_1,
#         shape_choice,
#         shape_params,
#     )


# # shape_choice = "two_ellipses"
# # shape_params = (-0.4, -0.4, 0.3, 0.2, 0.4, 0.4, 0.3, 0.2)  # Center, axis, center, axis

# # my_conductivity_Atilde = lambda x, y: (1, 0, 1)

# # inclusion_conductivity_1 = lambda x, y: (15, 0, 5)
# # inclusion_conductivity_2 = lambda x, y: (5, 0, 15)


# # def my_conductivity_AD(x, y):
# #     return conductivity_AD(
# #         x,
# #         y,
# #         my_conductivity_Atilde,
# #         (inclusion_conductivity_1, inclusion_conductivity_2),
# #         shape_choice,
# #         shape_params,
# #     )


# # %%

# solverA0 = FemConductivitySolver(mesh_characteristic_length=characteristic_length)
# solverA0.set_conductivity(my_conductivity_Atilde, my_conductivity_Atilde)

# recon_solver = FemConductivitySolver(mesh_characteristic_length=recon_h)
# recon_solver.set_conductivity(my_conductivity_AD, my_conductivity_Atilde)
# recon_mesh = recon_solver.mesh

# mesh = solverA0.mesh

# # Computing the "optimal" alpha, beta, c constants from checking the conductivity on each element
# solverAD = FemConductivitySolver(mesh=mesh)
# c, alpha, beta = solverAD.set_conductivity(my_conductivity_AD, my_conductivity_Atilde)

# solverA0.compute_ntd_map(max_freq=max_freq)
# solverAD.compute_ntd_map(max_freq=max_freq)

# recon_basis_solutions = interpolate_solutions_to_new_mesh(solverA0, recon_solver)

# # AD-A0 >= cI and alpha I <= AD <= beta I

# c = c - c_tol
# alpha = alpha - alpha_tol
# beta = beta + beta_tol

# rho = c * (alpha**2) / (beta**2)

# ntd_A0 = solverA0.ntd_matrix
# ntd_AD = solverAD.ntd_matrix

# ntd_AD_noise = add_noise(ntd_AD, noise_level=delta)

# plot_conductivity_components(my_conductivity_AD, title="")

# # # Solve with different boundary fluxes by modifying b
# # x = ufl.SpatialCoordinate(solverA0.get_mesh())

# # nc = ufl.cos(3 * ufl.atan2(x[1], x[0]))  # Neumann condition (flux) on boundary

# # # Re-assemble only the RHS with new flux
# # solverAD.assemble_system(neumann_cond=nc)
# # uAD = solverAD.solve_system()

# # # Get plot objects WITHOUT displaying them
# # solverAD.plot_solution("Solution $u_f^A$")

# # # Plot entire boundary
# # solverAD.plot_boundary_solution_with_neumann_cond(
# #     "Solution $u_f^A$ on boundary with Neumann condition"
# # )

# gradients = precompute_gradients(recon_basis_solutions, recon_mesh)

# # Call the function directly
# eigenvalue_dict, frechet_quantity_dict = compute_reconstruction_test_values(
#     mesh=recon_mesh,
#     gradients=gradients,
#     ntd_AD=ntd_AD_noise,
#     ntd_A0=ntd_A0,
#     rho=rho,
# )

# inclusion_errors = []
# alpha_regs = []

# mus = np.linspace(1, 1 + 1e-3, 500)
# for mu in mus:
#     alpha_reg = -mu * np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD_noise)))
#     alpha_regs.append(alpha_reg)

#     inclusion_indexes = reconstruct_inclusions(eigenvalue_dict, alpha_reg=alpha_reg)

#     inclusion_error = inclusion_reconstruction_error(recon_solver, inclusion_indexes)

#     inclusion_errors.append(inclusion_error)


# optim_alpha_idx = np.argmin(inclusion_errors)
# optim_alpha = alpha_regs[optim_alpha_idx]

# print(f"Optimal mu: {mus[optim_alpha_idx]}")

# inclusion_indexes_optim = reconstruct_inclusions(eigenvalue_dict, alpha_reg=optim_alpha)

# plot_inclusion_reconstruction_error(alpha_regs, inclusion_errors, optim_alpha)

# plot_highlighted_eigenvalues(recon_mesh, inclusion_indexes_optim, eigenvalue_dict)

# plt.show()
