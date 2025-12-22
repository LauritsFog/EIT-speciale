# %%

from fem_solver import FemConductivitySolver
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import ufl
from plotting_functions import plot_conductivity_components, plot_errors, plot_ND_map
from conductivity_functions import (
    conductivity_AD,
)
from inner_reconstruction_method import add_noise
from inner_reconstruction_method import (
    find_inclusion_elements,
)
# %%


def get_sine_eigenvalues(ND_matrix, max_freq):
    ND_matrix_sine = ND_matrix[max_freq:, max_freq:]

    ND_eigenvals_unsorted, ND_eigenvectors = np.linalg.eig(ND_matrix_sine)

    dominant_indices = np.argmax(np.abs(ND_eigenvectors), axis=0)

    canonical_sort_permutation = np.argsort(dominant_indices)

    ND_eigenvals = ND_eigenvals_unsorted[canonical_sort_permutation]

    return np.array(ND_eigenvals)


def polar_to_euclidean(polar_conductivities, x, y):
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

    a_r, a_theta = polar_conductivities

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

# Define geometry
r1, r2 = 0.3, 0.7

# Eigenmode frequency
n = 3

shape_choice = "two_ellipses"
shape_params = (0, 0, r1, r1, 0, 0, r2, r2)

# Each layer given in polar coordinates (a_r, a_theta)
# polar_conductivities = [
#     (1.0, 1.0),  # 0 <= r < r1, layer 3
#     (1.0, 1.0),  # r1 <= r < r2, layer 2
#     (1.0, 1.0),  # r2 <= r <= 1, layer 1
# ]

polar_conductivities = [
    (8.0, 8.0),  # 0 <= r < r1, layer 3
    (4.0, 4.0),  # r1 <= r < r2, layer 2
    (1.0, 2.0),  # r2 <= r <= 1, layer 1
]

# Anisotropy factors κ_j = sqrt(a_theta^(j) / a_r^(j))
ks = [
    np.sqrt(polar_conductivities[0][1] / polar_conductivities[0][0]),  # Layer 3
    np.sqrt(polar_conductivities[1][1] / polar_conductivities[1][0]),  # Layer 2
    np.sqrt(polar_conductivities[2][1] / polar_conductivities[2][0]),  # Layer 1
]

# Check inner most layer (layer 3) first.
my_conductivity_A0 = lambda x, y: polar_to_euclidean(polar_conductivities[2], x, y)
inclusion_conductivity_1 = lambda x, y: polar_to_euclidean(
    polar_conductivities[0], x, y
)
inclusion_conductivity_2 = lambda x, y: polar_to_euclidean(
    polar_conductivities[1], x, y
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


coeffs = compute_solution_coefficients(r1, r2, n, polar_conductivities)

# Visualization
fig, axes = plot_conductivity_components(
    my_conductivity_AD, title=f"{shape_choice} Conductivity Distribution"
)


def u_exact(x, y):
    return exact_solution(x, y, n, [r1, r2], coeffs, ks)


## Find error for different characteristic lengths and eigenfunctions (neumann conds)

noise_level = 0.0

max_freq = 5

modes = range(1, max_freq + 1)

characteristic_lengths = [0.08, 0.04]

errors = np.zeros((len(characteristic_lengths), 1))
eigenval_errors = np.zeros((max_freq, len(characteristic_lengths)))

for i, cl in enumerate(characteristic_lengths):
    print(f"Characeristic length: {cl}")

    solverAD = FemConductivitySolver(mesh_characteristic_length=cl)
    solverAD.set_conductivity(
        my_conductivity_AD,
        my_conductivity_A0,
    )

    x = ufl.SpatialCoordinate(solverAD.get_mesh())
    nc = ufl.cos(n * ufl.atan2(x[1], x[0]))

    solverAD.assemble_system(neumann_cond=nc)
    solverAD.solve_system()

    error = solverAD.compute_error(u_exact, norm_type="H1")
    errors[i] = error

    ntd_AD, n_freqs = solverAD.compute_ntd_map(max_freq=max_freq)

    if noise_level > 0:
        ntd_AD = add_noise(ntd_AD, noise_level=noise_level)

        ND_fem_eigenvals = get_sine_eigenvalues(ntd_AD, max_freq)
    else:
        ND_fem_eigenvals = np.diag(ntd_AD[max_freq:, max_freq:])

    ND_exact_eigenvals = []

    for n in modes:
        coeffs = compute_solution_coefficients(r1, r2, n, polar_conductivities)
        (alpha1, beta1), (alpha2, beta2), (alpha3, beta3) = coeffs

        ND_exact_eigenvals.append(alpha3 + beta3)

    eigenval_errors[:, i] = (ND_exact_eigenvals - ND_fem_eigenvals) / np.abs(
        ND_exact_eigenvals
    )

    solverAD.cleanup()

# Plot errors
plot_errors(
    errors.T, characteristic_lengths, [f"n = {n}"], "FEM solution error convergence"
)
# plt.savefig("FEM_convergence.pdf")

min_eigval_errors = np.min(np.abs(eigenval_errors), axis=0)

plot_errors(
    min_eigval_errors.T,
    characteristic_lengths,
    [f"n = {n}"],
    "Smallest eigenvalue error convergence",
)
# plt.savefig("Min_eigval_convergence.pdf")

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

for i in range(len(characteristic_lengths)):
    ax.semilogy(
        modes,
        np.abs(eigenval_errors[:, i]),
        "o-",
        linewidth=2,
        markersize=4,
        label=f"Mesh size (h) = {characteristic_lengths[i]}",
    )
ax.semilogy(
    modes,
    (characteristic_lengths[0] * np.array(modes)) ** 2,
    "k--",
    linewidth=2,
    markersize=4,
    label="O(n²)",
)
ax.set_title("Relative errors between exact and numerical ND map eigenvalues")
ax.set_xlabel("Eigenvalue Index $n$")
ax.set_ylabel("Relative error (Log Scale)")
ax.grid(True, which="both", ls="--")
ax.legend()
plt.yscale("log")
plt.savefig("ND_eigenvalue_relative_error.pdf")

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

for i in range(len(characteristic_lengths)):
    ax.plot(
        modes,
        eigenval_errors[:, i],
        "o-",
        linewidth=2,
        markersize=4,
        label=f"Mesh size (h) = {characteristic_lengths[i]}",
    )

ax.set_title("Signed relative errors between exact and numerical ND map eigenvalues")
ax.set_xlabel("Eigenvalue Index $n$")
ax.set_ylabel("Relative error (Log Scale)")
ax.grid(True, which="both", ls="--")
ax.legend()
plt.savefig("ND_eigenvalue_signed_relative_error.pdf")

plt.show()
