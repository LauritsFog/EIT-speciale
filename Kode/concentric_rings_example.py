# %%

from fem_solver import FemConductivitySolver
import matplotlib.pyplot as plt
import numpy as np
import ufl
from plotting_functions import plot_conductivity_components, plot_errors, plot_ND_map
from conductivity_functions import (
    conductivity_AD,
)

# %%


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

characteristic_length = 0.05

max_freq = 10

sampling_stride = 1

# Tolerance in the inclusion test: min(eigenvalues) + tol > 0
tol = 1e-15

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
# polar_conductivities = [
#     (16.0, 8.0),  # 0 <= r < r1, layer 3
#     (2.0, 4.0),  # r1 <= r < r2, layer 2
#     (1.0, 1.0),  # r2 <= r <= 1, layer 1
# ]

polar_conductivities = [
    (5, 0.3),  # 0 <= r < r1, layer 3
    (0.5, 0.2),  # r1 <= r < r2, layer 2
    (0.1, 0.1),  # r2 <= r <= 1, layer 1
]

# Anisotropy factors κ_j = sqrt(a_theta^(j) / a_r^(j))
ks = [
    np.sqrt(polar_conductivities[0][1] / polar_conductivities[0][0]),  # Layer 3
    np.sqrt(polar_conductivities[1][1] / polar_conductivities[1][0]),  # Layer 2
    np.sqrt(polar_conductivities[2][1] / polar_conductivities[2][0]),  # Layer 1
]

print("Anisotropy factors k_j: ")
print(ks)

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
print("Solution coefficients: ")
print(coeffs[0])
print(coeffs[1])
print(coeffs[2])

(alpha1, beta1), (alpha2, beta2), (alpha3, beta3) = coeffs

# Visualization
fig, axes = plot_conductivity_components(
    my_conductivity_AD, title=f"{shape_choice} Conductivity Distribution"
)

solverAD = FemConductivitySolver(mesh_characteristic_length=characteristic_length)
c, alpha, beta = solverAD.set_conductivity(my_conductivity_AD, my_conductivity_A0)

fig_components, axes_components = solverAD.plot_conductivity_components()

## Solve with example Neumann condition

x = ufl.SpatialCoordinate(solverAD.get_mesh())

nc = ufl.cos(n * ufl.atan2(x[1], x[0]))  # Neumann condition (flux) on boundary

# Re-assemble only the RHS with new flux
solverAD.assemble_system(neumann_cond=nc)
uAD = solverAD.solve_system()


def u_exact(x, y):
    return exact_solution(x, y, n, [r1, r2], coeffs, ks)


L2_error = solverAD.compute_error(u_exact, norm_type="L2")

print(f"L2 error of u_h: {L2_error}")

solverAD.plot_solution("FEM solution")
solverAD.plot_exact_solution(u_exact, "Exact solution")

# Plot entire boundary
solverAD.plot_boundary_solution_with_neumann_cond(
    "Solution $u_f^A$ on boundary with Neumann condition"
)

solverAD.cleanup()

## Investigate eiganvalues

ntd_AD, n_freqs = solverAD.compute_ntd_map(max_freq=max_freq)
ND_fem_eigenvals = np.linalg.eigvals(ntd_AD)

plot_ND_map(ntd_AD, n_freqs)

ND_exact_eigenvals = []

for n in n_freqs:
    coeffs = compute_solution_coefficients(r1, r2, n, polar_conductivities)
    (alpha1, beta1), (alpha2, beta2), (alpha3, beta3) = coeffs

    ND_exact_eigenvals.append(alpha3 + beta3)

print(np.sort(ND_fem_eigenvals))
print(np.sort(np.diag(ntd_AD)))  # The diag gives the eigenvalues in this example
print(np.sort(ND_exact_eigenvals))

eigenval_error = abs(np.diag(ntd_AD) - ND_exact_eigenvals)
eigenval_error = (np.diag(ntd_AD) - ND_exact_eigenvals) / ND_exact_eigenvals
print(eigenval_error)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.stem(eigenval_error, basefmt=" ")
ax.set_title("ND map eigenvalues errors")
ax.grid(True)

## Find error for different characteristic lengths and eigenfunctions (neumann conds)

characteristic_lengths = [0.2, 0.1, 0.05, 0.01]
frequencies = [16]
L2_errors = np.zeros((len(characteristic_lengths), len(frequencies)))
min_eigval_errors = []

for i, cl in enumerate(characteristic_lengths):
    print(f"Characeristic length: {cl}")
    for j, n in enumerate(frequencies):
        print(f"Frequency: {n}")

        solverAD = FemConductivitySolver(mesh_characteristic_length=cl)
        solverAD.set_conductivity(
            my_conductivity_AD,
            my_conductivity_A0,
        )

        x = ufl.SpatialCoordinate(solverAD.get_mesh())
        nc = ufl.cos(n * ufl.atan2(x[1], x[0]))

        solverAD.assemble_system(neumann_cond=nc)
        solverAD.solve_system()

        coeffs = compute_solution_coefficients(r1, r2, n, polar_conductivities)

        L2_error = solverAD.compute_error(u_exact, norm_type="L2")
        L2_errors[i, j] = L2_error

        print(L2_error)

    ntd_AD, n_freqs = solverAD.compute_ntd_map(max_freq=max_freq)
    ND_fem_eigenvals = np.linalg.eigvals(ntd_AD)

    ND_exact_eigenvals = []

    for n in n_freqs:
        coeffs = compute_solution_coefficients(r1, r2, n, polar_conductivities)
        (alpha1, beta1), (alpha2, beta2), (alpha3, beta3) = coeffs

        ND_exact_eigenvals.append(alpha3 + beta3)

    min_eigval_errors.append(abs(min(ND_exact_eigenvals) - min(ND_fem_eigenvals)))

    solverAD.cleanup()

# Plot L2 errors
plot_errors(
    L2_errors.T, characteristic_lengths, [f"n = {freq}" for freq in frequencies]
)
plot_errors(
    min_eigval_errors, characteristic_lengths, None, "Minimum eigenvalue errors"
)

plt.show()
