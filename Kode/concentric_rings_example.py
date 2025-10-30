# %%

from fem_solver import FemConductivitySolver
import matplotlib.pyplot as plt
import numpy as np
import ufl
from plotting_functions import (
    plot_conductivity_components,
    plot_highlighted_elements,
    plot_highlighted_elements_with_inclusions,
    plot_test_matrix_test_matrix_eigenvalues,
)
from inner_reconstruction_method import (
    find_inclusion_elements,
    find_inclusion_elements_manual,
)
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
        Anisotropy exponents κ_j for each layer.

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

    # Build layer-by-layer expression
    for j in range(len(ks)):
        if j == 0:
            cond = ufl.lt(r, radii[0])
        elif j < len(ks) - 1:
            cond = ufl.And(ufl.ge(r, radii[j - 1]), ufl.lt(r, radii[j]))
        else:
            # Outermost layer: r >= r_{N-1}
            cond = ufl.ge(r, radii[-1])

        alpha_j, beta_j = coeffs[j]
        k_j = ks[j]

        v_j = alpha_j * r ** (k_j * m) + beta_j * r ** (-k_j * m)
        v_expr += ufl.conditional(cond, v_j, 0)

    # Real-valued field (cosine mode)
    u_expr = v_expr * ufl.cos(n * theta)
    return u_expr


## Configuration

characteristic_length = 0.05
piecewise_const = True

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
n = 2

shape_choice = "two_ellipses"
shape_params = (0, 0, r1, r1, 0, 0, r2, r2)

# Each layer given in polar coordinates (a_r, a_theta)
polar_layers = [
    (1.0, 1.0),
    (8.0, 8.0),
    (4.0, 4.0),
]

# Anisotropy factors κ_j = sqrt(a_theta^(j) / a_r^(j))
ks = [
    np.sqrt(polar_layers[0][1] / polar_layers[0][0]),
    np.sqrt(polar_layers[1][1] / polar_layers[1][0]),
    np.sqrt(polar_layers[2][1] / polar_layers[2][0]),
]

# Each returns (sigma_r, sigma_t)
my_conductivity_A0 = lambda x, y: polar_to_euclidean(polar_layers[0], x, y)
inclusion_conductivity_1 = lambda x, y: polar_to_euclidean(polar_layers[1], x, y)
inclusion_conductivity_2 = lambda x, y: polar_to_euclidean(polar_layers[2], x, y)


def my_conductivity_AD(x, y):
    return conductivity_AD(
        x,
        y,
        my_conductivity_A0,
        (inclusion_conductivity_1, inclusion_conductivity_2),
        shape_choice,
        shape_params,
    )


coeffs = compute_solution_coefficients(r1, r2, n, polar_layers)
print("Solution coefficients: ")
print(coeffs[0])
print(coeffs[1])
print(coeffs[2])

# Visualization
fig, axes = plot_conductivity_components(
    my_conductivity_AD, title=f"{shape_choice} Conductivity Distribution"
)

solverA0 = FemConductivitySolver(mesh_characteristic_length=characteristic_length)
solverA0.set_conductivity(my_conductivity_A0, my_conductivity_A0, piecewise_const)

mesh = solverA0.mesh

solverAD = FemConductivitySolver(mesh=mesh)
c, alpha, beta = solverAD.set_conductivity(
    my_conductivity_AD, my_conductivity_A0, piecewise_const
)

solverAD.compute_ntd_map(max_freq=max_freq)

# AD-A0 >= cI and alpha I <= AD <= beta I

c = c - c_tol
alpha = alpha - alpha_tol
beta = beta + beta_tol

const = c * (alpha**2) / (beta**2)

ntd_AD = solverAD.ntd_matrix

fig_components, axes_components = solverAD.plot_conductivity_components(piecewise_const)

## Solve with example Neumann condition

x = ufl.SpatialCoordinate(solverAD.get_mesh())

nc = ufl.cos(n * ufl.atan2(x[1], x[0]))  # Neumann condition (flux) on boundary

# Re-assemble only the RHS with new flux
solverAD.assemble_system(neumann_cond=nc)
uAD = solverAD.solve_system()


def u_exact(x, y):
    return exact_solution(x, y, n, [r1, r2], coeffs, ks)


L2_error = solverAD.compute_L2_error(u_exact)

print(f"L2 error of u_h: {L2_error}")

solverAD.plot_solution("Solution $u_f^A$")

# Plot entire boundary
solverAD.plot_boundary_solution_with_neumann_cond(
    "Solution $u_f^A$ on boundary with Neumann condition"
)

solverAD.cleanup()

## Find error for different characteristic lengths

characteristic_lengths = [0.1, 0.05, 0.001, 0.0005]
L2_errors = []

for cl in characteristic_lengths:
    solverAD = FemConductivitySolver(mesh_characteristic_length=cl)
    c, alpha, beta = solverAD.set_conductivity(
        my_conductivity_AD, my_conductivity_A0, piecewise_const
    )

    x = ufl.SpatialCoordinate(solverAD.get_mesh())
    nc = ufl.cos(n * ufl.atan2(x[1], x[0]))

    solverAD.assemble_system(neumann_cond=nc)
    uAD = solverAD.solve_system()

    L2_error = solverAD.compute_L2_error(u_exact)
    L2_errors.append(L2_error)

    solverAD.cleanup()


print("L2 errors for different characteristic lengths:")
print(L2_errors)

plt.show()
