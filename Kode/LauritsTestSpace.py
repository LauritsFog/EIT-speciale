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
    conductivity_A0,
    conductivity_AD,
)

# %%

# AD-A0 >= cI in Loewner order. c = min(EV(AD)) - max(EV(A0)) - c_tol.
# If c_tol = 0, then AD-A0 = cI in Loewner order.
# Bounds on AD (alpha-alpha_tol) I <= AD <= (beta + beta_tol) I.
# Frechet derivative DLambda(A0;c(alpha/beta)^2 I) with test
# LambdaAD - LambdaA0 - DLambda(A0;c(alpha/beta)^2 I) <= 0

shape_choice = "two_ellipses"

# shape_params = ((0.0, 0.0, 0.6, 0.4), (10, 0, 10))  # One ellipse, center, axes

shape_params = (
    (-0.25, -0.25, 0.3, 0.2, 0.25, 0.25, 0.3, 0.2),
    (10, 0, 10),
)  # Two ellipses, centers, axes

# shape_params = (
#     ((-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)),
#     (10, 0, 10), "Triangle"
# )  # Corner points

# shape_params = ((0.2, 0.6), (10, 0, 10))  # U shape, Thickness, radius

background_conductivity = (1, 0, 1)
inclusion_conductivity = shape_params[1]

characteristic_length = 0.05
piecewise_const = False

max_freq = 22

sampling_stride = 1

c_tol = 0
alpha_tol = 0
beta_tol = 0

[a11_A0, a12_A0, a22_A0] = background_conductivity
[a11_AD, a12_AD, a22_AD] = inclusion_conductivity

A0_conductivity_matrix = np.array([[a11_A0, a12_A0], [a12_A0, a22_A0]])
AD_conductivity_matrix = np.array([[a11_AD, a12_AD], [a12_AD, a22_AD]])

A0_eigenvalues = np.linalg.eigvals(A0_conductivity_matrix)
AD_eigenvalues = np.linalg.eigvals(AD_conductivity_matrix)

print("AD eigenvalues:", AD_eigenvalues)
print("A0 eigenvalues:", A0_eigenvalues)

# With diagonal conductivity matrices (AD-A0 >= cI). If c = 0, then AD-A0 = cI in Loewner order
c = min(AD_eigenvalues) - max(A0_eigenvalues) - c_tol
# Lower bound on AD (alpha I <= AD <= beta I)
alpha = min(AD_eigenvalues) - alpha_tol
# Upper bound on AD
beta = max(AD_eigenvalues) + beta_tol

const = c * (alpha**2) / (beta**2)


def my_conductivity_A0(x, y):
    a11, a12, a22, inclusion_flag = conductivity_A0(x, y, background_conductivity)
    return a11, a12, a22, inclusion_flag


def my_conductivity_AD(x, y):
    a11, a12, a22, inclusion_flag = conductivity_AD(
        x, y, background_conductivity, shape_choice
    )
    return a11, a12, a22, inclusion_flag


fig, axes = plot_conductivity_components(
    my_conductivity_AD,
    title="Two ellipse inclusions",
)

# %%

solverA0 = FemConductivitySolver(mesh_characteristic_length=characteristic_length)
solverA0.set_conductivity(my_conductivity_A0, my_conductivity_A0, c, piecewise_const)

mesh = solverA0.mesh

solverAD = FemConductivitySolver(mesh=mesh)
solverAD.set_conductivity(my_conductivity_AD, my_conductivity_A0, c, piecewise_const)

solverA0.compute_ntd_map(max_freq=max_freq)
solverAD.compute_ntd_map(max_freq=max_freq)

ntd_A0 = solverA0.ntd_matrix
ntd_AD = solverAD.ntd_matrix

fig_components, axes_components = solverAD.plot_conductivity_components(piecewise_const)

# Solve with different boundary fluxes by modifying b
x = ufl.SpatialCoordinate(solverA0.get_mesh())

nc = ufl.cos(3 * ufl.atan2(x[1], x[0]))  # Neumann condition (flux) on boundary

# Re-assemble only the RHS with new flux
solverAD.assemble_system(neumann_cond=nc)
uAD = solverAD.solve_system()

# Get plot objects WITHOUT displaying them
solverAD.plot_solution("Solution $u_f^A$")

# Plot entire boundary
solverAD.plot_boundary_solution_with_neumann_cond(
    "Solution $u_f^A$ on boundary with Neumann condition"
)

# Call the function directly
inclusion_indexes, test_matrix_eigenvalues = find_inclusion_elements_manual(
    mesh=mesh,
    solutions_A0=solverA0.basis_solutions,
    ntd_AD=ntd_AD,
    ntd_A0=ntd_A0,
    sampling_stride=sampling_stride,
    const=const,
)

plot_highlighted_elements_with_inclusions(mesh, inclusion_indexes, solverAD.cell_tags)

plot_test_matrix_test_matrix_eigenvalues(mesh, test_matrix_eigenvalues)

plt.show()
