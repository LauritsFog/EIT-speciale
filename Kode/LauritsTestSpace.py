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

# AD-A0 >= cI in Loewner order. c = min(EV(AD)) - max(EV(A0)) - c_tol.
# If c_tol = 0, then AD-A0 = cI in Loewner order.
# Bounds on AD (alpha-alpha_tol) I <= AD <= (beta + beta_tol) I.
# Frechet derivative DLambda(A0;c(alpha/beta)^2 I) with test
# LambdaAD - LambdaA0 - DLambda(A0;c(alpha/beta)^2 I) <= 0

characteristic_length = 0.05
piecewise_const = True

max_freq = 15

sampling_stride = 1

# Tolerance in the inclusion test: min(eigenvalues) + tol > 0
tol = 1e-15

c_tol = 0
alpha_tol = 0
beta_tol = 0

# Example setup
shape_choice = "two_ellipses"
shape_params = (-0.4, -0.4, 0.3, 0.2, 0.4, 0.4, 0.3, 0.2)

# Define conductivity fields
my_conductivity_A0 = lambda x, y: (1, 0, 1)

inclusion_conductivity_1 = lambda x, y: (5, 0, 5)
inclusion_conductivity_2 = lambda x, y: (10, 0, 8)


def my_conductivity_AD(x, y):
    return conductivity_AD(
        x,
        y,
        my_conductivity_A0,
        (inclusion_conductivity_1, inclusion_conductivity_2),
        shape_choice,
        shape_params,
    )


# Visualization
fig, axes = plot_conductivity_components(
    my_conductivity_AD, title=f"{shape_choice} Conductivity Distribution"
)

# %%

solverA0 = FemConductivitySolver(mesh_characteristic_length=characteristic_length)
solverA0.set_conductivity(my_conductivity_A0, my_conductivity_A0, piecewise_const)

mesh = solverA0.mesh

solverAD = FemConductivitySolver(mesh=mesh)
c, alpha, beta = solverAD.set_conductivity(
    my_conductivity_AD, my_conductivity_A0, piecewise_const
)

solverA0.compute_ntd_map(max_freq=max_freq)
solverAD.compute_ntd_map(max_freq=max_freq)

# AD-A0 >= cI and alpha I <= AD <= beta I

c = c - c_tol
alpha = alpha - alpha_tol
beta = beta + beta_tol

const = c * (alpha**2) / (beta**2)

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
    tol=tol,
)

plot_highlighted_elements_with_inclusions(mesh, inclusion_indexes, solverAD.cell_tags)

plot_test_matrix_test_matrix_eigenvalues(mesh, test_matrix_eigenvalues)

plt.show()
