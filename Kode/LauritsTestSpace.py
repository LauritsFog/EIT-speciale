from fem_solver import FemConductivitySolver
import matplotlib.pyplot as plt
import numpy as np
import ufl
from plotting_functions import (
    plot_conductivity_components,
    plot_highlighted_elements,
    plot_test_matrix_test_matrix_eigenvalues,
)
from inner_reconstruction_method import (
    find_inclusion_elements,
    find_inclusion_elements_manual,
)

k11_A0 = 1
k12_A0 = 0
k22_A0 = 1

k11_AD = 10
k12_AD = 0
k22_AD = 10


def my_conductivity_A0(x, y):
    k11 = k11_A0
    k12 = k12_A0
    k22 = k22_A0
    return k11, k12, k22


def my_conductivity_AD(x, y):
    k11, k12, k22 = my_conductivity_A0(x, y)
    if np.sqrt(x**2 + y**2) <= 0.5:
        k11 = k11_AD
        k22 = k22_AD
        k12 = k12_AD
    return k11, k12, k22


solverA0 = FemConductivitySolver()
solverA0.set_conductivity(my_conductivity_A0)

mesh = solverA0.mesh

solverAD = FemConductivitySolver(mesh=mesh)
solverAD.set_conductivity(my_conductivity_AD)

max_freq = 10
solverA0.compute_ntd_map(max_freq=max_freq)
solverAD.compute_ntd_map(max_freq=max_freq)

ntd_A0 = solverA0.ntd_matrix
ntd_AD = solverAD.ntd_matrix

fig_components, axes_components = solverAD.plot_conductivity_components()

# Solve with different boundary fluxes by modifying b
x = ufl.SpatialCoordinate(solverA0.get_mesh())

nc = ufl.cos(3 * ufl.atan2(x[1], x[0]))  # Neumann condition (flux) on boundary

# Re-assemble only the RHS with new flux
solverAD.assemble_system(neumann_cond=nc)
uAD = solverAD.solve_system()

# Get plot objects WITHOUT displaying them
solverAD.plot_solution("Solution $u_f^{\Lambda_A}$")

# Plot entire boundary
solverAD.plot_boundary_solution_with_neumann_cond(
    "Solution $u_f^{\Lambda_A}$ on boundary with Neumann condition"
)

c_tol = 0
alpha_tol = 0
beta_tol = 0

# With diagonal conductivity matrices (AD-A0 >= cI). If c = 0, then AD-A0 = cI i Lo
c = min(k11_AD, k22_AD) - max(k11_A0, k22_A0) - c_tol
# Lower bound on AD (alpha I <= AD <= beta I)
alpha = min(k11_AD, k22_AD) - alpha_tol
# Upper bound on AD
beta = max(k11_AD, k22_AD) + beta_tol

const = c * (alpha**2) / (beta**2)

# Call the function directly
inclusion_indexes, test_matrix_eigenvalues = find_inclusion_elements_manual(
    mesh=mesh,
    solutions_A0=solverA0.basis_solutions,
    ntd_AD=ntd_AD,
    ntd_A0=ntd_A0,
    sampling_stride=1,
    const=const,
)

plot_highlighted_elements(mesh, inclusion_indexes)

plot_test_matrix_test_matrix_eigenvalues(mesh, test_matrix_eigenvalues)

plt.show()
