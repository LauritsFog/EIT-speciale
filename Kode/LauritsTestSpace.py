# %%

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
from conductivity_functions import (
    conductivity_A0,
    conductivity_AD,
)

# %%

background_conductivity = (1, 0, 1)
inclusion_conductivity = (10, 0, 10)

shape_choice = "U"
shape_params = ((0.2, 0.6), inclusion_conductivity)

# shape_choise = "triangle"
# shape_params = (((0.0, -0.6), (0.5, 0.3), (-0.5, 0.3)), inclusion_conductivity)

# shape_choice = "ellipse"
# shape_params = ((0.0, 0.0, 0.6, 0.4), inclusion_conductivity)

characteristic_length = 0.05

max_freq = 40


def my_conductivity_A0(x, y):
    a11, a12, a22 = conductivity_A0(x, y, background_conductivity)
    return a11, a12, a22


def my_conductivity_AD(x, y):
    a11, a12, a22 = conductivity_AD(
        x, y, background_conductivity, shape_choice, shape_params
    )
    return a11, a12, a22


fig, axes = plot_conductivity_components(
    my_conductivity_AD,
    title="U-shaped inclusion",
)

# %%

solverA0 = FemConductivitySolver(mesh_characteristic_length=characteristic_length)
solverA0.set_conductivity(my_conductivity_A0)

mesh = solverA0.mesh

solverAD = FemConductivitySolver(mesh=mesh)
solverAD.set_conductivity(my_conductivity_AD)

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
solverAD.plot_solution("Solution $u_f^A$")

# Plot entire boundary
solverAD.plot_boundary_solution_with_neumann_cond(
    "Solution $u_f^A$ on boundary with Neumann condition"
)

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
