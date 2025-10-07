from fem_solver import FemConductivitySolver
import matplotlib.pyplot as plt
import numpy as np
import ufl
from plotting_functions import plot_conductivity_components

solver1 = FemConductivitySolver()
mesh = solver1.get_mesh()

solver2 = FemConductivitySolver(mesh=mesh)


def my_Omega_0(x):
    return np.sqrt(x[1] ** 2 + x[0] ** 2) <= 0.5


def my_Omega_1(x):
    return np.sqrt(x[1] ** 2 + x[0] ** 2) >= 0.5


def my_conductivity_Omega_0(x, y):
    k11 = 4
    k12 = 0
    k22 = 4
    return k11, k12, k22


def my_conductivity_Omega_1(x, y):
    k11 = 0.1
    k12 = 0
    k22 = 0.1
    return k11, k12, k22


# Set conductivity with custom functions
solver1.set_conductivity(
    Omega_0=my_Omega_0,
    Omega_1=my_Omega_1,
    conductivity_Omega_0=my_conductivity_Omega_0,
    conductivity_Omega_1=my_conductivity_Omega_1,
)

# Plot conductivity components
# fig_components, axes_components = plot_conductivity_components(
#     my_Omega_0, my_Omega_1, my_conductivity_Omega_0, my_conductivity_Omega_1
# )

fig_components, axes_components = solver1.plot_conductivity_components()

# Solve with different boundary fluxes by modifying b
x = ufl.SpatialCoordinate(solver1.get_mesh())

nc = ufl.cos(3 * ufl.atan2(x[1], x[0]))  # Neumann condition (flux) on boundary

# Re-assemble only the RHS with new flux
solver1.assemble_system(neumann_cond=nc)
uh1 = solver1.solve_system()

solver1.plot_solution("Solution 1")

solver1.plot_boundary_solution_with_neumann_cond("Boundary solution 1")

solver1.plot_fourier_coefficients(max_freq=10)


def my_conductivity_Omega_0(x, y):
    k11 = 1
    k12 = 0
    k22 = 1
    return k11, k12, k22


def my_conductivity_Omega_1(x, y):
    k11 = 1
    k12 = 0
    k22 = 1
    return k11, k12, k22


# Set conductivity with custom functions
solver2.set_conductivity(
    Omega_0=my_Omega_0,
    Omega_1=my_Omega_1,
    conductivity_Omega_0=my_conductivity_Omega_0,
    conductivity_Omega_1=my_conductivity_Omega_1,
)

# Re-assemble only the RHS with new flux
solver2.assemble_system(neumann_cond=nc)
uh2 = solver2.solve_system()

# Get plot objects WITHOUT displaying them
solver2.plot_solution("Solution 2")

# Plot entire boundary
solver2.plot_boundary_solution_with_neumann_cond("Boundary solution 2")

solver2.plot_fourier_coefficients(max_freq=10)

ntd, n_values = solver2.compute_ntd_map(max_freq=5)

print(np.diag(ntd))

# Check if symmetric (should be for self-adjoint operators)
symmetry_error = np.linalg.norm(ntd - ntd.conj().T) / np.linalg.norm(ntd)
print(f"Symmetry error: {symmetry_error:.2e}")

plt.show()

solver1.cleanup()
solver2.cleanup()
