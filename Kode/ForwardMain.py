from fem_solver import FemConductivitySolver
import matplotlib.pyplot as plt
import numpy as np
import ufl

solver1 = FemConductivitySolver()
mesh = solver1.get_mesh()

solver2 = FemConductivitySolver(mesh=mesh)


def my_Omega_0(x):
    return np.sqrt(x[1] ** 2 + x[0] ** 2) <= 0.5


def my_Omega_1(x):
    return np.sqrt(x[1] ** 2 + x[0] ** 2) >= 0.5


def my_conductivity_Omega_0(x, y):
    k11 = 1.0
    k12 = 0
    k22 = 1.0
    return k11, k12, k22


def my_conductivity_Omega_1(x, y):
    k11 = 0.2
    k12 = 0
    k22 = 0.4
    return k11, k12, k22


# Set conductivity with custom functions
solver1.set_conductivity(
    Omega_0=my_Omega_0,
    Omega_1=my_Omega_1,
    conductivity_Omega_0=my_conductivity_Omega_0,
    conductivity_Omega_1=my_conductivity_Omega_1,
)


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

# Assemble system once (matrix A remains the same)
A, b, xh = solver1.assemble_system()

# Solve with different boundary fluxes by modifying b
x = ufl.SpatialCoordinate(solver1.get_mesh())

nc = ufl.cos(ufl.atan2(x[1], x[0]))  # Neumann condition (flux) on boundary

# Re-assemble only the RHS with new flux
solver1.assemble_system(boundary_flux=nc)
uh1 = solver1.solve_system()

# Assemble system once (matrix A remains the same)
A, b, xh = solver2.assemble_system()

# Solve with different boundary fluxes by modifying b
x = ufl.SpatialCoordinate(solver2.get_mesh())

nc = ufl.cos(ufl.atan2(x[1], x[0]))  # Neumann condition (flux) on boundary

# Re-assemble only the RHS with new flux
solver2.assemble_system(boundary_flux=nc)
uh2 = solver2.solve_system()

# Get plot objects WITHOUT displaying them
fig1, ax1 = solver1.plot_solution("Solution 1")
fig2, ax2 = solver2.plot_solution("Solution 2")

# Now show both figures simultaneously
plt.show()

solver1.cleanup()
solver2.cleanup()
