from fem_solver import FemConductivitySolver
import numpy as np
import ufl

solver = FemConductivitySolver()


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
solver.set_conductivity(
    Omega_0=my_Omega_0,
    Omega_1=my_Omega_1,
    conductivity_Omega_0=my_conductivity_Omega_0,
    conductivity_Omega_1=my_conductivity_Omega_1,
)

# Assemble system once (matrix A remains the same)
A, b, xh = solver.assemble_system()

# Solve with different boundary fluxes by modifying b
x = ufl.SpatialCoordinate(solver.get_mesh())

nc = ufl.cos(ufl.atan2(x[1], x[0]))  # Neumann condition (flux) on boundary

# Re-assemble only the RHS with new flux
solver.assemble_system(boundary_flux=nc)
uh = solver.solve_system()

solver.plot_solution()

# error = solver.compute_error()
# print(f"  L2 error: {error:.2e}")

solver.cleanup()
