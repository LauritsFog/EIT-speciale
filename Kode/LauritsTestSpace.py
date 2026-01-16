# %%

from fem_solver import FemConductivitySolver, interpolate_solutions_to_new_mesh
import matplotlib.pyplot as plt
from plotting_functions import (
    plot_conductivity_components,
    plot_inclusion_reconstruction_error,
    plot_highlighted_eigenvalues,
)
from inner_reconstruction_method import (
    compute_reconstruction_test_values,
    reconstruct_inclusions,
    add_noise,
    precompute_gradients,
    inclusion_reconstruction_error,
)
from conductivity_functions import (
    conductivity_AD,
)
import numpy as np


# AD-A0 >= cI in Loewner order. c = min(EV(AD)) - max(EV(A0)) - c_tol.
# If c_tol = 0, then AD-A0 = cI in Loewner order.
# Bounds on AD (alpha-alpha_tol) I <= AD <= (beta + beta_tol) I.
# Frechet derivative DLambda(A0;c(alpha/beta)^2 I) with test
# LambdaAD - LambdaA0 - DLambda(A0;c(alpha/beta)^2 I) <= 0

characteristic_length = 0.02
recon_h = 0.05

max_freq = 8

# min_eig -> 1 / (1 + np.exp(-sigmoid_scaling * min_eig)). Set to None for no scaling
sigmoid_scaling = 1e7

# Tolerance in the inclusion test: min(eigenvalues) + tol > 0
delta = 1e-6

c_tol = 0
alpha_tol = 0
beta_tol = 0

shape_choice = "ellipse"
shape_params = ((0.2, 0.2), (0.3, 0.3))  # Center, axes

# Define conductivities
my_conductivity_A0 = lambda x, y: (1, 0, 1)

inclusion_conductivity_1 = lambda x, y: (5, 0, 2)


def my_conductivity_AD(x, y):
    return conductivity_AD(
        x,
        y,
        my_conductivity_A0,
        inclusion_conductivity_1,
        shape_choice,
        shape_params,
    )


# shape_choice = "two_ellipses"
# shape_params = (-0.4, -0.4, 0.3, 0.2, 0.4, 0.4, 0.3, 0.2)  # Center, axis, center, axis

# my_conductivity_A0 = lambda x, y: (1, 0, 1)

# inclusion_conductivity_1 = lambda x, y: (15, 0, 5)
# inclusion_conductivity_2 = lambda x, y: (5, 0, 15)


# def my_conductivity_AD(x, y):
#     return conductivity_AD(
#         x,
#         y,
#         my_conductivity_A0,
#         (inclusion_conductivity_1, inclusion_conductivity_2),
#         shape_choice,
#         shape_params,
#     )


# %%

solverA0 = FemConductivitySolver(mesh_characteristic_length=characteristic_length)
solverA0.set_conductivity(my_conductivity_A0, my_conductivity_A0)

recon_solver = FemConductivitySolver(mesh_characteristic_length=recon_h)
recon_solver.set_conductivity(my_conductivity_AD, my_conductivity_A0)
recon_mesh = recon_solver.mesh

mesh = solverA0.mesh

# Computing the "optimal" alpha, beta, c constants from checking the conductivity on each element
solverAD = FemConductivitySolver(mesh=mesh)
c, alpha, beta = solverAD.set_conductivity(my_conductivity_AD, my_conductivity_A0)

solverA0.compute_ntd_map(max_freq=max_freq)
solverAD.compute_ntd_map(max_freq=max_freq)

recon_basis_solutions = interpolate_solutions_to_new_mesh(solverA0, recon_solver)

# AD-A0 >= cI and alpha I <= AD <= beta I

c = c - c_tol
alpha = alpha - alpha_tol
beta = beta + beta_tol

rho = c * (alpha**2) / (beta**2)

ntd_A0 = solverA0.ntd_matrix
ntd_AD = solverAD.ntd_matrix

ntd_AD_noise = add_noise(ntd_AD, noise_level=delta)

plot_conductivity_components(my_conductivity_AD, title="")

# # Solve with different boundary fluxes by modifying b
# x = ufl.SpatialCoordinate(solverA0.get_mesh())

# nc = ufl.cos(3 * ufl.atan2(x[1], x[0]))  # Neumann condition (flux) on boundary

# # Re-assemble only the RHS with new flux
# solverAD.assemble_system(neumann_cond=nc)
# uAD = solverAD.solve_system()

# # Get plot objects WITHOUT displaying them
# solverAD.plot_solution("Solution $u_f^A$")

# # Plot entire boundary
# solverAD.plot_boundary_solution_with_neumann_cond(
#     "Solution $u_f^A$ on boundary with Neumann condition"
# )

gradients = precompute_gradients(recon_basis_solutions, recon_mesh)

# Call the function directly
eigenvalue_dict, frechet_quantity_dict = compute_reconstruction_test_values(
    mesh=recon_mesh,
    gradients=gradients,
    ntd_AD=ntd_AD_noise,
    ntd_A0=ntd_A0,
    rho=rho,
)

inclusion_errors = []
alpha_regs = []

mus = np.linspace(1, 1 + 1e-3, 500)
for mu in mus:
    alpha_reg = -mu * np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD_noise)))
    alpha_regs.append(alpha_reg)

    inclusion_indexes = reconstruct_inclusions(eigenvalue_dict, alpha_reg=alpha_reg)

    inclusion_error = inclusion_reconstruction_error(recon_solver, inclusion_indexes)

    inclusion_errors.append(inclusion_error)


optim_alpha_idx = np.argmin(inclusion_errors)
optim_alpha = alpha_regs[optim_alpha_idx]

print(f"Optimal mu: {mus[optim_alpha_idx]}")

inclusion_indexes_optim = reconstruct_inclusions(eigenvalue_dict, alpha_reg=optim_alpha)

plot_inclusion_reconstruction_error(alpha_regs, inclusion_errors, optim_alpha)

plot_highlighted_eigenvalues(recon_mesh, inclusion_indexes_optim, eigenvalue_dict)

plt.show()
