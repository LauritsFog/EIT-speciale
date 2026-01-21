# %%

from fem_solver import FemConductivitySolver, interpolate_solutions_to_new_mesh
import matplotlib.pyplot as plt
from plotting_functions import (
    plot_conductivity_components,
    plot_highlighted_elements,
    plot_inclusion_reconstruction_error,
    plot_highlighted_eigenvalues,
)
from inner_reconstruction_method import (
    compute_reconstruction_test_values,
    reconstruct_inclusions,
    add_noise,
    precompute_gradients,
    compute_classification_error,
    compute_hausdorff_distance,
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


seed = np.random.seed()

characteristic_length = 0.01
recon_h = 0.04

max_freq = 14

# Tolerance in the inclusion test: min(eigenvalues) + tol > 0
delta = 1e-4

shape_choice = "smiley"

A0_fun = lambda x, y: (1, 0, 1)

ellipse_conductivity = lambda x, y: (3, 2, 5 + 10 * y**2)
square_conductivity = lambda x, y: (5, 3, 4)
mouth_conductivity = lambda x, y: (4 + 5 * (x - 0.5) ** 2, 2, 5)


def AD_fun(x, y):
    return conductivity_AD(
        x,
        y,
        A0_fun,
        (ellipse_conductivity, square_conductivity, mouth_conductivity),
        shape_choice,
        None,
    )


solverA0 = FemConductivitySolver(mesh_characteristic_length=characteristic_length)
solverA0.set_conductivity(A0_fun, A0_fun)

recon_solver = FemConductivitySolver(mesh_characteristic_length=recon_h)
recon_solver.set_conductivity(AD_fun, A0_fun)
recon_mesh = recon_solver.mesh

mesh = solverA0.mesh

# Computing the "optimal" alpha, beta, c constants from checking the conductivity on each element
solverAD = FemConductivitySolver(mesh=mesh)
c, alpha, beta = solverAD.set_conductivity(AD_fun, A0_fun)

print(f"c = {c}, alpha = {alpha}, beta = {beta}")

solverA0.compute_ntd_map(max_freq=max_freq)
solverAD.compute_ntd_map(max_freq=max_freq)

recon_basis_solutions = interpolate_solutions_to_new_mesh(solverA0, recon_solver)

# AD-A0 >= cI and alpha I <= AD <= beta I

rho = c * (alpha**2) / (beta**2)

ntd_A0 = solverA0.ntd_matrix
ntd_AD = solverAD.ntd_matrix

ntd_AD_noise = add_noise(ntd_AD, noise_level=delta, seed=seed)

plot_conductivity_components(AD_fun, title="")

plt.savefig("Figures/Regularization_parameter/Conductivity_AD.pdf")

gradients = precompute_gradients(recon_basis_solutions, recon_mesh)

### Reconstruction without noisy data

eigenvalue_dict, _ = compute_reconstruction_test_values(
    mesh=recon_mesh,
    gradients=gradients,
    ntd_AD=ntd_AD,
    ntd_A0=ntd_A0,
    rho=rho,
)

target_inclusion = recon_solver.get_target_inclusion_indeces()

plot_highlighted_elements(recon_mesh, target_inclusion, color="black", alpha=0.7)

plt.savefig("Figures/Regularization_parameter/Target_inclusion.pdf")

recon_errors_reg = []
alpha_regs = []

mus = np.linspace(0.9999, 0.9999999, 300)

for mu in mus:
    alpha_reg = -mu * np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD)))
    alpha_regs.append(alpha_reg)

    recon_inclusion_reg = reconstruct_inclusions(eigenvalue_dict, alpha_reg=alpha_reg)

    recon_error_reg = compute_classification_error(
        recon_mesh, target_inclusion, recon_inclusion_reg
    )

    # recon_error_reg = compute_hausdorff_distance(
    #     recon_mesh, recon_inclusion_reg, target_inclusion
    # )

    recon_errors_reg.append(recon_error_reg)

optim_alpha_idx = np.argmin(recon_errors_reg)
optim_alpha = alpha_regs[optim_alpha_idx]
sub_optim_alpha = alpha_regs[optim_alpha_idx - 50]

print(f"Optimal mu without noise: {mus[optim_alpha_idx]}")
print(f"Sub-optimal alpha without noise: {sub_optim_alpha}")
print(f"Optimal alpha without noise: {optim_alpha}")

recon_inclusion_optim = reconstruct_inclusions(eigenvalue_dict, alpha_reg=optim_alpha)

if min(recon_errors_reg) != np.inf:
    plot_inclusion_reconstruction_error(
        alpha_regs, recon_errors_reg, optim_alpha, sub_optim_alpha
    )

    plt.savefig("Figures/Regularization_parameter/Recon_error_no_noise.pdf")

if len(recon_inclusion_optim) > 0:
    plot_highlighted_eigenvalues(recon_mesh, recon_inclusion_optim, eigenvalue_dict)

    plt.savefig("Figures/Regularization_parameter/Optimal_alpha_recon_no_noise.pdf")

recon_inclusion_reg = reconstruct_inclusions(eigenvalue_dict, alpha_reg=sub_optim_alpha)

if len(recon_inclusion_reg) > 0:
    plot_highlighted_eigenvalues(recon_mesh, recon_inclusion_reg, eigenvalue_dict)

    plt.savefig("Figures/Regularization_parameter/Sub_optimal_alpha_recon_no_noise.pdf")

recon_inclusion_reg = reconstruct_inclusions(eigenvalue_dict, alpha_reg=0)

if len(recon_inclusion_reg) > 0:
    plot_highlighted_eigenvalues(recon_mesh, recon_inclusion_reg, eigenvalue_dict)

    plt.savefig("Figures/Regularization_parameter/Alpha_zero_recon_no_noise.pdf")

### Reconstruction with noisy data

eigenvalue_dict_noise, _ = compute_reconstruction_test_values(
    mesh=recon_mesh,
    gradients=gradients,
    ntd_AD=ntd_AD_noise,
    ntd_A0=ntd_A0,
    rho=rho,
)

recon_errors_reg = []
alpha_regs = []

mus = np.linspace(1.00000000001, 1.00001, 300)

for mu in mus:
    alpha_reg = -mu * np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD_noise)))
    alpha_regs.append(alpha_reg)

    recon_inclusion_reg = reconstruct_inclusions(
        eigenvalue_dict_noise, alpha_reg=alpha_reg
    )

    recon_error_reg = compute_classification_error(
        recon_mesh, target_inclusion, recon_inclusion_reg
    )

    # recon_error_reg = compute_hausdorff_distance(
    #     recon_mesh, recon_inclusion_reg, target_inclusion
    # )

    recon_errors_reg.append(recon_error_reg)

optim_alpha_idx = np.argmin(recon_errors_reg)
optim_alpha = alpha_regs[optim_alpha_idx]
if optim_alpha_idx + 30 < len(alpha_regs):
    sub_optim_alpha = alpha_regs[optim_alpha_idx + 50]
else:
    sub_optim_alpha = alpha_regs[len(alpha_regs) - 1]

print(f"Optimal mu with noise: {mus[optim_alpha_idx]}")
print(f"Sub-optimal alpha with noise: {sub_optim_alpha}")
print(f"Optimal alpha with noise: {optim_alpha}")

recon_inclusion_optim = reconstruct_inclusions(
    eigenvalue_dict_noise, alpha_reg=optim_alpha
)

if min(recon_errors_reg) != np.inf:
    plot_inclusion_reconstruction_error(
        alpha_regs, recon_errors_reg, optim_alpha, sub_optim_alpha
    )

    plt.savefig("Figures/Regularization_parameter/Recon_error_noise.pdf")

if len(recon_inclusion_optim) > 0:
    plot_highlighted_eigenvalues(
        recon_mesh, recon_inclusion_optim, eigenvalue_dict_noise
    )

    plt.savefig("Figures/Regularization_parameter/Optimal_alpha_recon_noise.pdf")

recon_inclusion_reg = reconstruct_inclusions(
    eigenvalue_dict_noise, alpha_reg=sub_optim_alpha
)

if len(recon_inclusion_reg) > 0:
    plot_highlighted_eigenvalues(recon_mesh, recon_inclusion_reg, eigenvalue_dict_noise)

    plt.savefig("Figures/Regularization_parameter/Sub_optimal_alpha_recon_noise.pdf")

recon_inclusion_reg = reconstruct_inclusions(eigenvalue_dict_noise, alpha_reg=0)

if len(recon_inclusion_reg) > 0:
    plot_highlighted_eigenvalues(recon_mesh, recon_inclusion_reg, eigenvalue_dict_noise)

    plt.savefig("Figures/Regularization_parameter/Alpha_zero_recon_noise.pdf")

plt.show()
