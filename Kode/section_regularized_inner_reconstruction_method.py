# %%

from fem_solver import FemConductivitySolver, interpolate_solutions_to_new_mesh
import matplotlib.pyplot as plt
from plotting_functions import (
    plot_conductivity_components,
    plot_highlighted_elements,
    plot_inclusion_reconstruction_error,
    plot_highlighted_eigenvalues,
    plot_isotropic_conductivity,
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
from reconstruction_method_2 import *

# %%

# AD-A0 >= cI in Loewner order. c = min(EV(AD)) - max(EV(A0)) - c_tol.
# If c_tol = 0, then AD-A0 = cI in Loewner order.
# Bounds on AD (alpha-alpha_tol) I <= AD <= (beta + beta_tol) I.
# Frechet derivative DLambda(A0;c(alpha/beta)^2 I) with test
# LambdaAD - LambdaA0 - DLambda(A0;c(alpha/beta)^2 I) <= 0


seed = np.random.seed()

characteristic_length = 0.005
recon_h = 0.05

max_freq = 10

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

# plot_highlighted_elements(recon_mesh, target_inclusion, color="black", alpha=0.7)

# plt.savefig("Figures/Regularization_parameter/Target_inclusion_more_noise.pdf")

# %% #######################################

recon_errors_reg = []
alpha_regs = []

mus = np.linspace(0.9999, 0.9999999, 1000)

for mu in mus:
    alpha_reg = -mu * np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD)))
    alpha_regs.append(alpha_reg)

    recon_inclusion_reg = reconstruct_inclusions(eigenvalue_dict, alpha_reg=alpha_reg)

    recon_error_reg = compute_classification_error(
        recon_mesh, target_inclusion, recon_inclusion_reg
    )

    recon_errors_reg.append(recon_error_reg)

optim_alpha_idx = np.argmin(recon_errors_reg)
optim_alpha = alpha_regs[optim_alpha_idx]

idx_diff = 300
if optim_alpha_idx - idx_diff >= 0:
    sub_optim_alpha = alpha_regs[optim_alpha_idx - idx_diff]
else:
    sub_optim_alpha = alpha_regs[0]

print(f"Optimal mu without noise: {mus[optim_alpha_idx]}")
print(f"Optimal alpha without noise: {optim_alpha}")

recon_inclusion_optim = reconstruct_inclusions(eigenvalue_dict, alpha_reg=optim_alpha)

if len(recon_inclusion_optim) > 0:
    plot_highlighted_eigenvalues(recon_mesh, recon_inclusion_optim, eigenvalue_dict)

    plt.savefig("Figures/Regularization_parameter/Optimal_alpha_recon_no_noise.pdf")

rho0 = 0.1
rho_step = 0.5
mu_2 = 1.1
alpha_large = np.abs(mu_2 * np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD))))
# alpha_large = 1e-5
inclusion_counter = reconstruction2(
    recon_mesh, gradients, ntd_AD, ntd_A0, alpha_large, rho0, rho_step, max_it=1000
)

fig, ax = plot_method_2(recon_mesh, inclusion_counter)
plt.savefig(
    "Figures/Regularization_parameter/Method_2_isotropic_recon_no_noise.pdf",
    bbox_inches="tight",
)


# %% #######################################

### Reconstruction with noisy data

delta = 1e-4
ntd_AD_noise = add_noise(ntd_AD, noise_level=delta, seed=seed)

eigenvalue_dict_noise, _ = compute_reconstruction_test_values(
    mesh=recon_mesh,
    gradients=gradients,
    ntd_AD=ntd_AD_noise,
    ntd_A0=ntd_A0,
    rho=rho,
)

recon_errors_reg = []
alpha_regs = []

mus = np.linspace(0.999, 1.0001, 1000)

for mu in mus:
    alpha_reg = -mu * np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD_noise)))
    alpha_regs.append(alpha_reg)

    recon_inclusion_reg = reconstruct_inclusions(
        eigenvalue_dict_noise, alpha_reg=alpha_reg
    )

    recon_error_reg = compute_classification_error(
        recon_mesh, target_inclusion, recon_inclusion_reg
    )

    recon_errors_reg.append(recon_error_reg)

optim_alpha_idx = np.argmin(recon_errors_reg)
optim_alpha = alpha_regs[optim_alpha_idx]
if alpha_regs[optim_alpha_idx] > 0:
    if optim_alpha_idx + idx_diff < len(alpha_regs):
        sub_optim_alpha = alpha_regs[optim_alpha_idx + idx_diff]
    else:
        sub_optim_alpha = alpha_regs[len(alpha_regs) - 1]
else:
    if optim_alpha_idx - idx_diff >= 0:
        sub_optim_alpha = alpha_regs[optim_alpha_idx - idx_diff]
    else:
        sub_optim_alpha = alpha_regs[0]

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
else:
    print("No inclusion reconstructed for optimal alpha with noise")

recon_inclusion_reg = reconstruct_inclusions(
    eigenvalue_dict_noise, alpha_reg=sub_optim_alpha
)

if len(recon_inclusion_reg) > 0:
    plot_highlighted_eigenvalues(recon_mesh, recon_inclusion_reg, eigenvalue_dict_noise)

    plt.savefig("Figures/Regularization_parameter/Sub_optimal_alpha_recon_noise.pdf")

rho0 = 0.1
rho_step = 0.5
mu_2 = 1.1
alpha_large = np.abs(mu_2 * np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD))))
# alpha_large = 1e-5
inclusion_counter = reconstruction2(
    recon_mesh,
    gradients,
    ntd_AD_noise,
    ntd_A0,
    alpha_large,
    rho0,
    rho_step,
    max_it=1000,
)

fig, ax = plot_method_2(recon_mesh, inclusion_counter)
plt.savefig(
    "Figures/Regularization_parameter/Method_2_isotropic_recon_noise.pdf",
    bbox_inches="tight",
)

# %% #######################################

delta = 5e-4
ntd_AD_noise = add_noise(ntd_AD, noise_level=delta, seed=seed)

eigenvalue_dict_noise, _ = compute_reconstruction_test_values(
    mesh=recon_mesh,
    gradients=gradients,
    ntd_AD=ntd_AD_noise,
    ntd_A0=ntd_A0,
    rho=rho,
)

recon_errors_reg = []
alpha_regs = []

mus = np.linspace(1, 1.00005, 1000)

for mu in mus:
    alpha_reg = -mu * np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD_noise)))
    alpha_regs.append(alpha_reg)

    recon_inclusion_reg = reconstruct_inclusions(
        eigenvalue_dict_noise, alpha_reg=alpha_reg
    )

    recon_error_reg = compute_classification_error(
        recon_mesh, target_inclusion, recon_inclusion_reg
    )

    recon_errors_reg.append(recon_error_reg)

optim_alpha_idx = np.argmin(recon_errors_reg)
optim_alpha = alpha_regs[optim_alpha_idx]

print(f"Optimal mu with noise: {mus[optim_alpha_idx]}")
print(f"Optimal alpha with noise: {optim_alpha}")

recon_inclusion_optim = reconstruct_inclusions(
    eigenvalue_dict_noise, alpha_reg=optim_alpha
)

if len(recon_inclusion_optim) > 0:
    plot_highlighted_eigenvalues(
        recon_mesh, recon_inclusion_optim, eigenvalue_dict_noise
    )

    plt.savefig(
        "Figures/Regularization_parameter/Optimal_alpha_recon_noise_more_noise.pdf"
    )
else:
    print("No inclusion reconstructed for optimal alpha with noise")

rho0 = 0.1
rho_step = 0.5
mu_2 = 1.1
alpha_large = np.abs(mu_2 * np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD))))
# alpha_large = 1e-5
inclusion_counter = reconstruction2(
    recon_mesh,
    gradients,
    ntd_AD_noise,
    ntd_A0,
    alpha_large,
    rho0,
    rho_step,
    max_it=1000,
)

fig, ax = plot_method_2(recon_mesh, inclusion_counter)
plt.savefig(
    "Figures/Regularization_parameter/Method_2_isotropic_recon_more_noise.pdf",
    bbox_inches="tight",
)

# %% #######################################

delta = 1e-3
ntd_AD_noise = add_noise(ntd_AD, noise_level=delta, seed=seed)

eigenvalue_dict_noise, _ = compute_reconstruction_test_values(
    mesh=recon_mesh,
    gradients=gradients,
    ntd_AD=ntd_AD_noise,
    ntd_A0=ntd_A0,
    rho=rho,
)

recon_errors_reg = []
alpha_regs = []

mus = np.linspace(1.00000001, 1.00009, 1000)
# mus = np.linspace(0.9995, 1.0000, 1000)

for mu in mus:
    alpha_reg = -mu * np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD_noise)))
    alpha_regs.append(alpha_reg)

    recon_inclusion_reg = reconstruct_inclusions(
        eigenvalue_dict_noise, alpha_reg=alpha_reg
    )

    recon_error_reg = compute_classification_error(
        recon_mesh, target_inclusion, recon_inclusion_reg
    )

    recon_errors_reg.append(recon_error_reg)

optim_alpha_idx = np.argmin(recon_errors_reg)
optim_alpha = alpha_regs[optim_alpha_idx]

print(f"Optimal mu with noise: {mus[optim_alpha_idx]}")
print(f"Optimal alpha with noise: {optim_alpha}")

recon_inclusion_optim = reconstruct_inclusions(
    eigenvalue_dict_noise, alpha_reg=optim_alpha
)

if len(recon_inclusion_optim) > 0:
    plot_highlighted_eigenvalues(
        recon_mesh, recon_inclusion_optim, eigenvalue_dict_noise
    )

    plt.savefig(
        "Figures/Regularization_parameter/Optimal_alpha_recon_noise_much_more_noise.pdf"
    )
else:
    print("No inclusion reconstructed for optimal alpha with noise")

rho0 = 0.1
rho_step = 0.5
mu_2 = 1.1
alpha_large = np.abs(mu_2 * np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD))))
# alpha_large = 1e-5
inclusion_counter = reconstruction2(
    recon_mesh,
    gradients,
    ntd_AD_noise,
    ntd_A0,
    alpha_large,
    rho0,
    rho_step,
    max_it=1000,
)

fig, ax = plot_method_2(recon_mesh, inclusion_counter)
plt.savefig(
    "Figures/Regularization_parameter/Method_2_isotropic_recon_much_more_noise.pdf",
    bbox_inches="tight",
)
