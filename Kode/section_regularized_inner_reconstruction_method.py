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
import time
from reconstruction_method_3 import *

# %%

# AD-A0 >= cI in Loewner order. c = min(EV(AD)) - max(EV(A0)) - c_tol.
# If c_tol = 0, then AD-A0 = cI in Loewner order.
# Bounds on AD (alpha-alpha_tol) I <= AD <= (beta + beta_tol) I.
# Frechet derivative DLambda(A0;c(alpha/beta)^2 I) with test
# LambdaAD - LambdaA0 - DLambda(A0;c(alpha/beta)^2 I) <= 0


seed = np.random.seed(42)

characteristic_length = 0.005
recon_h = 0.05

delta_vals = [0, 1e-4, 5e-4, 1e-3]
mu_2_vals = [1.05, 0.95]  # Positive vs negative alpha
# alpha_large_vals = [0, 1e-3, 1e-3, 1e-3]

max_freq = 10
max_freq_algo_2 = 10  # Larger truncation frequency for algo 2

mus_neg = np.linspace(1, 1.0000001, 1000)
mus_pos = np.linspace(0.9999999, 1, 1000)

algo_1_idx = [(max_freq_algo_2 - max_freq), (max_freq_algo_2 + max_freq)]

opt_mu_vals = []
opt_alpha_vals = []
alpha_large_vals = []
mu_2_choices = []

idx_diff = 300

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

solverA0.compute_ntd_map(max_freq=max_freq_algo_2)
solverAD.compute_ntd_map(max_freq=max_freq_algo_2)

recon_basis_solutions = interpolate_solutions_to_new_mesh(solverA0, recon_solver)

# AD-A0 >= cI and alpha I <= AD <= beta I

rho = c * (alpha**2) / (beta**2)

ntd_A0 = solverA0.ntd_matrix
ntd_AD = solverAD.ntd_matrix

plot_conductivity_components(AD_fun, title="")

plt.savefig("Figures/Regularization_parameter/Conductivity_AD.pdf")

gradients = precompute_gradients(recon_basis_solutions, recon_mesh)

# %% #######################################

### Reconstruction without noisy data

eigenvalue_dict, _ = compute_reconstruction_test_values(
    mesh=recon_mesh,
    gradients=gradients[algo_1_idx[0] : algo_1_idx[1]],
    ntd_AD=ntd_AD[algo_1_idx[0] : algo_1_idx[1], algo_1_idx[0] : algo_1_idx[1]],
    ntd_A0=ntd_A0[algo_1_idx[0] : algo_1_idx[1], algo_1_idx[0] : algo_1_idx[1]],
    rho=rho,
)

target_inclusion = recon_solver.get_target_inclusion_indeces()

# plot_highlighted_elements(recon_mesh, target_inclusion, color="black", alpha=0.7)

# plt.savefig("Figures/Regularization_parameter/Target_inclusion_more_noise.pdf")

recon_errors_reg = []
alpha_regs = []

if np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD))) < 0:
    mus = mus_neg
else:
    mus = mus_pos

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

recon_inclusion_optim = reconstruct_inclusions(eigenvalue_dict, alpha_reg=optim_alpha)

if len(recon_inclusion_optim) > 0:
    plot_highlighted_eigenvalues(recon_mesh, recon_inclusion_optim, eigenvalue_dict)

    plt.savefig("Figures/Regularization_parameter/Optimal_alpha_recon_no_noise.pdf")

rho0 = 0.1
rho_step = 0.5
# mu_2 = mu_2_vals[0]
if np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD))) < 0:
    mu_2 = mu_2_vals[0]
else:
    mu_2 = mu_2_vals[1]
alpha_large = -mu_2 * np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD)))
# alpha_large = alpha_large_vals[0]

opt_mu_vals.append(mus[optim_alpha_idx])
opt_alpha_vals.append(optim_alpha)
alpha_large_vals.append(alpha_large)
mu_2_choices.append(mu_2)

inclusion_counter = reconstruction2(
    recon_mesh, gradients, ntd_AD, ntd_A0, alpha_large, rho0, rho_step, max_it=1000
)

fig, ax = plot_method_2(recon_mesh, inclusion_counter)
plt.savefig(
    "Figures/Regularization_parameter/Method_2_recon_no_noise.pdf",
    bbox_inches="tight",
)


# %% #######################################

### Reconstruction with noisy data

delta = delta_vals[1]
ntd_AD_noise = add_noise(ntd_AD, noise_level=delta, seed=seed)

eigenvalue_dict_noise, _ = compute_reconstruction_test_values(
    mesh=recon_mesh,
    gradients=gradients[algo_1_idx[0] : algo_1_idx[1]],
    ntd_AD=ntd_AD_noise[algo_1_idx[0] : algo_1_idx[1], algo_1_idx[0] : algo_1_idx[1]],
    ntd_A0=ntd_A0[algo_1_idx[0] : algo_1_idx[1], algo_1_idx[0] : algo_1_idx[1]],
    rho=rho,
)

recon_errors_reg = []
alpha_regs = []

if np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD_noise))) < 0:
    mus = mus_neg
else:
    mus = mus_pos

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
# mu_2 = mu_2_vals[1]
if np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD_noise))) < 0:
    mu_2 = mu_2_vals[0]
else:
    mu_2 = mu_2_vals[1]
alpha_large = -mu_2 * np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD_noise)))
# alpha_large = alpha_large_vals[1]

opt_mu_vals.append(mus[optim_alpha_idx])
opt_alpha_vals.append(optim_alpha)
alpha_large_vals.append(alpha_large)
mu_2_choices.append(mu_2)

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
    "Figures/Regularization_parameter/Method_2_recon_noise.pdf",
    bbox_inches="tight",
)

# %% #######################################

delta = delta_vals[2]
ntd_AD_noise = add_noise(ntd_AD, noise_level=delta, seed=seed)

# %%
eigenvalue_dict_noise, _ = compute_reconstruction_test_values(
    mesh=recon_mesh,
    gradients=gradients[algo_1_idx[0] : algo_1_idx[1]],
    ntd_AD=ntd_AD_noise[algo_1_idx[0] : algo_1_idx[1], algo_1_idx[0] : algo_1_idx[1]],
    ntd_A0=ntd_A0[algo_1_idx[0] : algo_1_idx[1], algo_1_idx[0] : algo_1_idx[1]],
    rho=rho,
)

recon_errors_reg = []
alpha_regs = []

# mus = np.linspace(1, 1.000001, 1000)
if np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD_noise))) < 0:
    mus = mus_neg
else:
    mus = mus_pos

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
# mu_2 = mu_2_vals[2]
if np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD_noise))) < 0:
    mu_2 = mu_2_vals[0]
else:
    mu_2 = mu_2_vals[1]
alpha_large = -mu_2 * np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD_noise)))
# alpha_large = alpha_large_vals[2]

opt_mu_vals.append(mus[optim_alpha_idx])
opt_alpha_vals.append(optim_alpha)
alpha_large_vals.append(alpha_large)
mu_2_choices.append(mu_2)

tic = time.perf_counter()
inclusion_counter = reconstruction2(
    recon_mesh,
    gradients,
    ntd_AD_noise,
    ntd_A0,
    alpha_large,
    rho0,
    rho_step,
    max_it=1000,
    print_output=False
)
t2 = time.perf_counter()-tic
print(f"Method 2 took {t2:.2f} seconds")

fig, ax = plot_method_2(recon_mesh, inclusion_counter)
plt.savefig(
    "Figures/Regularization_parameter/Method_2_recon_more_noise.pdf",
    bbox_inches="tight",
)
# %% ##### method 3 ######
tic = time.perf_counter()
rhos = reconstruction3(recon_mesh, gradients, ntd_AD_noise, ntd_A0, alpha_large)
t3 = time.perf_counter()-tic
print(f"Method 3 took {t3:.2f} seconds")

fig, ax = plot_method_3(recon_mesh, rhos)
plt.savefig(
    "Figures/Regularization_parameter/Method_3_recon_more_noise.pdf",
    bbox_inches="tight",
)
# %% #######################################

delta = delta_vals[3]
ntd_AD_noise = add_noise(ntd_AD, noise_level=delta, seed=seed)

eigenvalue_dict_noise, _ = compute_reconstruction_test_values(
    mesh=recon_mesh,
    gradients=gradients[algo_1_idx[0] : algo_1_idx[1]],
    ntd_AD=ntd_AD_noise[algo_1_idx[0] : algo_1_idx[1], algo_1_idx[0] : algo_1_idx[1]],
    ntd_A0=ntd_A0[algo_1_idx[0] : algo_1_idx[1], algo_1_idx[0] : algo_1_idx[1]],
    rho=rho,
)

recon_errors_reg = []
alpha_regs = []

# mus = np.linspace(1.00000001, 1.00009, 1000)
if np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD_noise))) < 0:
    mus = mus_neg
else:
    mus = mus_pos

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
# mu_2 = mu_2_vals[3]
if np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD_noise))) < 0:
    mu_2 = mu_2_vals[0]
else:
    mu_2 = mu_2_vals[1]
alpha_large = -mu_2 * np.min(np.real(np.linalg.eigvals(ntd_A0 - ntd_AD_noise)))

opt_mu_vals.append(mus[optim_alpha_idx])
opt_alpha_vals.append(optim_alpha)
alpha_large_vals.append(alpha_large)
mu_2_choices.append(mu_2)

# alpha_large = alpha_large_vals[3]
inclusion_counter = reconstruction2(
    recon_mesh,
    gradients,
    ntd_AD_noise,
    ntd_A0,
    alpha_large,
    rho0,
    rho_step,
    max_it=1000
)

fig, ax = plot_method_2(recon_mesh, inclusion_counter)
plt.savefig(
    "Figures/Regularization_parameter/Method_2_recon_much_more_noise.pdf",
    bbox_inches="tight",
)

print(f"Opt mus {opt_mu_vals}")
print(f"Opt alphas {opt_alpha_vals}")
print(f"mu 2 {mu_2_choices}")
print(f"alpha large {alpha_large_vals}")

# %%
