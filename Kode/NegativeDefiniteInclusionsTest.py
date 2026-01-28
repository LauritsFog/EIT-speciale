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

seed = 49

characteristic_length = 0.005
recon_h = 0.05

mu_2_pos = 1.01
mu_2_neg = 0.99

max_freq = 10
max_freq_algo_2 = 10  # Larger truncation frequency for algo 2

mus_pos = np.linspace(1, 1.01, 3000)
mus_neg = np.linspace(0.99, 1, 3000)

algo_1_idx = [(max_freq_algo_2 - max_freq), (max_freq_algo_2 + max_freq)]

opt_mu_vals = []
opt_alpha_vals = []
alpha_large_vals = []
mu_2_choices = []
relative_noise_vals = []

shape_choice = "smiley"

A0_fun = lambda x, y: (2, 0.5, 2)

s = 0.1
ellipse_conductivity = lambda x, y: (3 * s, 2 * s, (5 + 10 * y**2) * s)
square_conductivity = lambda x, y: (5 * s, 3 * s, 4 * s)
mouth_conductivity = lambda x, y: ((4 + 5 * (x - 0.5) ** 2) * s, 2 * s, 5 * s)

# ellipse_conductivity = lambda x, y: (1 / 3, 1 / 2, 1 / (5 + 10 * y**2))
# square_conductivity = lambda x, y: (1 / 5, 1 / 3, 1 / 4)
# mouth_conductivity = lambda x, y: (1 / (4 + 5 * (x - 0.5) ** 2), 1 / 2, 1 / 5)


def AD_fun(x, y):
    return conductivity_AD(
        x,
        y,
        A0_fun,
        (ellipse_conductivity, square_conductivity, mouth_conductivity),
        shape_choice,
        None,
    )


plot_conductivity_components(AD_fun, title="")

plt.savefig("Figures/Negative_definite_inclusions/Conductivity_AD.pdf")

# %%

solverA0 = FemConductivitySolver(mesh_characteristic_length=characteristic_length)
solverA0.set_conductivity(A0_fun, A0_fun, pos_def=False)

recon_solver = FemConductivitySolver(mesh_characteristic_length=recon_h)
recon_solver.set_conductivity(AD_fun, A0_fun, pos_def=False)
recon_mesh = recon_solver.mesh

mesh = solverA0.mesh

# Computing the "optimal" alpha, beta, c constants from checking the conductivity on each element
solverAD = FemConductivitySolver(mesh=mesh)
c, alpha, beta = solverAD.set_conductivity(AD_fun, A0_fun, pos_def=False)

print(f"c = {c}, alpha = {alpha}, beta = {beta}")

# %%

solverA0.compute_ntd_map(max_freq=max_freq_algo_2)
solverAD.compute_ntd_map(max_freq=max_freq_algo_2)

recon_basis_solutions = interpolate_solutions_to_new_mesh(solverA0, recon_solver)

# AD-A0 >= cI and alpha I <= AD <= beta I

rho = -c
print(rho)

ntd_A0 = solverA0.ntd_matrix
ntd_AD = solverAD.ntd_matrix

# %%

gradients = precompute_gradients(recon_basis_solutions, recon_mesh)

# %% #######################################

### Reconstruction without noisy data

eigenvalue_dict, _ = compute_reconstruction_test_values(
    mesh=recon_mesh,
    gradients=gradients[algo_1_idx[0] : algo_1_idx[1]],
    ntd_AD=ntd_AD[algo_1_idx[0] : algo_1_idx[1], algo_1_idx[0] : algo_1_idx[1]],
    ntd_A0=ntd_A0[algo_1_idx[0] : algo_1_idx[1], algo_1_idx[0] : algo_1_idx[1]],
    rho=rho,
    pos_def=False,
)

target_inclusion = recon_solver.get_target_inclusion_indeces()

# %%

# plot_highlighted_elements(recon_mesh, target_inclusion)

recon_errors_reg = []
alpha_regs = []

if np.min(np.real(np.linalg.eigvals(ntd_AD - ntd_A0))) < 0:
    mus = mus_pos
else:
    mus = mus_neg

for mu in mus:
    alpha_reg = -mu * np.min(np.real(np.linalg.eigvals(ntd_AD - ntd_A0)))
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

    plt.savefig(
        "Figures/Negative_definite_inclusions/Method_1_no_noise.pdf",
        bbox_inches="tight",
    )


# %%

rho0 = -0.1
rho_step = -0.5
if np.min(np.real(np.linalg.eigvals(ntd_AD - ntd_A0))) < 0:
    mu_2 = mu_2_pos
else:
    mu_2 = mu_2_neg
alpha_large = -mu_2 * np.min(np.real(np.linalg.eigvals(ntd_AD - ntd_A0)))

opt_mu_vals.append(mus[optim_alpha_idx])
opt_alpha_vals.append(optim_alpha)
alpha_large_vals.append(alpha_large)
mu_2_choices.append(mu_2)

inclusion_counter = reconstruction2(
    recon_mesh,
    gradients,
    ntd_AD,
    ntd_A0,
    alpha_large,
    rho0,
    rho_step,
    max_it=300,
    minimum_inclusions=1,
    print_output=True,
    pos_def=False,
)

fig, ax = plot_method_2(recon_mesh, inclusion_counter)
plt.savefig(
    "Figures/Negative_definite_inclusions/Method_2_no_noise.pdf",
    bbox_inches="tight",
)

# %%

rhos = reconstruction3(
    recon_mesh, gradients, ntd_AD, ntd_A0, alpha_large, pos_def=False
)

fig, ax = plot_method_3(recon_mesh, rhos, pos_def=False)
plt.savefig(
    "Figures/Negative_definite_inclusions/Method_3_no_noise.pdf",
    bbox_inches="tight",
)

# %%

delta = 1e-4

ntd_AD_noise, relative_noise = add_noise(ntd_AD, noise_level=delta, seed=seed)

eigenvalue_dict_noise, _ = compute_reconstruction_test_values(
    mesh=recon_mesh,
    gradients=gradients[algo_1_idx[0] : algo_1_idx[1]],
    ntd_AD=ntd_AD_noise[algo_1_idx[0] : algo_1_idx[1], algo_1_idx[0] : algo_1_idx[1]],
    ntd_A0=ntd_A0[algo_1_idx[0] : algo_1_idx[1], algo_1_idx[0] : algo_1_idx[1]],
    rho=rho,
    pos_def=False,
)

recon_errors_reg = []
alpha_regs = []

if np.min(np.real(np.linalg.eigvals(ntd_AD_noise - ntd_A0))) < 0:
    mus = mus_pos
else:
    mus = mus_neg

for mu in mus:
    alpha_reg = -mu * np.min(np.real(np.linalg.eigvals(ntd_AD_noise - ntd_A0)))
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
        "Figures/Negative_definite_inclusions/Method_1_noise.pdf",
        bbox_inches="tight",
    )
else:
    print("No inclusion reconstructed for optimal alpha with noise")

rho0 = -0.1
rho_step = -0.5
if np.min(np.real(np.linalg.eigvals(ntd_AD_noise - ntd_A0))) < 0:
    mu_2 = mu_2_pos
else:
    mu_2 = mu_2_neg
alpha_large = -mu_2 * np.min(np.real(np.linalg.eigvals(ntd_AD_noise - ntd_A0)))

opt_mu_vals.append(mus[optim_alpha_idx])
opt_alpha_vals.append(optim_alpha)
alpha_large_vals.append(alpha_large)
mu_2_choices.append(mu_2)
relative_noise_vals.append(relative_noise)

inclusion_counter = reconstruction2(
    recon_mesh,
    gradients,
    ntd_AD_noise,
    ntd_A0,
    alpha_large,
    rho0,
    rho_step,
    max_it=1000,
    minimum_inclusions=1,
    print_output=True,
    pos_def=False,
)

fig, ax = plot_method_2(recon_mesh, inclusion_counter)
plt.savefig(
    "Figures/Negative_definite_inclusions/Method_2_noise.pdf",
    bbox_inches="tight",
)

rhos = reconstruction3(
    recon_mesh, gradients, ntd_AD_noise, ntd_A0, alpha_large, pos_def=False
)

fig, ax = plot_method_3(recon_mesh, rhos, pos_def=False)
plt.savefig(
    "Figures/Negative_definite_inclusions/Method_3_noise.pdf",
    bbox_inches="tight",
)

# plt.show()

# %%

delta = 5e-4

ntd_AD_noise, relative_noise = add_noise(ntd_AD, noise_level=delta, seed=seed)

eigenvalue_dict_noise, _ = compute_reconstruction_test_values(
    mesh=recon_mesh,
    gradients=gradients[algo_1_idx[0] : algo_1_idx[1]],
    ntd_AD=ntd_AD_noise[algo_1_idx[0] : algo_1_idx[1], algo_1_idx[0] : algo_1_idx[1]],
    ntd_A0=ntd_A0[algo_1_idx[0] : algo_1_idx[1], algo_1_idx[0] : algo_1_idx[1]],
    rho=rho,
    pos_def=False,
)

recon_errors_reg = []
alpha_regs = []

if np.min(np.real(np.linalg.eigvals(ntd_AD_noise - ntd_A0))) < 0:
    mus = mus_pos
else:
    mus = mus_neg

for mu in mus:
    alpha_reg = -mu * np.min(np.real(np.linalg.eigvals(ntd_AD_noise - ntd_A0)))
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
        "Figures/Negative_definite_inclusions/Method_1_more_noise.pdf",
        bbox_inches="tight",
    )
else:
    print("No inclusion reconstructed for optimal alpha with noise")

rho0 = -0.1
rho_step = -0.5
if np.min(np.real(np.linalg.eigvals(ntd_AD_noise - ntd_A0))) < 0:
    mu_2 = mu_2_pos
else:
    mu_2 = mu_2_neg
alpha_large = -mu_2 * np.min(np.real(np.linalg.eigvals(ntd_AD_noise - ntd_A0)))

opt_mu_vals.append(mus[optim_alpha_idx])
opt_alpha_vals.append(optim_alpha)
alpha_large_vals.append(alpha_large)
mu_2_choices.append(mu_2)
relative_noise_vals.append(relative_noise)

inclusion_counter = reconstruction2(
    recon_mesh,
    gradients,
    ntd_AD_noise,
    ntd_A0,
    alpha_large,
    rho0,
    rho_step,
    max_it=1000,
    minimum_inclusions=1,
    print_output=True,
    pos_def=False,
)

fig, ax = plot_method_2(recon_mesh, inclusion_counter)
plt.savefig(
    "Figures/Negative_definite_inclusions/Method_2_more_noise.pdf",
    bbox_inches="tight",
)

rhos = reconstruction3(
    recon_mesh, gradients, ntd_AD_noise, ntd_A0, alpha_large, pos_def=False
)

fig, ax = plot_method_3(recon_mesh, rhos, pos_def=False)
plt.savefig(
    "Figures/Negative_definite_inclusions/Method_3_more_noise.pdf",
    bbox_inches="tight",
)

# plt.show()

print(f"Opt mus {opt_mu_vals}")
print(f"Opt alphas {opt_alpha_vals}")
print(f"mu 2 {mu_2_choices}")
print(f"alpha large {alpha_large_vals}")
print(f"relative noise {relative_noise_vals}")

# %%
