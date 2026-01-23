# %% Imports and initialization
from fem_solver import FemConductivitySolver, interpolate_solutions_to_new_mesh
import matplotlib.pyplot as plt
import ufl
from inner_reconstruction_method import *
from plotting_functions import *
from conductivity_functions import *
from reconstruction_method_2 import *
# %%

characteristic_length = 0.01
num_partitions = None

max_freq = 10

c_tol = 0
alpha_tol = 0
beta_tol = 0

shape_choice = "isotropy_test"

# Define conductivities
my_conductivity_A0 = lambda x, y: (1, 0, 1)

cond = 5
inclusion_conductivity = lambda x, y: (cond, 0, cond)


def my_conductivity_AD(x, y):
    return conductivity_AD(
        x, y, my_conductivity_A0, inclusion_conductivity, shape_choice, None
    )


# Visualization
# fig, axes = plot_conductivity_components(
#   my_conductivity_AD, cmap = 'binary', isotropic=True)
# plt.show()

# fig.savefig("isotropic_target.pdf", bbox_inches="tight")
# %%
solverA0 = FemConductivitySolver(mesh_characteristic_length=characteristic_length)
solverA0.set_conductivity(my_conductivity_A0, my_conductivity_A0)

mesh = solverA0.mesh

# %%

# Computing the "optimal" alpha, beta, c constants from checking the conductivity on each element
solverAD = FemConductivitySolver(mesh=mesh)
c, alpha, beta = solverAD.set_conductivity(my_conductivity_AD, my_conductivity_A0)
c = c - c_tol
alpha = alpha - alpha_tol
beta = beta + beta_tol

const = c * (alpha**2) / (beta**2)

# %%
solverA0.compute_ntd_map(max_freq=max_freq)
solverAD.compute_ntd_map(max_freq=max_freq)

ntd_A0 = solverA0.ntd_matrix
ntd_AD = solverAD.ntd_matrix

# %% new mesh
recon_characteristic_length = 0.05
mesh_solver = FemConductivitySolver(
    mesh_characteristic_length=recon_characteristic_length
)
recon_mesh = mesh_solver.mesh

sols = interpolate_solutions_to_new_mesh(solverA0, mesh_solver)

mesh_solver.set_conductivity(my_conductivity_AD, my_conductivity_A0)


# %%
# Tolerance in the inclusion test: min(eigenvalues) + tol > 0
mu = 0.97
M = ntd_A0 - ntd_AD
Msym = 0.5 * (M + M.T)
tol = 0
const_garde = 0.8

print(f"Const garde: {const_garde}")
print(f"Const: {const}  ")

gradients = precompute_gradients(sols, recon_mesh)
# %%
ev_dict, fr_dict = compute_reconstruction_test_values(
    recon_mesh, gradients, ntd_AD, ntd_A0, const
)
inclusion_indexes = reconstruct_inclusions(ev_dict, tol)


# %%

fig, ax = plot_highlighted_eigenvalues(recon_mesh, inclusion_indexes, ev_dict)
fig.savefig("Figures/Isotropy/isotropic_reconstruction.pdf", bbox_inches="tight")

gradients = precompute_gradients(sols, recon_mesh)

ev_dict, fr_dict = compute_reconstruction_test_values(
    recon_mesh, gradients, ntd_AD, ntd_A0, const_garde
)
inclusion_indexes = reconstruct_inclusions(ev_dict, tol)

fig, ax = plot_highlighted_eigenvalues(recon_mesh, inclusion_indexes, ev_dict)
fig.savefig(
    "Figures/Isotropy/isotropic_reconstruction_w_iso_probing.pdf", bbox_inches="tight"
)


# %% ####################

noiselevel = 0
ntd_AD_noisy = add_noise(ntd_AD, noiselevel)
M = ntd_A0 - ntd_AD_noisy
Msym = 0.5 * (M + M.T)
# %%
mu = 0  # 1.0001
# tol = -mu * np.min(np.real(np.linalg.eigvals(Msym)))
tol = 1e-5
print("Regularisation:", tol)
# %%
gradients = precompute_gradients(sols, recon_mesh)

# %%
ev_dict, fr_dict = compute_reconstruction_test_values(
    recon_mesh, gradients, ntd_AD_noisy, ntd_A0, const
)
inclusion_indexes = reconstruct_inclusions(ev_dict, tol)
# %%

fig, ax = plot_isotropic_conductivity(my_conductivity_AD, title="")
plt.savefig("Figures/Isotropy/isotropic_target.pdf", bbox_inches="tight")

fig, ax = plot_highlighted_eigenvalues(recon_mesh, inclusion_indexes, ev_dict)

# %% ###### METHOD 2 ##########
rho0 = 0.1
rho_step = 0.25
# alpha = -mu * np.min(np.real(np.linalg.eigvals(Msym)))
alpha = 1e-5
inclusion_counter = reconstruction2(
    recon_mesh, gradients, ntd_AD_noisy, ntd_A0, alpha, rho0, rho_step, max_it=1000
)
# %%
fig, ax = plot_method_2(recon_mesh, inclusion_counter)
plt.savefig("Figures/Isotropy/Method_2_isotropic_recon.pdf", bbox_inches="tight")

# %%
# plt.show()
