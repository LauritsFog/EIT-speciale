# %% Imports and initialization
from fem_solver import (
    FemConductivitySolver,
    interpolate_solutions_to_new_mesh)
import matplotlib.pyplot as plt
import ufl
from inner_reconstruction_method import (
    find_inclusion_elements)
from plotting_functions import *
from conductivity_functions import *
# %%

characteristic_length = 0.005
num_partitions = None

max_freq = 15

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
        x,
        y,
        my_conductivity_A0,
        inclusion_conductivity,
        shape_choice,
        None
    )

# Visualization
fig, axes = plot_conductivity_components(
    my_conductivity_AD, cmap = 'binary',isotropic=True)
plt.show()

#fig.savefig("isotropic_target.pdf", bbox_inches="tight")
#%%
solverA0 = FemConductivitySolver(mesh_characteristic_length=characteristic_length)
solverA0.set_conductivity(my_conductivity_A0, my_conductivity_A0)

mesh = solverA0.mesh

#%%

# Computing the "optimal" alpha, beta, c constants from checking the conductivity on each element
solverAD = FemConductivitySolver(mesh=mesh)
c, alpha, beta = solverAD.set_conductivity(my_conductivity_AD, my_conductivity_A0)

solverA0.compute_ntd_map(max_freq=max_freq)
solverAD.compute_ntd_map(max_freq=max_freq)

c = c - c_tol
alpha = alpha - alpha_tol
beta = beta + beta_tol

const = c * (alpha**2) / (beta**2)

ntd_A0 = solverA0.ntd_matrix
ntd_AD = solverAD.ntd_matrix

#%% new mesh
recon_characteristic_length = 0.06
mesh_solver = FemConductivitySolver(mesh_characteristic_length=recon_characteristic_length)
recon_mesh = mesh_solver.mesh

sols = interpolate_solutions_to_new_mesh(
    solverA0,
    mesh_solver)

mesh_solver.set_conductivity(my_conductivity_AD, my_conductivity_A0)


#%%
# Tolerance in the inclusion test: min(eigenvalues) + tol > 0
mu = 0.99
tol = 0#-mu*np.min(np.real(np.linalg.eigvals(ntd_A0-ntd_AD)))

inclusion_indexes, test_matrix_eigenvalues, partitions = find_inclusion_elements(
    mesh=recon_mesh,
    solutions_A0=sols,
    ntd_AD=ntd_AD,
    ntd_A0=ntd_A0,
    num_partitions=num_partitions,
    const=const,
    tol=tol,
)

#%%
plot_highlighted_elements_with_inclusions(recon_mesh, inclusion_indexes, mesh_solver.cell_tags)

fig, ax = plot_highlighted_eigenvalues(recon_mesh, inclusion_indexes, test_matrix_eigenvalues)
# plot_mesh_partitions(mesh, partitions)
#fig.savefig("isotropic_reconstruction.pdf", bbox_inches="tight")

plt.show()


# %%
