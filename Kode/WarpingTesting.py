# %% Imports and initialization
from fem_solver import (FemConductivitySolver,
    interpolate_solutions_to_new_mesh)
import matplotlib.pyplot as plt
import ufl
from inner_reconstruction_method import find_inclusion_elements
from plotting_functions import *
from conductivity_functions import *

#%%
characteristic_length = 0.01
num_partitions = None
max_freq = 15

shape_choice = "ellipse"
params = [[0.4,0],[0.4,0.4]]#[[0,0],[0.4,0.4]]

# Define conductivities
my_conductivity_A0 = lambda x, y: (1, 0, 1)
inclusion_conductivity = lambda x, y: (5, 0, 2)

def my_conductivity_AD(x, y):
    return conductivity_AD(
        x,
        y,
        my_conductivity_A0,
        inclusion_conductivity,
        shape_choice,
        params
    )

# Visualization
fig, axes = plot_conductivity_components(
    my_conductivity_AD, title=f"{shape_choice} Conductivity Distribution"
)

solverA0 = FemConductivitySolver(mesh_characteristic_length=characteristic_length)
solverA0.set_conductivity(my_conductivity_A0, my_conductivity_A0)

mesh = solverA0.mesh

# %%
solverAD = FemConductivitySolver(mesh=mesh)
c, alpha, beta = solverAD.set_conductivity(my_conductivity_AD, my_conductivity_A0)

solverA0.compute_ntd_map(max_freq=max_freq)
solverAD.compute_ntd_map(max_freq=max_freq)


ntd_A0 = solverA0.ntd_matrix
ntd_AD = solverAD.ntd_matrix
# %% Reconstruction Mesh
recon_characteristic_length = 0.07
recon_solver = FemConductivitySolver(mesh_characteristic_length=recon_characteristic_length)
recon_mesh = recon_solver.mesh

sols = interpolate_solutions_to_new_mesh(
    solverA0,
    recon_solver)

recon_solver.set_conductivity(my_conductivity_AD, my_conductivity_A0)

const = c * (alpha**2) / (beta**2) 

# %%
mu = 1.00
tol = 0#-mu*np.min(np.real(np.linalg.eigvals(((ntd_A0-ntd_AD)+(ntd_A0-ntd_AD).T)/2)))

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
plot_highlighted_elements_with_inclusions(recon_mesh, inclusion_indexes, recon_solver.cell_tags)


plt.show()
# %% #######################
inclusion_conductivity_2 = lambda x, y: (2, 0, 5)  # Different values for comparison

def my_conductivity_AD_2(x, y):
    return conductivity_AD(
        x,
        y,
        my_conductivity_A0,
        inclusion_conductivity_2,
        shape_choice,
        params
    )

# Create new solver with different conductivity
solverAD_2 = FemConductivitySolver(mesh=mesh)
c2, alpha2, beta2 = solverAD_2.set_conductivity(my_conductivity_AD_2, my_conductivity_A0)
solverAD_2.compute_ntd_map(max_freq=max_freq)

# Adjust parameters
const2 = c2 * (alpha2**2) / (beta2**2) 

# Get NTD for second conductivity
ntd_AD_2 = solverAD_2.ntd_matrix

# Run reconstruction for second conductivity
inclusion_indexes_2, test_matrix_eigenvalues_2, partitions_2 = find_inclusion_elements(
    mesh=recon_mesh,
    solutions_A0=sols,
    ntd_AD=ntd_AD_2,
    ntd_A0=ntd_A0,
    num_partitions=num_partitions,
    const=const,
    tol=0,
)

# %%
plot_highlighted_eigenvalues(recon_mesh, inclusion_indexes, test_matrix_eigenvalues)

plt.show()

plot_highlighted_eigenvalues(recon_mesh, inclusion_indexes_2, test_matrix_eigenvalues_2)

plt.show()

plot_highlighted_elements_with_inclusions(recon_mesh, inclusion_indexes, np.array(inclusion_indexes_2))
# %%


ev_diff = np.array(list(test_matrix_eigenvalues.values()))-np.array(list(test_matrix_eigenvalues_2.values()))
print(np.max(ev_diff))
print(np.sum(np.abs(ev_diff)>1e-6))
dbin = ev_diff > 0
didx = np.where(dbin)[0]



# %%
plot_highlighted_eigenvalues(recon_mesh, inclusion_indexes_2, ev_diff)
plt.show()
# %%
plot_highlighted_elements_with_inclusions(recon_mesh, didx)
plt.show()

# %%
plot_test_matrix_eigenvalues(
    recon_mesh, ev_diff, sigmoid_scaling=None)

plt.show()
# %% #######################
