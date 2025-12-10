# %% Imports and initialization
from fem_solver import FemConductivitySolver
import matplotlib.pyplot as plt
import ufl
from inner_reconstruction_method import find_inclusion_elements
from plotting_functions import *
from conductivity_functions import *
# %%

characteristic_length = 0.05
num_partitions = None

max_freq = 15



c_tol = 0
alpha_tol = 0
beta_tol = 0

shape_choice = "isotropy_test"

# Define conductivities
my_conductivity_A0 = lambda x, y: (1, 0, 1)

inclusion_conductivity = lambda x, y: (5, 0, 5)

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
    my_conductivity_AD, title=f"{shape_choice} Conductivity Distribution"
)

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

const = 0.8/5#c * (alpha**2) / (beta**2)

ntd_A0 = solverA0.ntd_matrix
ntd_AD = solverAD.ntd_matrix

#%%
# Tolerance in the inclusion test: min(eigenvalues) + tol > 0
mu = 0.99
tol = -mu*np.min(np.real(np.linalg.eigvals(ntd_A0-ntd_AD)))

inclusion_indexes, test_matrix_eigenvalues, partitions = find_inclusion_elements(
    mesh=mesh,
    solutions_A0=solverA0.basis_solutions,
    ntd_AD=ntd_AD,
    ntd_A0=ntd_A0,
    num_partitions=num_partitions,
    const=const,
    tol=tol,
)

#%%
plot_highlighted_elements_with_inclusions(mesh, inclusion_indexes, solverAD.cell_tags)

plot_test_matrix_eigenvalues(
    mesh, test_matrix_eigenvalues, sigmoid_scaling=None
)

# plot_mesh_partitions(mesh, partitions)

plt.show()


# %%
