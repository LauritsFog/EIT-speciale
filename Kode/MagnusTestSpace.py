# %% Imports and initialization
from packaging.version import Version
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx.fem.petsc
import numpy as np
from scifem import create_real_functionspace, assemble_scalar
from dolfinx.fem.petsc import apply_lifting, set_bc
import ufl
import pyvista
from dolfinx.io import gmshio
import gmsh
import matplotlib.pyplot as plt
from plotting_functions import *
from inner_reconstruction_method import *

#%%
from fem_solver import FemConductivitySolver

solverA0 = FemConductivitySolver()

def my_conductivity_A0(x, y):
    k11 = 1
    k12 = 0
    k22 = 1
    return k11, k12, k22

class Inclusion:
    def __init__(self, condition_function, k11, k12=None, k22=None):
        # condition_function: (x, y) -> real number (inside if <= 0, outside if > 0)
        self.condition = condition_function 
        # k11, k12, k22: conductivity components inside the inclusion
        # functions: (x,y) -> real number
        # If k12 or k22 is None, it is assumed to be 0 or k11 respectively
        if np.isscalar(k11):
            self.k11 = lambda x, y: k11
        else:
            self.k11 = k11
        if k12 is None:
            self.k12 = lambda x, y: 0
        else:
            if np.isscalar(k12):
                self.k12 = lambda x, y: k12
            else:
                self.k12 = k12
        if k22 is None:
            self.k22 = self.k11
        else:
            if np.isscalar(k22):
                self.k22 = lambda x, y: k22
            else:
                self.k22 = k22

D1 = Inclusion(lambda x, y: (x-0.3)**2/0.3**2 + (y-0.3)**2/0.25**2 - 1, k11=9)
D2 = Inclusion(lambda x, y: (x+0.3)**2/0.3**2 + (y+0.3)**2/0.25**2 - 1, k11=9)
Dcircle = Inclusion(lambda x, y: x**2 + y**2 - 0.5**2, k11=9)
inclusions = [D1, D2]

def inclusion_conductivity(background_conductivity, inclusions):
    def conductivity_function(x, y):
        k11, k12, k22 = background_conductivity(x, y)
        for inclusion in inclusions:
            if inclusion.condition(x, y) <= 0:
                k11 += inclusion.k11(x, y)
                k12 += inclusion.k12(x, y)
                k22 += inclusion.k22(x, y)
        return k11, k12, k22
    return conductivity_function

my_conductivity_AD = inclusion_conductivity(my_conductivity_A0, inclusions)

# Set conductivity with custom functions
solverA0.set_conductivity(
    my_conductivity_A0
)

solverA0.compute_ntd_map(max_freq=10)

mesh = solverA0.mesh

solverAD = FemConductivitySolver(mesh=mesh)
solverAD.set_conductivity(
    my_conductivity_AD
)
solverAD.compute_ntd_map(max_freq=10)
solverAD.plot_conductivity_components()

ntd_A0 = solverA0.ntd_matrix
ntd_AD = solverAD.ntd_matrix
# %%
# With diagonal conductivity matrices (AD-A0 >= cI). If c = 0, then AD-A0 = cI i Lo
c = 9
# Lower bound on AD (alpha I <= AD <= beta I)
alpha = 1
# Upper bound on AD
beta = 10

const = c * (alpha**2) / (beta**2) * 100

# Call the function directly
inclusion_indexes, test_matrix_eigenvalues = find_inclusion_elements_manual(
    mesh=mesh,
    solutions_A0=solverA0.basis_solutions,
    ntd_AD=ntd_AD,
    ntd_A0=ntd_A0,
    sampling_stride=1,
    const=const,
)

_ , ax = plot_highlighted_elements(mesh, inclusion_indexes, inclusions)


#plot_test_matrix_test_matrix_eigenvalues(mesh, test_matrix_eigenvalues)

plt.show()
# %%
