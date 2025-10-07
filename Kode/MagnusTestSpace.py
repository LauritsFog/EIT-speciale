#%% Imports and initialization
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

#%%
gmsh.initialize()
#%% create unit disk mesh with gmsh
unitdisk = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
gmsh.model.occ.synchronize()

gdim = 2
gmsh.model.addPhysicalGroup(gdim, [unitdisk], 1)

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
gmsh.model.mesh.generate(gdim)

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD

mesh, cell_markers, facet_markers = gmshio.model_to_mesh(
    gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim
)

V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
#%% setting up functions 
# div(Conductivity*grad(u)) = 0 in Omega
# du/dn = f on dOmega
# int_dOmega u dx = 0
def u_exact(x):
    return x[0]
x = ufl.SpatialCoordinate(mesh)
#n = ufl.FacetNormal(mesh)
theta = ufl.atan2(x[1], x[0])
f = 2 * ufl.cos(theta) + ufl.sin(theta)
Conductivity = ufl.as_matrix([[2.0, 1.0],
                   [1.0, 3.0]])
# %% create function space for real numbers and mixed function space
R = create_real_functionspace(mesh)
W = ufl.MixedFunctionSpace(V, R)
u, lmbda = ufl.TrialFunctions(W)
du, dl = ufl.TestFunctions(W)

#%% defining variational problem
zero = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))

a00 = ufl.inner(Conductivity * ufl.grad(u), ufl.grad(du)) * ufl.dx
a01 = ufl.inner(lmbda, du) * ufl.ds
a10 = ufl.inner(u, dl) * ufl.ds
L0 = ufl.inner(f, du) * ufl.ds
L1 = ufl.inner(zero, dl) * ufl.ds

#%% assemble system
a = [[a00, a01], [a10, None]]
L = [L0, L1]
a_compiled = dolfinx.fem.form(a)
L_compiled = dolfinx.fem.form(L)

# assembling system matrix
try:
    A = dolfinx.fem.petsc.assemble_matrix_block(a_compiled)
except AttributeError:
    A = dolfinx.fem.petsc.assemble_matrix(a_compiled)
A.assemble()

#%% right hand side
try:
    b = dolfinx.fem.petsc.assemble_vector_block(L_compiled, a_compiled, bcs=[])
except AttributeError:
    b = dolfinx.fem.petsc.assemble_vector(L_compiled, kind="mpi")
    apply_lifting(b, [], a_compiled)
    set_bc(b, [])

#%% solve linear system
ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")

xh = dolfinx.fem.petsc.create_vector_block(L_compiled)
xh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD) #??

ksp.solve(b, xh)

#%%
from dolfinx.cpp.la.petsc import scatter_local_vectors, get_local_vectors
maps = [(Wi.dofmap.index_map, Wi.dofmap.index_map_bs) for Wi in W.ufl_sub_spaces()]

uh = dolfinx.fem.Function(V, name="u")
x_local = get_local_vectors(xh, maps)
uh.x.array[: len(x_local[0])] = x_local[0]
uh.x.scatter_forward()

# %% destroy all petsc objects (?)
b.destroy()
xh.destroy()
A.destroy()
ksp.destroy()

#%% post processing
diff = uh - u_exact(x)
error = ufl.inner(diff, diff) * ufl.dx
print(f"L2 error: {np.sqrt(assemble_scalar(error)):.2e}")

# %% matplotlib plotting as per chatGPT
import matplotlib.pyplot as plt
coords = mesh.geometry.x.copy()       # shape (num_vertices, 2)
cells = mesh.topology.connectivity(mesh.topology.dim, 0).array.reshape(-1, mesh.topology.dim+1)

# Flatten for matplotlib
x = coords[:, 0]
y = coords[:, 1]

# FEM solution values at vertices
u_vals = uh.x.array.real

# Triangulation for matplotlib
import matplotlib.tri as tri
triang = tri.Triangulation(x, y, cells)

# Plot filled contour
plt.figure(figsize=(6,5))
cont = plt.tricontourf(triang, u_vals, levels=50, cmap="viridis")
plt.colorbar(cont, label="u")
plt.tricontour(triang, u_vals, levels=10, colors="k", linewidths=0.5)  # optional contours
plt.xlabel("x")
plt.ylabel("y")
plt.title("FEM solution u")
plt.axis("equal")
plt.show()
# %% u on the boundary ===================================
# mesh.geometry.x: all vertex coordinates
coords_all = mesh.geometry.x

# Find boundary vertices (distance from origin ~ R)
tol = 1e-12
b_vertices = np.where(np.abs(np.linalg.norm(coords_all, axis=1) - 1.0) < tol)[0]

# Coordinates of boundary vertices
coords = coords_all[b_vertices]

# FEM solution at boundary vertices
u_boundary = uh.x.array[b_vertices]

# Compute angles for sorting
theta = np.arctan2(coords[:,1], coords[:,0])
# Shift angles so they are in [0, 2pi)
theta = (theta + 2*np.pi) % (2*np.pi)
sort_idx = np.argsort(theta)
theta_sorted = theta[sort_idx]
u_sorted = u_boundary[sort_idx]


# %%
# Number of points for FFT
N = len(u_sorted)
u_hat = np.fft.fft(u_sorted) / N  # no roll
freqs = np.fft.fftfreq(N, d=(2*np.pi)/N)

a_k = 2 * np.real(u_hat)   # cosine coefficients
b_k = -2 * np.imag(u_hat)  # sine coefficients

# %% ==================================================
from fem_solver import FemConductivitySolver

solverA0 = FemConductivitySolver()

def my_conductivity_A0(x, y):
    k11 = 1
    k12 = 0
    k22 = 1
    return k11, k12, k22

def my_conductivity_AD(x, y):
    k11, k12, k22 = my_conductivity_A0(x, y)
    if x**2 + y**2 < 0.5:
        k11 += 10
        k22 += 5        
    return k11, k12, k22


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

ntd_A0 = solverA0.ntd_matrix
ntd_AD = solverAD.ntd_matrix
# %%
def plot_mesh_2d_highlight_elements(mesh, element_indices):
    """
    Plot 2D mesh with highlighted elements using matplotlib
    
    Parameters:
    -----------
    mesh : dolfinx.mesh.Mesh
        The mesh
    element_indices : list or int
        List of element indices to highlight, or single index
    """
    # Convert single index to list for uniform handling
    if isinstance(element_indices, int):
        element_indices = [element_indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the entire mesh
    topology = mesh.topology
    tdim = topology.dim
    cells = topology.connectivity(tdim, 0).array.reshape(-1, tdim+1)
    vertices = mesh.geometry.x
    
    # Plot all cells (background mesh)
    for cell in cells:
        poly = plt.Polygon(vertices[cell][:, :2], fill=None, edgecolor='black', alpha=0.3)
        ax.add_patch(poly)
    
    # Highlight the specified elements in red
    for element_index in element_indices:
        if element_index < len(cells):
            highlight_cell = cells[element_index]
            poly_highlight = plt.Polygon(vertices[highlight_cell][:, :2], 
                                       fill=True, color='red', alpha=0.7)
            ax.add_patch(poly_highlight)
        else:
            print(f"Warning: Element index {element_index} out of range (0-{len(cells)-1})")
    
    # Set plot properties
    ax.set_aspect('equal')
    ax.autoscale_view()
    
    title = f"Mesh with {len(element_indices)} highlighted elements"
    if len(element_indices) == 1:
        title = f"Mesh with highlighted element {element_indices[0]}"
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
#%%
plot_mesh_2d_highlight_elements(mesh, 0)

#%%
# Usage for 2D mesh
element_index = 0  # Index of the element to highlight
solutions_A0 = solverA0.basis_solutions

# Create mesh tags marking only the target element
ct = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, 
                  np.array([element_index], dtype=np.int32), 
                  np.array([1], dtype=np.int32))

# Define the integration measure restricted to the single element
dx_single = ufl.Measure("dx", domain=mesh, subdomain_data=ct)

N = len(solutions_A0)
Dfrechet = np.empty((N, N))

const = 10
for i in range(N):
    for j in range(N):
        ui = solutions_A0[i]
        uj = solutions_A0[j]
        integrand = -const*ufl.dot(ufl.grad(ui), ufl.grad(uj))
        Dfrechet[i, j] = dolfinx.fem.assemble_scalar(dolfinx.fem.form(integrand * dx_single(1)))

check = Dfrechet - ntd_AD + ntd_A0
print(min(np.linalg.eigvals(check)))

# %%
num_elements = mesh.topology.index_map(mesh.topology.dim).size_local
inclusion_indexes = []
for elem in range(num_elements):
    if elem % 10 == 0:
        if elem % 50 == 0:
            print(f"Element {elem}/{num_elements}")
        ct = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, 
                        np.array([elem], dtype=np.int32), 
                        np.array([1], dtype=np.int32))
        dx_single = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
        Dfrechet = np.empty((N, N))
        const = 10
        for i in range(N):
            for j in range(N):
                ui = solutions_A0[i]
                uj = solutions_A0[j]
                integrand = -const*ufl.dot(ufl.grad(ui), ufl.grad(uj))
                Dfrechet[i, j] = dolfinx.fem.assemble_scalar(dolfinx.fem.form(integrand * dx_single(1)))
        check = Dfrechet - ntd_AD + ntd_A0
        if min(np.linalg.eigvals(check)) > 0:
            inclusion_indexes.append(elem)
#%%
plot_mesh_2d_highlight_elements(mesh, inclusion_indexes)
# %%
