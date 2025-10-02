# %% Imports and initialization
from packaging.version import Version
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx.fem.petsc
from dolfinx.fem import Function, assemble_scalar
from dolfinx.mesh import locate_entities

import numpy as np
from scifem import create_real_functionspace, assemble_scalar
from dolfinx.fem.petsc import apply_lifting, set_bc
import ufl
import pyvista
from dolfinx.io import gmshio
import gmsh
from dolfinx.cpp.la.petsc import scatter_local_vectors, get_local_vectors

import matplotlib.pyplot as plt
import matplotlib.tri as tri

gmsh.initialize()


# %% Set up subdomains and conductivities
def Omega_0(x):
    return np.sqrt(x[1] ** 2 + x[0] ** 2) <= 0.5


def Omega_1(x):
    return np.sqrt(x[1] ** 2 + x[0] ** 2) > 0.5


# Define your conductivity functions for each subdomain
def conductivity_Omega_0(x, y):
    """Spatially varying conductivity in inner region (r <= 0.5)"""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    k11 = 1.0 + 0.5 * np.sin(4 * theta)
    k12 = 0.2 * np.cos(2 * theta) * r
    k22 = 1.0 + 0.3 * np.cos(4 * theta)

    return k11, k12, k22


def conductivity_Omega_1(x, y):
    """Spatially varying conductivity in outer region (r >= 0.5)"""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    k11 = 0.1 + 0.9 * (r - 0.5)
    k12 = 0.05 * np.sin(8 * theta)
    k22 = 0.2 + 0.8 * (r - 0.5)

    return k11, k12, k22


# %% create unit disk mesh with gmsh
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


# %% setting up functions
# div(grad(u)) = 0 in Omega
# du/dn = f on dOmega
# int_dOmega u dx = 0
exact_solution = False
x = ufl.SpatialCoordinate(mesh)
u_exact = x[0]  # UFL expression: u(x,y) = x

# n = ufl.FacetNormal(mesh)
theta = ufl.atan2(x[1], x[0])
f = ufl.cos(theta)

# %% Set up subdomains and conductivities

# Create tensor function space for 2x2 matrices
V_tensor = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2, 2)))
conductivity = Function(V_tensor)

# Get the dofmap and geometry
dofmap = V_tensor.dofmap
mesh_geometry = mesh.geometry.x
cell_geometry = mesh.geometry.dofmap  # This is a numpy array

cells_0 = locate_entities(mesh, mesh.topology.dim, Omega_0)
cells_1 = locate_entities(mesh, mesh.topology.dim, Omega_1)

# Set values for Omega_0 cells
for cell in cells_0:
    cell_dofs = dofmap.cell_dofs(cell)
    geometry_dofs = cell_geometry[cell]  # Use array indexing instead of .links()

    for i, dof in enumerate(cell_dofs):
        # Get the coordinate of this dof
        coord_idx = geometry_dofs[i // 4]  # 4 components per dof
        x, y = mesh_geometry[coord_idx][0], mesh_geometry[coord_idx][1]

        # Compute conductivity matrix at this point
        k11, k12, k22 = conductivity_Omega_0(x, y)

        # Set the tensor components
        conductivity.x.array[dof * 4 + 0] = k11
        conductivity.x.array[dof * 4 + 1] = k12
        conductivity.x.array[dof * 4 + 2] = k12  # symmetric
        conductivity.x.array[dof * 4 + 3] = k22

# Set values for Omega_1 cells
for cell in cells_1:
    cell_dofs = dofmap.cell_dofs(cell)
    geometry_dofs = cell_geometry[cell]  # Use array indexing

    for i, dof in enumerate(cell_dofs):
        coord_idx = geometry_dofs[i // 4]
        x, y = mesh_geometry[coord_idx][0], mesh_geometry[coord_idx][1]

        k11, k12, k22 = conductivity_Omega_1(x, y)

        conductivity.x.array[dof * 4 + 0] = k11
        conductivity.x.array[dof * 4 + 1] = k12
        conductivity.x.array[dof * 4 + 2] = k12  # symmetric
        conductivity.x.array[dof * 4 + 3] = k22

# %% create function space for real numbers and mixed function space
R = create_real_functionspace(mesh)
W = ufl.MixedFunctionSpace(V, R)
u, lmbda = ufl.TrialFunctions(W)
du, dl = ufl.TestFunctions(W)

# %% defining variational problem
zero = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))

a00 = ufl.inner(conductivity * ufl.grad(u), ufl.grad(du)) * ufl.dx
a01 = ufl.inner(lmbda, du) * ufl.ds
a10 = ufl.inner(u, dl) * ufl.ds
L0 = ufl.inner(f, du) * ufl.ds
L1 = ufl.inner(zero, dl) * ufl.ds

# %% assemble system
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

# %% right hand side
try:
    b = dolfinx.fem.petsc.assemble_vector_block(L_compiled, a_compiled, bcs=[])
except AttributeError:
    b = dolfinx.fem.petsc.assemble_vector(L_compiled, kind="mpi")
    apply_lifting(b, [], a_compiled)
    set_bc(b, [])

# %% solve linear system
ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")

xh = dolfinx.fem.petsc.create_vector_block(L_compiled)
xh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # ??

ksp.solve(b, xh)

# %%

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

# %% post processing
# Create a Function for the exact solution and interpolate
if exact_solution:
    u_exact_func = Function(V)
    u_exact_func.interpolate(lambda x: x[0])  # Interpolate u(x,y) = x

    diff = uh - u_exact_func
    error = ufl.inner(diff, diff) * ufl.dx
    print(f"L2 error: {np.sqrt(assemble_scalar(error)):.2e}")

# %% plotting (doesnt work for me)
# vtk_mesh = dolfinx.plot.vtk_mesh(V)
# grid = pyvista.UnstructuredGrid(*vtk_mesh)
# grid.point_data["u"] = uh.x.array.real
# warped = grid.warp_by_scalar("u", factor=1)
# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, style="wireframe")
# plotter.add_mesh(warped)
# if not pyvista.OFF_SCREEN:
#     plotter.show()
# %% matplotlib plotting as per chatGPT

coords = mesh.geometry.x.copy()  # shape (num_vertices, 2)
cells = mesh.topology.connectivity(mesh.topology.dim, 0).array.reshape(
    -1, mesh.topology.dim + 1
)

# Flatten for matplotlib
x = coords[:, 0]
y = coords[:, 1]

# FEM solution values at vertices
u_vals = uh.x.array.real

# Triangulation for matplotlib

triang = tri.Triangulation(x, y, cells)

# Plot filled contour
plt.figure(figsize=(6, 5))
cont = plt.tricontourf(triang, u_vals, levels=50, cmap="viridis")
plt.colorbar(cont, label="u")
plt.tricontour(
    triang, u_vals, levels=10, colors="k", linewidths=0.5
)  # optional contours
plt.xlabel("x")
plt.ylabel("y")
plt.title("FEM solution u")
plt.axis("equal")
plt.show()
