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

# %% plotting (doesnt work for me)
vtk_mesh = dolfinx.plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(*vtk_mesh)
grid.point_data["u"] = uh.x.array.real
warped = grid.warp_by_scalar("u", factor=1)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, style="wireframe")
plotter.add_mesh(warped)
if not pyvista.OFF_SCREEN:
    plotter.show()
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
# %%
