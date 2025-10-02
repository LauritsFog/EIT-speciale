import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx.fem.petsc
from dolfinx.fem import (
    Function,
    Constant,
    functionspace,
    assemble_scalar,
    form,
    dirichletbc,
    locate_dofs_geometrical,
)
from dolfinx.mesh import locate_entities
import ufl
from dolfinx.io import gmshio
import gmsh
from scifem import create_real_functionspace
from dolfinx.fem.petsc import apply_lifting, set_bc


class FemConductivitySolver:
    def __init__(self, mesh_characteristic_length=0.05):
        self.mesh_characteristic_length = mesh_characteristic_length
        self.mesh = None
        self.V = None
        self.conductivity = None
        self.uh = None
        self.A = None
        self.b = None
        self.xh = None
        self.ksp = None
        self.W = None
        self._setup_mesh()

    def _setup_mesh(self):
        """Create the unit disk mesh"""
        gmsh.initialize()

        # Create disk
        unitdisk = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()

        gdim = 2
        gmsh.model.addPhysicalGroup(gdim, [unitdisk], 1)

        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMin", self.mesh_characteristic_length
        )
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMax", self.mesh_characteristic_length
        )
        gmsh.model.mesh.generate(gdim)

        gmsh_model_rank = 0
        mesh_comm = MPI.COMM_WORLD

        self.mesh, cell_markers, facet_markers = gmshio.model_to_mesh(
            gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim
        )

        gmsh.finalize()

        # Create function space
        self.V = functionspace(self.mesh, ("Lagrange", 1))

    def set_conductivity(
        self, Omega_0, Omega_1, conductivity_Omega_0, conductivity_Omega_1
    ):
        """
        Set up conductivity tensor with user-defined functions

        Parameters:
        -----------
        Omega_0 : function
            Function that takes coordinates x and returns boolean mask for subdomain 0
        Omega_1 : function
            Function that takes coordinates x and returns boolean mask for subdomain 1
        conductivity_Omega_0 : function
            Function that takes (x, y) coordinates and returns (k11, k12, k22) for subdomain 0
        conductivity_Omega_1 : function
            Function that takes (x, y) coordinates and returns (k11, k12, k22) for subdomain 1
        """
        # Create tensor function space
        V_tensor = functionspace(self.mesh, ("Lagrange", 1, (2, 2)))
        self.conductivity = Function(V_tensor)

        # Get dofmap and geometry
        dofmap = V_tensor.dofmap
        mesh_geometry = self.mesh.geometry.x
        cell_geometry = self.mesh.geometry.dofmap

        cells_0 = locate_entities(self.mesh, self.mesh.topology.dim, Omega_0)
        cells_1 = locate_entities(self.mesh, self.mesh.topology.dim, Omega_1)

        # Set values for Omega_0 cells
        for cell in cells_0:
            cell_dofs = dofmap.cell_dofs(cell)
            geometry_dofs = cell_geometry[cell]
            for i, dof in enumerate(cell_dofs):
                coord_idx = geometry_dofs[i // 4]
                x, y = mesh_geometry[coord_idx][0], mesh_geometry[coord_idx][1]
                k11, k12, k22 = conductivity_Omega_0(x, y)
                self.conductivity.x.array[dof * 4 + 0] = k11
                self.conductivity.x.array[dof * 4 + 1] = k12
                self.conductivity.x.array[dof * 4 + 2] = k12  # symmetric
                self.conductivity.x.array[dof * 4 + 3] = k22

        # Set values for Omega_1 cells
        for cell in cells_1:
            cell_dofs = dofmap.cell_dofs(cell)
            geometry_dofs = cell_geometry[cell]
            for i, dof in enumerate(cell_dofs):
                coord_idx = geometry_dofs[i // 4]
                x, y = mesh_geometry[coord_idx][0], mesh_geometry[coord_idx][1]
                k11, k12, k22 = conductivity_Omega_1(x, y)
                self.conductivity.x.array[dof * 4 + 0] = k11
                self.conductivity.x.array[dof * 4 + 1] = k12
                self.conductivity.x.array[dof * 4 + 2] = k12  # symmetric
                self.conductivity.x.array[dof * 4 + 3] = k22

    def set_constant_conductivity(self, kappa_0=1.0, kappa_1=0.1):
        """
        Convenience method for constant conductivity in each subdomain
        """

        def Omega_0(x):
            return np.sqrt(x[1] ** 2 + x[0] ** 2) <= 0.5

        def Omega_1(x):
            return np.sqrt(x[1] ** 2 + x[0] ** 2) > 0.5

        def conductivity_Omega_0(x, y):
            return kappa_0, 0.0, kappa_0  # k11, k12, k22

        def conductivity_Omega_1(x, y):
            return kappa_1, 0.0, kappa_1  # k11, k12, k22

        self.set_conductivity(
            Omega_0, Omega_1, conductivity_Omega_0, conductivity_Omega_1
        )

    def assemble_system(self, boundary_flux=None):
        """
        Assemble the linear system A*x = b
        """
        if self.conductivity is None:
            # Default to constant conductivity
            self.set_constant_conductivity()

        # Set up boundary flux
        x = ufl.SpatialCoordinate(self.mesh)
        if boundary_flux is None:
            theta = ufl.atan2(x[1], x[0])
            f = ufl.cos(theta)  # Default flux
        else:
            f = boundary_flux

        # Create mixed function space for real numbers
        R = create_real_functionspace(self.mesh)
        self.W = ufl.MixedFunctionSpace(self.V, R)
        u, lmbda = ufl.TrialFunctions(self.W)
        du, dl = ufl.TestFunctions(self.W)

        # Variational problem
        zero = Constant(self.mesh, dolfinx.default_scalar_type(0.0))

        a00 = ufl.inner(self.conductivity * ufl.grad(u), ufl.grad(du)) * ufl.dx
        a01 = ufl.inner(lmbda, du) * ufl.ds
        a10 = ufl.inner(u, dl) * ufl.ds
        L0 = ufl.inner(f, du) * ufl.ds
        L1 = ufl.inner(zero, dl) * ufl.ds

        # Assemble forms
        a = [[a00, a01], [a10, None]]
        L = [L0, L1]
        a_compiled = form(a)
        L_compiled = form(L)

        # Assemble system matrix
        try:
            self.A = dolfinx.fem.petsc.assemble_matrix_block(a_compiled)
        except AttributeError:
            self.A = dolfinx.fem.petsc.assemble_matrix(a_compiled)
        self.A.assemble()

        # Assemble right hand side
        try:
            self.b = dolfinx.fem.petsc.assemble_vector_block(
                L_compiled, a_compiled, bcs=[]
            )
        except AttributeError:
            self.b = dolfinx.fem.petsc.assemble_vector(L_compiled, kind="mpi")
            apply_lifting(self.b, [], a_compiled)
            set_bc(self.b, [])

        # Create solution vector
        self.xh = dolfinx.fem.petsc.create_vector_block(L_compiled)

        return self.A, self.b, self.xh

    def solve_system(self):
        """
        Solve the assembled linear system
        """
        if self.A is None or self.b is None or self.xh is None:
            raise ValueError("Must assemble system first using assemble_system()")

        # Create linear solver
        self.ksp = PETSc.KSP().create(self.mesh.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType("preonly")
        pc = self.ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")

        # Ghost update
        self.xh.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # Solve
        self.ksp.solve(self.b, self.xh)

        # Extract solution
        self._extract_solution()

        return self.uh

    def _extract_solution(self):
        """Extract solution from mixed space"""
        from dolfinx.cpp.la.petsc import scatter_local_vectors, get_local_vectors

        maps = [
            (Wi.dofmap.index_map, Wi.dofmap.index_map_bs)
            for Wi in self.W.ufl_sub_spaces()
        ]

        self.uh = Function(self.V, name="u")
        x_local = get_local_vectors(self.xh, maps)
        self.uh.x.array[: len(x_local[0])] = x_local[0]
        self.uh.x.scatter_forward()

    def solve(self, boundary_flux=None):
        """
        Convenience method that assembles and solves in one call
        """
        self.assemble_system(boundary_flux)
        return self.solve_system()

    def cleanup(self):
        """Clean up PETSc objects"""
        if self.b is not None:
            self.b.destroy()
        if self.xh is not None:
            self.xh.destroy()
        if self.A is not None:
            self.A.destroy()
        if self.ksp is not None:
            self.ksp.destroy()

    def compute_error(self, exact_solution=None):
        """Compute L2 error against exact solution"""
        if self.uh is None:
            raise ValueError("Must solve the problem first")

        if exact_solution is None:
            # Default exact solution: u(x,y) = x

            raise ValueError("Supply exact_solution function")

            exact_solution = Function(self.V)
            exact_solution.interpolate(lambda x: x[0])

        diff = self.uh - exact_solution
        error = ufl.inner(diff, diff) * ufl.dx
        return np.sqrt(assemble_scalar(error))

    def plot_solution(self):
        """Plot the solution using matplotlib"""
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri

        if self.uh is None:
            raise ValueError("No solution to plot")

        coords = self.mesh.geometry.x.copy()
        cells = self.mesh.topology.connectivity(
            self.mesh.topology.dim, 0
        ).array.reshape(-1, self.mesh.topology.dim + 1)

        x = coords[:, 0]
        y = coords[:, 1]
        u_vals = self.uh.x.array.real

        triang = tri.Triangulation(x, y, cells)

        plt.figure(figsize=(6, 5))
        cont = plt.tricontourf(triang, u_vals, levels=50, cmap="viridis")
        plt.colorbar(cont, label="u")
        plt.tricontour(triang, u_vals, levels=10, colors="k", linewidths=0.5)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("FEM solution u")
        plt.axis("equal")
        plt.show()

    def get_solution(self):
        """Return the solution function"""
        return self.uh

    def get_mesh(self):
        """Return the mesh"""
        return self.mesh

    def get_conductivity(self):
        """Return the conductivity function"""
        return self.conductivity

    def get_linear_system(self):
        """Return the linear system components"""
        return self.A, self.b, self.xh

    def get_solver_info(self):
        """Get solver information"""
        if self.ksp is None:
            return "Solver not run yet"

        return {
            "iterations": self.ksp.getIterationNumber(),
            "converged": self.ksp.getConvergedReason(),
            "residual": self.ksp.getResidualNorm(),
        }
