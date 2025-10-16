import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx.fem.petsc
from dolfinx.fem import (
    Function,
    Expression,
    Constant,
    functionspace,
    assemble_scalar,
    form,
    locate_dofs_geometrical,
)
from dolfinx.mesh import locate_entities
import ufl
from dolfinx.io import gmshio
import gmsh
from scifem import create_real_functionspace
from dolfinx.fem.petsc import apply_lifting, set_bc
from dolfinx.mesh import locate_entities_boundary
import matplotlib.pyplot as plt
import matplotlib.tri as tri


class FemConductivitySolver:
    def __init__(self, mesh_characteristic_length=0.05, mesh=None):
        self.conductivity = None
        self.uh = None
        self.A = None
        self.b = None
        self.xh = None
        self.ksp = None
        self.W = None
        self.boundary_vertices = None  # Initialize
        self.boundary_coords = None  # Initialize
        self.ntd_matrix = None
        self.basis_frequencies = None
        self.basis_solutions = None
        if mesh is None:
            self.mesh_characteristic_length = mesh_characteristic_length
            self.mesh = None
            self.V = None
            self.setup_mesh()
        else:
            self.mesh = mesh
            self.V = functionspace(mesh, ("Lagrange", 1))
            self.setup_boundary_data()  # Setup boundary data for existing mesh

    def setup_mesh(self):
        """Create the unit disk mesh with uniformly spaced boundary vertices"""
        gmsh.initialize()

        # Create disk
        unitdisk = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()

        gdim = 2

        # Get boundary curves
        boundary_curves = gmsh.model.getBoundary([(2, unitdisk)], oriented=False)

        # Set mesh size specifically on boundary curves
        target_boundary_spacing = (
            2 * np.pi * self.mesh_characteristic_length
        )  # Adjust as needed

        for curve in boundary_curves:
            # Set finer mesh size on boundary curves
            gmsh.model.mesh.setSize(gmsh.model.getEntities(1), target_boundary_spacing)

        gmsh.model.addPhysicalGroup(gdim, [unitdisk], 1)

        # Set general mesh size (interior)
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMin", self.mesh_characteristic_length
        )
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMax", self.mesh_characteristic_length
        )

        # Enable mesh size from points for better boundary distribution
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

        gmsh.model.mesh.generate(gdim)

        gmsh_model_rank = 0
        mesh_comm = MPI.COMM_WORLD

        self.mesh, cell_markers, facet_markers = gmshio.model_to_mesh(
            gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim
        )

        gmsh.finalize()

        # Create function space
        self.V = functionspace(self.mesh, ("Lagrange", 1))
        self.setup_boundary_data()

    def setup_boundary_data(self):
        """Find and save boundary vertices and coordinates"""
        coords_all = self.mesh.geometry.x

        # Find boundary vertices (distance from origin ~ R)
        tol = 1e-12
        self.boundary_vertices = np.where(
            np.abs(np.linalg.norm(coords_all, axis=1) - 1.0) < tol
        )[0]

        # Coordinates of boundary vertices
        self.boundary_coords = coords_all[self.boundary_vertices]

    def set_conductivity(self, conductivity_func):
        """
        Set up conductivity tensor with a user-defined function

        Parameters:
        -----------
        conductivity_func : function
            Function that takes (x, y) coordinates and returns (a11, a12, a22)
            for the entire domain
        """
        # Create tensor function space
        V_tensor = functionspace(self.mesh, ("Lagrange", 1, (2, 2)))
        self.conductivity = Function(V_tensor)

        # Get mesh coordinates
        mesh_geometry = self.mesh.geometry.x

        # Get the dofmap for the tensor function space
        dofmap = V_tensor.dofmap

        # Create a set to track which vertices we've already processed
        processed_vertices = set()

        # Iterate over all cells in the mesh
        for cell in range(
            self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        ):
            cell_dofs = dofmap.cell_dofs(cell)

            # Process each DOF in this cell
            for i, dof in enumerate(cell_dofs):
                # Skip if we've already processed this DOF
                if dof in processed_vertices:
                    continue

                # Mark this DOF as processed
                processed_vertices.add(dof)

                # Get the coordinate for this DOF
                # For Lagrange elements, each DOF corresponds to a mesh vertex
                # We need to find which mesh vertex this DOF corresponds to
                # coord_idx = dof
                x, y = mesh_geometry[dof][0], mesh_geometry[dof][1]

                a11, a12, a22 = conductivity_func(x, y)
                self.conductivity.x.array[dof * 4 + 0] = a11
                self.conductivity.x.array[dof * 4 + 1] = a12
                self.conductivity.x.array[dof * 4 + 2] = a12  # symmetric
                self.conductivity.x.array[dof * 4 + 3] = a22

        # Ensure proper parallel communication
        self.conductivity.x.scatter_forward()

    def assemble_system(self, neumann_cond):
        """
        Assemble the linear system A*x = b
        """
        if self.conductivity is None:
            # Default to constant conductivity
            self.set_constant_conductivity()

        # Store the boundary flux for later use
        self.neumann_cond = neumann_cond

        # Set up boundary flux
        f = neumann_cond

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
        self.extract_solution()

        return self.uh

    def extract_solution(self):
        """Extract solution from mixed space"""
        from dolfinx.cpp.la.petsc import get_local_vectors

        maps = [
            (Wi.dofmap.index_map, Wi.dofmap.index_map_bs)
            for Wi in self.W.ufl_sub_spaces()
        ]

        self.uh = Function(self.V, name="u")
        x_local = get_local_vectors(self.xh, maps)
        self.uh.x.array[: len(x_local[0])] = x_local[0]
        self.uh.x.scatter_forward()

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

    def compute_ntd_map(self, max_freq=None):
        """Compute Neumann-to-Dirichlet map with ordering [a_N,...,a₁, b₁,...,b_N]"""

        N = len(self.boundary_coords)

        if max_freq is None:
            max_freq = N // 2

        # Column ordering: [cos(Nθ), cos((N-1)θ), ..., cos(θ), sin(θ), sin(2θ), ..., sin(Nθ)]
        n_values = np.concatenate(
            [
                np.arange(max_freq, 0, -1),  # Cosine modes: N, N-1, ..., 1
                np.arange(1, max_freq + 1),  # Sine modes: 1, 2, ..., N
            ]
        )
        num_modes = len(n_values) // 2
        self.basis_frequencies = n_values  # Store frequencies
        self.basis_solutions = []

        ntd_matrix = np.zeros((2 * num_modes, 2 * num_modes), dtype=float)

        print(f"Computing NtD map: {2 * num_modes} × {2 * num_modes} real matrix")

        # Cosine input modes (first N columns) in reverse order
        for j in range(num_modes):
            n = max_freq - j  # n = N, N-1, ..., 1
            print(f"  Column {j + 1}/{2 * num_modes}: Input cos({n}θ)")

            x = ufl.SpatialCoordinate(self.mesh)
            theta = ufl.atan2(x[1], x[0])
            nc_mode = ufl.cos(n * theta)

            self.assemble_system(neumann_cond=nc_mode)
            self.solve_system()
            self.basis_solutions.append(self.uh.copy())  # Store copy of solution

            coefficients, _ = self.get_boundary_solution_fourier_coefficients(max_freq)
            ntd_matrix[:, j] = coefficients

        # Sine input modes (next N columns) in forward order
        for j in range(num_modes):
            n = j + 1  # n = 1, 2, ..., N
            col_idx = num_modes + j
            print(f"  Column {col_idx + 1}/{2 * num_modes}: Input sin({n}θ)")

            x = ufl.SpatialCoordinate(self.mesh)
            theta = ufl.atan2(x[1], x[0])
            nc_mode = ufl.sin(n * theta)

            self.assemble_system(neumann_cond=nc_mode)
            self.solve_system()
            self.basis_solutions.append(self.uh.copy())  # Store copy of solution

            coefficients, _ = self.get_boundary_solution_fourier_coefficients(max_freq)
            ntd_matrix[:, col_idx] = coefficients

        self.ntd_matrix = np.real(ntd_matrix)
        return ntd_matrix, n_values

    def plot_solution(self, title="FEM solution u"):
        if self.uh is None:
            raise ValueError("No solution to plot")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(6, 5))

        coords = self.mesh.geometry.x.copy()
        cells = self.mesh.topology.connectivity(
            self.mesh.topology.dim, 0
        ).array.reshape(-1, self.mesh.topology.dim + 1)

        x = coords[:, 0]
        y = coords[:, 1]
        u_vals = self.uh.x.array.real

        triang = tri.Triangulation(x, y, cells)

        # Create the plot
        cont = ax.tricontourf(triang, u_vals, levels=50, cmap="viridis")
        ax.tricontour(triang, u_vals, levels=10, colors="k", linewidths=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.axis("equal")

        # Add colorbar
        fig.colorbar(cont, ax=ax, label="u")

        # Return the plot objects WITHOUT showing
        return fig, ax

    def plot_boundary_solution_with_neumann_cond(
        self, title="Boundary solution and Neumann condition"
    ):
        """Plot the boundary solution and Neumann flux together"""

        if self.uh is None:
            raise ValueError("No solution to plot")

        if not hasattr(self, "neumann_cond") or self.neumann_cond is None:
            raise ValueError("No boundary flux stored. Call assemble_system() first.")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 5))

        # Get sorted boundary solutoin
        theta_sorted, u_sorted, coords_sorted = self.get_ordered_boundary_solution()

        # Get sorted boundary flux
        theta_flux, flux_sorted, coords_flux = self.get_ordered_neumann_cond()

        # Create the plot
        ax.plot(
            theta_sorted, u_sorted, "b-", linewidth=2, label="Boundary solution (u)"
        )
        ax.plot(
            theta_flux,
            flux_sorted,
            "r--",
            linewidth=2,
            label="Neumann condition (∂u/∂n)",
        )
        ax.set_xlabel("Angle (radians)")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

        # Return the plot objects WITHOUT showing
        return fig, ax

    def plot_fourier_coefficients(self, max_freq=None):
        """Plot the Fourier coefficients in the desired order"""

        c_n, n_values = self.get_boundary_solution_fourier_coefficients()

        if max_freq is None:
            max_freq = len(n_values) // 2

        # Filter to show only up to max_freq
        mask = np.abs(n_values) <= max_freq
        c_filtered = c_n[mask]

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        # Plot magnitude
        ax.stem(np.abs(c_filtered), basefmt=" ")
        ax.set_xlabel("Frequency (n)")
        ax.set_ylabel("|c_n|")
        ax.set_title("Fourier Coefficients Magnitude (basis: sine, cosine)")
        ax.grid(True)

        return fig, ax

    def plot_conductivity_components(self, title="Conductivity Tensor Components"):
        """
        Plot the three components of the anisotropic conductivity tensor.
        For conductivity matrix [a b; b c], plots a, b, and c separately.
        """
        if self.conductivity is None:
            raise ValueError("Conductivity not set. Call set_conductivity() first.")

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Get mesh coordinates and connectivity
        coords = self.mesh.geometry.x.copy()
        cells = self.mesh.topology.connectivity(
            self.mesh.topology.dim, 0
        ).array.reshape(-1, self.mesh.topology.dim + 1)

        x = coords[:, 0]
        y = coords[:, 1]

        # Extract conductivity components at vertices
        conductivity_array = self.conductivity.x.array.reshape(-1, 4)

        # For each vertex, extract the components [a, b, b, c]
        a_vals = conductivity_array[:, 0]  # a11 component
        b_vals = conductivity_array[:, 1]  # a12 component (off-diagonal)
        c_vals = conductivity_array[:, 3]  # a22 component

        # Calculate collective min and max across all components
        all_vals = np.concatenate([a_vals, b_vals, c_vals])
        vmin = np.min(all_vals)
        vmax = np.max(all_vals)

        levels = np.linspace(vmin, vmax + 1e-10, 100)

        # Create triangulation
        triang = tri.Triangulation(x, y, cells)

        # Plot a11 component with shared color limits
        cont1 = axes[0].tricontourf(
            triang, a_vals, levels=levels, cmap="viridis", vmin=vmin, vmax=vmax
        )
        # Add black triangle edges
        axes[0].triplot(triang, color="black", linewidth=0.5, alpha=0.5)
        axes[0].set_title(r"$k_{11}$ Component")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].axis("equal")
        plt.colorbar(cont1, ax=axes[0], label=r"$k_{11}$")

        # Plot a12 component with shared color limits
        cont2 = axes[1].tricontourf(
            triang, b_vals, levels=levels, cmap="viridis", vmin=vmin, vmax=vmax
        )
        # Add black triangle edges
        axes[1].triplot(triang, color="black", linewidth=0.5, alpha=0.5)
        axes[1].set_title(r"$k_{12}$ Component")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].axis("equal")
        plt.colorbar(cont2, ax=axes[1], label=r"$k_{12}$")

        # Plot a22 component with shared color limits
        cont3 = axes[2].tricontourf(
            triang, c_vals, levels=levels, cmap="viridis", vmin=vmin, vmax=vmax
        )
        # Add black triangle edges
        axes[2].triplot(triang, color="black", linewidth=0.5, alpha=0.5)
        axes[2].set_title(r"$k_{22}$ Component")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].axis("equal")
        plt.colorbar(cont3, ax=axes[2], label=r"$k_{22}$")

        fig.suptitle(title, fontsize=16)
        plt.tight_layout()

        return fig, axes

    def get_ordered_boundary_solution(self):
        """Extract and sort boundary solution by angle"""
        if self.uh is None:
            raise ValueError("No solution available")

        if self.boundary_vertices is None or self.boundary_coords is None:
            raise ValueError(
                "Boundary data not initialized. Call setup_boundary_data() first."
            )

        # FEM solution at boundary vertices
        u_boundary = self.uh.x.array[self.boundary_vertices]

        # Compute angles for sorting
        theta = np.arctan2(self.boundary_coords[:, 1], self.boundary_coords[:, 0])
        # Shift angles so they are in [0, 2π)
        theta = (theta + 2 * np.pi) % (2 * np.pi)

        sort_idx = np.argsort(theta)
        theta_sorted = theta[sort_idx]

        # print("Difference in angles (radians):")
        # print(np.diff(theta_sorted))

        u_sorted = u_boundary[sort_idx]
        coords_sorted = self.boundary_coords[sort_idx]

        return theta_sorted, u_sorted, coords_sorted

    def get_ordered_neumann_cond(self):
        """Get the Neumann condition values on the boundary by evaluating the stored UFL expression"""

        if not hasattr(self, "neumann_cond") or self.neumann_cond is None:
            raise ValueError("No boundary flux stored. Call assemble_system() first.")

        flux_function = Function(self.V)

        # Create an Expression from the stored neumann_cond and interpolate it
        flux_expression = Expression(
            self.neumann_cond, self.V.element.interpolation_points()
        )
        flux_function.interpolate(flux_expression)

        # Get the flux values at boundary vertices
        flux_values = flux_function.x.array[self.boundary_vertices]

        # Sort by angle
        theta = np.arctan2(self.boundary_coords[:, 1], self.boundary_coords[:, 0])
        theta = (theta + 2 * np.pi) % (2 * np.pi)
        sort_idx = np.argsort(theta)
        theta_sorted = theta[sort_idx]
        flux_sorted = flux_values[sort_idx]
        coords_sorted = self.boundary_coords[sort_idx]

        return theta_sorted, flux_sorted, coords_sorted

    def get_boundary_solution_fourier_coefficients(self, max_freq=None):
        """Compute real Fourier coefficients using FFT in ordering [a_N,...,a₁, b₁,...,b_N]

        Returns:
        --------
        coefficients : numpy.ndarray
            Real Fourier coefficients ordered as:
            [a_N, a_{N-1}, ..., a₁, b₁, b₂, ..., b_N] where:
            - a_n are cosine coefficients for cos(nθ) (in reverse order)
            - b_n are sine coefficients for sin(nθ) (in forward order)
        n_values : numpy.ndarray
            The n values [N, N-1, ..., 1, 1, 2, ..., N]
        """
        import numpy as np

        theta_sorted, u_sorted, coords_sorted = self.get_ordered_boundary_solution()
        N = len(u_sorted)

        if max_freq is None:
            max_freq = N // 2

        # Compute FFT
        u_hat = np.fft.fft(u_sorted) / N

        # Extract real Fourier coefficients from FFT
        a_k = np.zeros(max_freq + 1)  # a₀, a₁, ..., a_max_freq
        b_k = np.zeros(max_freq + 1)  # b₀, b₁, ..., b_max_freq

        # DC component
        a_k[0] = np.real(u_hat[0])

        # Positive frequencies (n > 0)
        for n in range(1, max_freq + 1):
            a_k[n] = 2 * np.real(u_hat[n])
            b_k[n] = -2 * np.imag(u_hat[n])

        # Create ordering: [a_N, a_{N-1}, ..., a₁, b₁, b₂, ..., b_N]
        coefficients = np.concatenate([a_k[max_freq:0:-1], b_k[1:]])
        n_values = np.concatenate(
            [np.arange(max_freq, 0, -1), np.arange(1, max_freq + 1)]
        )

        return coefficients, n_values

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
