import numpy as np
import ufl
import dolfinx
from dolfinx.fem import form
from dolfinx.fem import assemble_scalar
from dolfinx.mesh import meshtags


def find_inclusion_elements(
    mesh, solutions_A0, ntd_AD, ntd_A0, sampling_stride=10, const=10
):
    """
    Find potential inclusion elements by checking the Frechet derivative condition.

    Parameters:
    -----------
    mesh : dolfinx.mesh.Mesh
        The mesh to search for inclusions
    solutions_A0 : list
        List of basis solutions from reference solver
    ntd_AD : numpy.ndarray
        NtD matrix for the perturbed case
    ntd_A0 : numpy.ndarray
        NtD matrix for the reference case
    sampling_stride : int, optional
        Stride for sampling elements (default: 10)
    const : float, optional
        Constant multiplier for the integrand (default: 10)

    Returns:
    --------
    inclusion_indexes : list
        List of element indices that satisfy the inclusion condition
    """
    N = len(solutions_A0)

    # Simple sampling approach - use this instead
    num_elements = mesh.topology.index_map(mesh.topology.dim).size_local
    sampled_elements = range(0, num_elements, sampling_stride)

    print(f"Sampling {len(sampled_elements)} elements out of {num_elements} total")

    inclusion_indexes = []
    for idx, elem in enumerate(sampled_elements):
        if idx % 10 == 0:
            print(f"Processing element {idx}/{len(sampled_elements)}")

        ct = dolfinx.mesh.meshtags(
            mesh,
            mesh.topology.dim,
            np.array([elem], dtype=np.int32),
            np.array([1], dtype=np.int32),
        )
        dx_single = ufl.Measure("dx", domain=mesh, subdomain_data=ct)

        Dfrechet = np.empty((N, N))
        for i in range(N):
            for j in range(N):
                ui = solutions_A0[i]
                uj = solutions_A0[j]
                integrand = -const * ufl.dot(ufl.grad(ui), ufl.grad(uj))
                Dfrechet[i, j] = dolfinx.fem.assemble_scalar(
                    dolfinx.fem.form(integrand * dx_single(1))
                )

        check = Dfrechet - ntd_AD + ntd_A0
        if min(np.linalg.eigvals(check)) > 0:
            inclusion_indexes.append(elem)

    print(f"Found {len(inclusion_indexes)} potential inclusion elements")
    return inclusion_indexes


def find_inclusion_elements_manual(
    mesh, solutions_A0, ntd_AD, ntd_A0, sampling_stride=10, const=10
):
    """
    Optimized version - uses constant gradient property of P1 elements
    """
    N = len(solutions_A0)

    # Precompute all gradients at the element level
    print("Precomputing gradients for all basis solutions...")
    gradients = []  # gradients[i][element] = grad(ui) on that element

    for ui in solutions_A0:
        # Get the function space and dofmap
        V = ui.function_space
        dm = V.dofmap

        # Compute gradient for each element
        ui_grads = []
        for elem in range(mesh.topology.index_map(mesh.topology.dim).size_local):
            # Get cell dofs and vertex coordinates
            cell_dofs = dm.cell_dofs(elem)
            cell_vertices = mesh.geometry.x[
                mesh.topology.connectivity(mesh.topology.dim, 0).links(elem)
            ]

            # Get function values at vertices
            u_vals = ui.x.array[cell_dofs]

            # Extract only x,y coordinates (ignore z if present)
            # cell_vertices might be (x,y,z) even for 2D mesh
            v0 = cell_vertices[0][:2]  # Take first two coordinates
            v1 = cell_vertices[1][:2]
            v2 = cell_vertices[2][:2]

            x0, y0 = v0
            x1, y1 = v1
            x2, y2 = v2

            # Area of triangle
            area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

            # Shape function gradients (constant for P1)
            grad_phi0 = np.array([y1 - y2, x2 - x1]) / (2 * area)
            grad_phi1 = np.array([y2 - y0, x0 - x2]) / (2 * area)
            grad_phi2 = np.array([y0 - y1, x1 - x0]) / (2 * area)

            # Gradient of u on this element
            grad_u = (
                u_vals[0] * grad_phi0 + u_vals[1] * grad_phi1 + u_vals[2] * grad_phi2
            )
            ui_grads.append(grad_u)

        gradients.append(ui_grads)

    print("Gradients precomputed!")

    # Simple sampling approach
    num_elements = mesh.topology.index_map(mesh.topology.dim).size_local
    sampled_elements = range(0, num_elements, sampling_stride)

    print(f"Sampling {len(sampled_elements)} elements out of {num_elements} total")

    inclusion_indexes = []

    inclusion_test_matrix_eigenvalues = []
    eigenvalue_indeces = []

    for idx, elem in enumerate(sampled_elements):
        if idx % 100 == 0:  # Can print less frequently now since it's faster
            print(f"Processing element {idx}/{len(sampled_elements)}")

        # Get area of current element
        cell_vertices = mesh.geometry.x[
            mesh.topology.connectivity(mesh.topology.dim, 0).links(elem)
        ]

        # Extract only x,y coordinates
        v0 = cell_vertices[0][:2]
        v1 = cell_vertices[1][:2]
        v2 = cell_vertices[2][:2]

        x0, y0 = v0
        x1, y1 = v1
        x2, y2 = v2

        area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

        # Compute Dfrechet matrix directly
        Dfrechet = np.empty((N, N))
        for i in range(N):
            for j in range(N):
                # grad(ui) · grad(uj) is constant over the element
                grad_dot = np.dot(gradients[i][elem], gradients[j][elem])
                # Integral over element = (grad_dot) * area
                Dfrechet[i, j] = -const * grad_dot * area

        check = Dfrechet - ntd_AD + ntd_A0

        eigenvalue_indeces.append(elem)
        eigenvalues = np.real(np.linalg.eigvals(check))
        # min_eigenvalue = min(eigenvalues)
        min_eigenvalue = np.tan(min(eigenvalues) + np.pi / 2)
        inclusion_test_matrix_eigenvalues.append(min_eigenvalue)

        tol = 1e-8
        if min(eigenvalues) > tol:
            inclusion_indexes.append(elem)

    eigenvalue_dict = dict(zip(eigenvalue_indeces, inclusion_test_matrix_eigenvalues))

    print(f"Found {len(inclusion_indexes)} potential inclusion elements")
    return (inclusion_indexes, eigenvalue_dict)
