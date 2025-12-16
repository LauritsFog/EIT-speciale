import numpy as np
from mpi4py import MPI
from dolfinx.fem import (
    functionspace,
)
from dolfinx.io import gmshio
import gmsh

from sklearn.cluster import KMeans


def add_noise(ntd_matrix, noise_level):
    def H_minus_half_norm(E):
        E_mod = E.copy()

        for i, j in np.ndindex(E.shape):
            m = i - E.shape[0] // 2
            n = j - E.shape[1] // 2
            E_mod[i, j] = (
                E_mod[i, j]
                * (1 + abs(m) ** 2) ** (-0.25)
                * (1 + abs(n) ** 2) ** (-0.25)
            )  # (13.44) in inv prob book

        return np.linalg.norm(E_mod, ord=2)

    # Create noise matrix with N(0,1) entries
    noise_matrix = np.random.normal(loc=0.0, scale=1.0, size=ntd_matrix.shape)

    noise_matrix = 0.5 * (noise_matrix + noise_matrix.T)

    noise_scaling = noise_level / H_minus_half_norm(noise_matrix)

    ntd_matrix_noise = ntd_matrix + noise_scaling * noise_matrix

    return ntd_matrix_noise


# Possibly use instead of computing all eigenvalues. This is allegedly faster.
def is_pos_def(K):
    try:
        np.linalg.cholesky(K)
        return True
    except np.linalg.LinAlgError:
        return False


def setup_nonuniform_mesh(characteristic_length):
    """
    Create the unit disk mesh with non-uniform element size (smaller elements
    at the boundary, larger elements at the center) and return the mesh,
    cell_markers, and facet_markers.

    Args:
        characteristic_length (float): A base value used to define l_min.

    Returns:
        tuple: (mesh, cell_markers, facet_markers)
    """
    gmsh.initialize()

    # Define characteristic lengths
    l_min = characteristic_length * 1
    l_max = characteristic_length * 10.0

    # Create disk
    unitdisk = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()
    gdim = 2

    # Get the ID of the boundary curve (the edge of the disk)
    boundary_curves = gmsh.model.getBoundary([(2, unitdisk)], oriented=False)
    boundary_tags = [curve[1] for curve in boundary_curves]

    # Define a Distance Field (Field 1)
    # This field computes the distance to the boundary curves (tags)
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", boundary_tags)

    # Define a Threshold Field (Field 2)
    # This field defines the mesh size based on the distance to the boundary.
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)

    # lc_max is the size far away from the boundary (i.e., the center)
    gmsh.model.mesh.field.setNumber(2, "LcMax", l_max)

    # lc_min is the size close to the boundary
    gmsh.model.mesh.field.setNumber(2, "LcMin", l_min)

    # Distances to define the transition:
    # Size is l_min when distance is < d_min
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.05)
    # Size is l_max when distance is > d_max
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.5)

    # Set the field as the background mesh size field
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    # Create a physical group for the domain
    domain_tag = 1
    gmsh.model.addPhysicalGroup(gdim, [unitdisk], domain_tag)

    # Set global options and generate the mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", l_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", l_max)
    gmsh.model.mesh.generate(gdim)

    # Convert Gmsh model to DOLFINx mesh and markers
    gmsh_model_rank = 0

    mesh_comm = MPI.COMM_WORLD

    mesh, cell_markers, facet_markers = gmshio.model_to_mesh(
        gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim
    )

    gmsh.finalize()

    return mesh, cell_markers, facet_markers


def create_mesh_partitions(mesh, num_partitions):
    """Create compact, connected mesh partitions using k-means on element centroids."""
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, tdim - 1)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mesh.topology.create_connectivity(tdim, 0)

    num_cells = mesh.topology.index_map(tdim).size_local
    connectivity = mesh.topology.connectivity(tdim, 0)
    vertices = mesh.geometry.x

    # Compute centroids
    centroids = np.zeros((num_cells, 2))
    for c in range(num_cells):
        cell_vertices = connectivity.links(c)
        centroids[c] = vertices[cell_vertices, :2].mean(axis=0)

    # Run k-means
    kmeans = KMeans(n_clusters=num_partitions, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(centroids)

    # Build adjacency (cell neighbors)
    f_c = mesh.topology.connectivity(tdim - 1, tdim)
    neighbors = [[] for _ in range(num_cells)]
    for f in range(mesh.topology.index_map(tdim - 1).size_local):
        linked = f_c.links(f)
        if len(linked) == 2:
            a, b = linked
            neighbors[a].append(b)
            neighbors[b].append(a)

    # Split disconnected clusters
    visited = np.zeros(num_cells, dtype=bool)
    partitions = []
    for label in range(num_partitions):
        cluster_cells = np.where(labels == label)[0]
        cluster_set = set(cluster_cells)
        for c in cluster_cells:
            if visited[c]:
                continue
            q = [c]
            visited[c] = True
            comp = []
            while q:
                u = q.pop()
                comp.append(u)
                for nb in neighbors[u]:
                    if nb in cluster_set and not visited[nb]:
                        visited[nb] = True
                        q.append(nb)
            partitions.append(comp)

    print(
        f"Created {len(partitions)} connected partitions (requested {num_partitions})."
    )
    return partitions


def precompute_gradients(solutions, mesh, element_indices):
    """
    Precompute ∇u for the given subset of elements.

    Returns
    -------
    gradients : list[list[np.ndarray]]
        gradients[i][k] = grad(u_i) on element element_indices[k]
    elem_index_map : dict[int, int]
        Maps global element index -> local index into gradients[i]
    """
    tdim = mesh.topology.dim

    connectivity = mesh.topology.connectivity(tdim, 0)

    gradients = []

    for u_idx, ui in enumerate(solutions):
        V = ui.function_space
        dm = V.dofmap
        ui_grads = []

        print(f"Computing gradient of solution {u_idx} out of {len(solutions)}")

        for elem in element_indices:
            cell_dofs = dm.cell_dofs(elem)  # local dofs on this cell
            verts = mesh.geometry.x[connectivity.links(elem)]
            v0, v1, v2 = verts[:, :2]  # assumes triangles (P1)
            u_vals = ui.x.array[cell_dofs]  # values at dofs

            area = 0.5 * abs(
                (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
            )
            # Gradients of P1 basis functions on the triangle
            grad_phi0 = np.array([v1[1] - v2[1], v2[0] - v1[0]]) / (2 * area)
            grad_phi1 = np.array([v2[1] - v0[1], v0[0] - v2[0]]) / (2 * area)
            grad_phi2 = np.array([v0[1] - v1[1], v1[0] - v0[0]]) / (2 * area)

            grad_u = (
                u_vals[0] * grad_phi0 + u_vals[1] * grad_phi1 + u_vals[2] * grad_phi2
            )
            ui_grads.append(grad_u)

        gradients.append(ui_grads)

    return gradients


def find_inclusion_elements(
    mesh,
    solutions_A0,
    ntd_AD,
    ntd_A0,
    num_partitions=50,
    const=10,
    tol=1e-10,
):
    """
    Compute inclusion indices by processing mesh partitions sequentially.
    """
    tdim = mesh.topology.dim
    connectivity = mesh.topology.connectivity(tdim, 0)

    # Owned cells only
    num_elements = mesh.topology.index_map(tdim).size_local

    N = len(solutions_A0)
    inclusion_indices = []
    min_eig_indices = []
    min_eig_values = []

    gradients = precompute_gradients(solutions_A0, mesh, list(range(num_elements)))

    if num_partitions is None:
        # No partitioning: per-element test
        for elem in range(num_elements):
            if elem % 100 == 0:
                print(f"Processing element {elem}/{num_elements}")

            verts = mesh.geometry.x[connectivity.links(elem)][:, :2]
            v0, v1, v2 = verts
            area = 0.5 * abs(
                (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
            )

            Dfrechet = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    grad_dot = np.dot(gradients[i][elem], gradients[j][elem])
                    Dfrechet[i, j] = -const * grad_dot * area

            check = Dfrechet - ntd_AD + ntd_A0
            min_eig = np.min(np.real(np.linalg.eigvals(check)))

            min_eig_values.append(min_eig)
            min_eig_indices.append(elem)
            if min_eig + tol > 0:
                inclusion_indices.append(elem)

    else:
        partitions = create_mesh_partitions(mesh, num_partitions)

        # Partitioned: accumulate cell contributions inside Dfrechet, then one check per partition
        for p_idx, part in enumerate(partitions):
            if p_idx % 100 == 0:
                print(
                    f"Processing partition {p_idx + 1}/{len(partitions)} with {len(part)} elements"
                )

            Dfrechet = np.zeros((N, N))

            for elem in part:
                verts = mesh.geometry.x[connectivity.links(elem)][:, :2]
                v0, v1, v2 = verts
                area = 0.5 * abs(
                    (v1[0] - v0[0]) * (v2[1] - v0[1])
                    - (v2[0] - v0[0]) * (v1[1] - v0[1])
                )

                for i in range(N):
                    for j in range(i, N):
                        grad_dot = np.dot(gradients[i][elem], gradients[j][elem])
                        Dfrechet[i, j] -= const * grad_dot * area

                        if i != j:
                            Dfrechet[j, i] = Dfrechet[i, j]

            check = Dfrechet - ntd_AD + ntd_A0
            min_eig = np.min(np.real(np.linalg.eigvals(check)))
            is_inclusion = min_eig + tol > 0

            # Assign the same partition-level min_eig to every cell in the partition
            for elem in part:
                min_eig_values.append(min_eig)
                min_eig_indices.append(elem)
                if is_inclusion:
                    inclusion_indices.append(elem)

    min_eig_values = np.asarray(min_eig_values, dtype=float)
    eigenvalue_dict = dict(zip(min_eig_indices, min_eig_values))

    print(
        f"Found {len(inclusion_indices)} potential inclusion elements across all partitions"
    )
    if num_partitions is None:
        return inclusion_indices, eigenvalue_dict, None
    else:
        return inclusion_indices, eigenvalue_dict, partitions
