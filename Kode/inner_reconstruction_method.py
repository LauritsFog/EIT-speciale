import numpy as np

# import pymetis
from sklearn.cluster import KMeans


# Possibly use instead of computing all eigenvalues. This is allegedly faster.
def is_pos_def(K):
    try:
        np.linalg.cholesky(K)
        return True
    except np.linalg.LinAlgError:
        return False


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


# def create_mesh_partitions(mesh, num_partitions, include_ghosts=False):
#     tdim = mesh.topology.dim
#     mesh.topology.create_connectivity(tdim, tdim - 1)
#     mesh.topology.create_connectivity(tdim - 1, tdim)

#     cell_imap = mesh.topology.index_map(tdim)
#     n_local = cell_imap.size_local
#     n_ghost = cell_imap.num_ghosts

#     if include_ghosts:
#         local_cells = np.arange(n_local + n_ghost, dtype=np.int32)
#     else:
#         local_cells = np.arange(n_local, dtype=np.int32)

#     # Build adjacency list for METIS
#     c_f = mesh.topology.connectivity(tdim, tdim - 1)
#     f_c = mesh.topology.connectivity(tdim - 1, tdim)
#     adj = []

#     for c in local_cells:
#         nbrs = []
#         for f in c_f.links(c):
#             for nb in f_c.links(f):
#                 if nb == c:
#                     continue
#                 if not include_ghosts and nb >= n_local:
#                     continue
#                 nbrs.append(int(nb))
#         # METIS expects neighbors as a Python list (duplicates ok)
#         adj.append(nbrs)

#     # Partition
#     _, membership = pymetis.part_graph(num_partitions, adjacency=adj)

#     # Group cells
#     parts = [[] for _ in range(num_partitions)]
#     for lc, part_id in zip(local_cells, membership):
#         if lc < n_local:  # keep owned only
#             parts[part_id].append(int(lc))

#     # (METIS sometimes assigns two distant islands to the same part)
#     def split_connected(component_cells):
#         # BFS on the induced subgraph
#         cell_set = set(component_cells)
#         visited = set()
#         comps = []
#         for c in component_cells:
#             if c in visited:
#                 continue
#             q = [c]
#             visited.add(c)
#             cur = []
#             while q:
#                 u = q.pop()
#                 cur.append(u)
#                 for v in adj[u]:
#                     if v in cell_set and v not in visited:
#                         visited.add(v)
#                         q.append(v)
#             comps.append(cur)
#         return comps

#     refined = []
#     for p in parts:
#         if not p:
#             continue
#         comps = split_connected(p)
#         refined.extend(comps)

#     # If splitting made more than num_partitions partitions, you can merge tiny components
#     # into neighboring larger ones as a post-process. Often not necessary.

#     # Sort indices inside each partition
#     # refined = [sorted(comp) for comp in refined if len(comp) > 0]
#     return refined


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
                    for j in range(N):
                        grad_dot = np.dot(gradients[i][elem], gradients[j][elem])
                        Dfrechet[i, j] -= const * grad_dot * area

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


# def find_inclusion_elements(
#     mesh, solutions_A0, ntd_AD, ntd_A0, sampling_stride=10, const=10
# ):
#     """
#     Find potential inclusion elements by checking the Frechet derivative condition.

#     Parameters:
#     -----------
#     mesh : dolfinx.mesh.Mesh
#         The mesh to search for inclusions
#     solutions_A0 : list
#         List of basis solutions from reference solver
#     ntd_AD : numpy.ndarray
#         NtD matrix for the perturbed case
#     ntd_A0 : numpy.ndarray
#         NtD matrix for the reference case
#     sampling_stride : int, optional
#         Stride for sampling elements (default: 10)
#     const : float, optional
#         Constant multiplier for the integrand (default: 10)

#     Returns:
#     --------
#     inclusion_indexes : list
#         List of element indices that satisfy the inclusion condition
#     """
#     N = len(solutions_A0)

#     # Simple sampling approach - use this instead
#     num_elements = mesh.topology.index_map(mesh.topology.dim).size_local
#     sampled_elements = range(0, num_elements, sampling_stride)

#     print(f"Sampling {len(sampled_elements)} elements out of {num_elements} total")

#     inclusion_indexes = []
#     for idx, elem in enumerate(sampled_elements):
#         if idx % 10 == 0:
#             print(f"Processing element {idx}/{len(sampled_elements)}")

#         ct = dolfinx.mesh.meshtags(
#             mesh,
#             mesh.topology.dim,
#             np.array([elem], dtype=np.int32),
#             np.array([1], dtype=np.int32),
#         )
#         dx_single = ufl.Measure("dx", domain=mesh, subdomain_data=ct)

#         Dfrechet = np.empty((N, N))
#         for i in range(N):
#             for j in range(N):
#                 ui = solutions_A0[i]
#                 uj = solutions_A0[j]
#                 integrand = -const * ufl.dot(ufl.grad(ui), ufl.grad(uj))
#                 Dfrechet[i, j] = dolfinx.fem.assemble_scalar(
#                     dolfinx.fem.form(integrand * dx_single(1))
#                 )

#         check = Dfrechet - ntd_AD + ntd_A0
#         if min(np.linalg.eigvals(check)) > 0:
#             inclusion_indexes.append(elem)

#     print(f"Found {len(inclusion_indexes)} potential inclusion elements")
#     return inclusion_indexes
