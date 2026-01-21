import numpy as np
from scipy.spatial.distance import directed_hausdorff


def add_noise(ntd_matrix, noise_level, seed=None):
    np.random.seed(seed)

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

    # noise_scaling = noise_level / H_minus_half_norm(noise_matrix)
    noise_scaling = noise_level / np.linalg.norm(noise_matrix, ord=2)

    ntd_matrix_noise = ntd_matrix + noise_scaling * noise_matrix

    return ntd_matrix_noise


def precompute_gradients(solutions, mesh):
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

    num_cells = mesh.topology.index_map(tdim).size_local
    element_indices = range(num_cells)

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


def compute_reconstruction_test_values(
    mesh,
    gradients,
    ntd_AD,
    ntd_A0,
    rho,
):
    """
    Compute inclusion indices by processing mesh partitions sequentially.
    """
    tdim = mesh.topology.dim
    connectivity = mesh.topology.connectivity(tdim, 0)

    # Owned cells only
    num_elements = mesh.topology.index_map(tdim).size_local

    N = np.shape(ntd_A0)[0]
    element_indices = []
    min_eig_values = []
    frechet_quantity_values = []

    if N > 16:
        frechet_mode_sample_idx = 8
    else:
        frechet_mode_sample_idx = 0

    print(f"Sampling Frechet quantity at mode index {frechet_mode_sample_idx}")

    for elem in range(num_elements):
        if elem % 500 == 0:
            print(f"Processing element {elem}/{num_elements}")

        verts = mesh.geometry.x[connectivity.links(elem)][:, :2]
        v0, v1, v2 = verts
        area = 0.5 * abs(
            (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
        )

        Dfrechet = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                grad_dot = np.dot(gradients[i][elem], gradients[j][elem])
                Dfrechet[i, j] -= grad_dot * area

                if i != j:
                    Dfrechet[j, i] = Dfrechet[i, j]

        check = ntd_A0 + rho * Dfrechet - ntd_AD
        check = (check + check.T) / 2  # ensure symmetry
        min_eig = np.min(np.real(np.linalg.eigvals(check)))

        idx = round(N / 2 + frechet_mode_sample_idx)
        frechet_quantity = np.abs(Dfrechet[idx, idx])

        min_eig_values.append(min_eig)
        frechet_quantity_values.append(frechet_quantity)
        element_indices.append(elem)

    min_eig_values = np.asarray(min_eig_values, dtype=float)
    frechet_quantity_values = np.asarray(frechet_quantity_values, dtype=float)

    eigenvalue_dict = dict(zip(element_indices, min_eig_values))
    frechet_quantity_dict = dict(zip(element_indices, frechet_quantity_values))

    return eigenvalue_dict, frechet_quantity_dict


def reconstruct_inclusions(eigenvalue_dict, alpha_reg):
    inclusion_indices = []

    for elem_idx, min_eig in eigenvalue_dict.items():
        if min_eig + alpha_reg > 0:
            inclusion_indices.append(elem_idx)

    return inclusion_indices


def compute_hausdorff_distance(mesh, indices_A, indices_B):
    if len(indices_A) == 0 or len(indices_B) == 0:
        return np.inf

    tdim = mesh.topology.dim
    connectivity = mesh.topology.connectivity(tdim, 0)

    def get_vertices(indices):
        all_vertex_indices = []
        for i in indices:
            all_vertex_indices.extend(connectivity.links(i))
        unique_indices = np.unique(all_vertex_indices)
        return mesh.geometry.x[unique_indices][:, :tdim]

    points_A = get_vertices(indices_A)
    points_B = get_vertices(indices_B)

    d_A_B = directed_hausdorff(points_A, points_B)[0]
    d_B_A = directed_hausdorff(points_B, points_A)[0]

    return max(d_A_B, d_B_A)


def compute_classification_error(mesh, target_indeces, recon_inclusion):
    tdim = mesh.topology.dim

    # Owned cells only
    num_elements = mesh.topology.index_map(tdim).size_local

    target_set = set(target_indeces)
    reconstructed_set = set(recon_inclusion)

    false_negatives = 1.7 * len(target_set - reconstructed_set)
    false_positives = len(reconstructed_set - target_set)

    classification_error = false_negatives + false_positives

    return classification_error
