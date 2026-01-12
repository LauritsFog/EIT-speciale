# %% Imports and initialization
from fem_solver import (FemConductivitySolver,
    interpolate_solutions_to_new_mesh)
import matplotlib.pyplot as plt
import ufl
from inner_reconstruction_method import find_inclusion_elements
from plotting_functions import *
from conductivity_functions import *

#%%
characteristic_length = 0.01
num_partitions = None
max_freq = 15

shape_choice = "ellipse"
params = [[0.4,0],[0.4,0.4]]#[[0,0],[0.4,0.4]]

# Define conductivities
my_conductivity_A0 = lambda x, y: (1, 0, 1)
inclusion_conductivity = lambda x, y: (5, 0, 2)

def my_conductivity_AD(x, y):
    return conductivity_AD(
        x,
        y,
        my_conductivity_A0,
        inclusion_conductivity,
        shape_choice,
        params
    )

# Visualization
fig, axes = plot_conductivity_components(
    my_conductivity_AD, title=f"{shape_choice} Conductivity Distribution"
)

solverA0 = FemConductivitySolver(mesh_characteristic_length=characteristic_length)
solverA0.set_conductivity(my_conductivity_A0, my_conductivity_A0)

mesh = solverA0.mesh

# %%
solverAD = FemConductivitySolver(mesh=mesh)
c, alpha, beta = solverAD.set_conductivity(my_conductivity_AD, my_conductivity_A0)

solverA0.compute_ntd_map(max_freq=max_freq)
solverAD.compute_ntd_map(max_freq=max_freq)


ntd_A0 = solverA0.ntd_matrix
ntd_AD = solverAD.ntd_matrix
# %% Reconstruction Mesh
recon_characteristic_length = 0.07
recon_solver = FemConductivitySolver(mesh_characteristic_length=recon_characteristic_length)
recon_mesh = recon_solver.mesh

sols = interpolate_solutions_to_new_mesh(
    solverA0,
    recon_solver)

recon_solver.set_conductivity(my_conductivity_AD, my_conductivity_A0)

const = c * (alpha**2) / (beta**2) 

# %%
mu = 1.00
tol = 0#-mu*np.min(np.real(np.linalg.eigvals(((ntd_A0-ntd_AD)+(ntd_A0-ntd_AD).T)/2)))

inclusion_indexes, test_matrix_eigenvalues, partitions = find_inclusion_elements(
    mesh=recon_mesh,
    solutions_A0=sols,
    ntd_AD=ntd_AD,
    ntd_A0=ntd_A0,
    num_partitions=num_partitions,
    const=const,
    tol=tol,
)
#%%
plot_highlighted_elements_with_inclusions(recon_mesh, inclusion_indexes, recon_solver.cell_tags)


plt.show()
# %% #######################
inclusion_conductivity_2 = lambda x, y: (2, 0, 5)  # Different values for comparison

def my_conductivity_AD_2(x, y):
    return conductivity_AD(
        x,
        y,
        my_conductivity_A0,
        inclusion_conductivity_2,
        shape_choice,
        params
    )

# Create new solver with different conductivity
solverAD_2 = FemConductivitySolver(mesh=mesh)
c2, alpha2, beta2 = solverAD_2.set_conductivity(my_conductivity_AD_2, my_conductivity_A0)
solverAD_2.compute_ntd_map(max_freq=max_freq)

# Adjust parameters
const2 = c2 * (alpha2**2) / (beta2**2) 

# Get NTD for second conductivity
ntd_AD_2 = solverAD_2.ntd_matrix

# Run reconstruction for second conductivity
inclusion_indexes_2, test_matrix_eigenvalues_2, partitions_2 = find_inclusion_elements(
    mesh=recon_mesh,
    solutions_A0=sols,
    ntd_AD=ntd_AD_2,
    ntd_A0=ntd_A0,
    num_partitions=num_partitions,
    const=const,
    tol=0,
)

# %%
plot_highlighted_eigenvalues(recon_mesh, inclusion_indexes, test_matrix_eigenvalues)
#plt.savefig("horizontal_reconstruction.pdf", bbox_inches="tight")
plt.show()

plot_highlighted_eigenvalues(recon_mesh, inclusion_indexes_2, test_matrix_eigenvalues_2)
#plt.savefig("vertical_reconstruction.pdf", bbox_inches="tight")
plt.show()

plot_highlighted_elements_with_inclusions(recon_mesh, inclusion_indexes, np.array(inclusion_indexes_2))
# %%


ev_diff = np.array(list(test_matrix_eigenvalues.values()))-np.array(list(test_matrix_eigenvalues_2.values()))
print(np.max(ev_diff))
print(np.sum(np.abs(ev_diff)>1e-6))
dbin = ev_diff > 0
didx = np.where(dbin)[0]



# %%
plot_highlighted_eigenvalues(recon_mesh, inclusion_indexes_2, ev_diff)
plt.show()
# %%
plot_highlighted_elements_with_inclusions(recon_mesh, didx)
plt.show()

# %%
fig, ax = plot_highlighted_elements(recon_mesh, didx, color="black", alpha=0.7)
ax.set_title("")
#plt.savefig("binary_difference.pdf", bbox_inches="tight")
plt.show()

# %%
plot_test_matrix_eigenvalues(
    recon_mesh, ev_diff, sigmoid_scaling=None)

plt.show()
# %% #######################
# conductivity plots
from matplotlib.patches import Circle, Ellipse

# ----------------------------
# Ellipse definition
# ----------------------------
center = params[0]      # (x0, y0)
axes = params[1]        # semi-axes (a, b)

# Arrow ratio (vx, vy)
arrow_ratio = (2, 5)

# ----------------------------
# Create figure and axis
# ----------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal")

# ----------------------------
# Unit circle (domain)
# ----------------------------
unit_circle = Circle(
    (0.0, 0.0),
    radius=1.0,
    fill=False,
    linewidth=1
)
ax.add_patch(unit_circle)

# ----------------------------
# Filled ellipse
# ----------------------------
ellipse = Ellipse(
    xy=center,
    width=2 * axes[0],     # full width = 2a
    height=2 * axes[1],    # full height = 2b
    alpha=0.4,
    color='grey'
)

# ----------------------------
# Ellipse outline (line)
# ----------------------------
ellipse_line = Ellipse(
    xy=center,
    width=2 * axes[0],
    height=2 * axes[1],
    fill=False,
    linewidth=1
)

# Clip ellipse to unit circle
ellipse.set_clip_path(unit_circle)
ax.add_patch(ellipse)
ax.add_patch(ellipse_line)

# ----------------------------
# Compute arrow scaling
# ----------------------------
vx, vy = arrow_ratio
a, b = axes

scale = min(a / abs(vx), b / abs(vy))*0.9

dx = scale * vx
dy = scale * vy

x0, y0 = center

def draw_xy_arrows(ax, x0, y0, vx, vy, scale,
                   head_width=0.03,
                   label_offset=0.02):

    dx = scale * vx
    dy = scale * vy

    # X-direction arrow
    ax.arrow(
        x0, y0, dx, 0,
        head_width=head_width,
        length_includes_head=True,
        color='black',
        linewidth=0.5
    )

    # Y-direction arrow
    ax.arrow(
        x0, y0, 0, dy,
        head_width=head_width,
        length_includes_head=True,
        color='black',
        linewidth=0.5
    )

    # X-arrow label
    ax.text(
        x0 + dx * 0.55,
        y0 - label_offset,
        f"{vx}",
        ha="right",
        va="top"
    )

    # Y-arrow label
    ax.text(
        x0 - label_offset * 1.5,
        y0 + dy * 0.55,
        f"{vy}",
        ha="right",
        va="top"
    )

draw_xy_arrows(ax, x0, y0, vx, vy, scale)
draw_xy_arrows(ax, -x0, y0, 1, 1, 2*scale)

ax.set_xlabel("x")
ax.set_ylabel("y")
# ----------------------------
# Plot settings
# ----------------------------
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
plt.savefig("vertical_conductivity.pdf", bbox_inches="tight")
#plt.savefig("horizontal_conductivity.pdf", bbox_inches="tight")

plt.show()


# %%
