from fem_solver import FemConductivitySolver
import matplotlib.pyplot as plt
import numpy as np
from plotting_functions import (
    plot_conductivity_components,
)
from conductivity_functions import (
    conductivity_AD,
)


def polar_to_euclidean(polar_conductivities_A, x, y):
    """
    Convert a diagonal polar conductivity tensor diag(a_r, a_theta)
    into its Cartesian (x, y) coordinate form, automatically computing theta.

    Parameters
    ----------
    a_r : float
        Radial conductivity component.
    a_theta : float
        Tangential (angular) conductivity component.
    x, y : float or array_like
        Cartesian coordinates.

    Returns
    -------
    A : Euclidean conducitivity tensor components (Axx, Axy, Ayy)
    """

    a_r, a_theta = polar_conductivities_A

    # Compute polar angle
    theta = np.arctan2(y, x)

    # Compute cos/sin
    c = np.cos(theta)
    s = np.sin(theta)

    # Transform to Euclidean
    Axx = a_r * c**2 + a_theta * s**2
    Axy = (a_r - a_theta) * s * c
    Ayy = a_r * s**2 + a_theta * c**2

    return (Axx, Axy, Ayy)


def map_forward(x, y, s):
    """
    Maps (x, y) -> (r, theta) -> (r^s, theta) -> (x_new, y_new)
    """
    # 1. Cartesian to Polar
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Handle origin to avoid numerical errors
    if r < 1e-12:
        return 0.0, 0.0

    # 2. Apply Diffeomorphism (Forward)
    # R(r) = r^s
    r_new = r**s

    # 3. Polar to Cartesian (using the NEW radius)
    x_new = r_new * np.cos(theta)
    y_new = r_new * np.sin(theta)

    return x_new, y_new


### Conductivity

# Define geometry
r = 0.5

shape_choice = "ellipse"
shape_params = (0, 0, r, r)

# Each layer given in polar coordinates (a_r, a_theta)
polar_conductivities_A = [
    (4.0, 8.0),  # r1 <= r < r2, layer 2
    (1.0, 1.0),  # r2 <= r <= 1, layer 1
]

s = 2.0

polar_conductivities_Atilde = [
    (s * polar_conductivities_A[0][0], 1 / s * polar_conductivities_A[0][1]),
    (s * polar_conductivities_A[1][0], 1 / s * polar_conductivities_A[1][1]),
]

### Setings

h = 0.02

max_freq = 10
modes = range(1, max_freq + 1)

A0 = lambda x, y: polar_to_euclidean(polar_conductivities_A[1], x, y)
inclusion_conductivity_A = lambda x, y: polar_to_euclidean(
    polar_conductivities_A[0], x, y
)

A0tilde = lambda x, y: polar_to_euclidean(polar_conductivities_Atilde[1], x, y)
inclusion_conductivity_Atilde = lambda x, y: polar_to_euclidean(
    polar_conductivities_Atilde[0], x, y
)


def my_conductivity_A(x, y):
    return conductivity_AD(
        x,
        y,
        A0,
        inclusion_conductivity_A,
        shape_choice,
        shape_params,
    )


def my_conductivity_Atilde(x, y):
    x, y = map_forward(x, y, 1.0 / s)
    return conductivity_AD(
        x,
        y,
        A0tilde,
        inclusion_conductivity_Atilde,
        shape_choice,
        shape_params,
    )


solver_A = FemConductivitySolver(mesh_characteristic_length=h)
solver_A.set_conductivity(my_conductivity_A, A0)

mesh = solver_A.get_mesh()

solver_Atilde = FemConductivitySolver(mesh=mesh)
solver_Atilde.set_conductivity(my_conductivity_Atilde, A0tilde)

ntd_A, n_freqs = solver_A.compute_ntd_map(max_freq=max_freq)
ntd_Atilde, n_freqs = solver_Atilde.compute_ntd_map(max_freq=max_freq)

# Magnitude of Lambda(A) - Lambda(Atilde) eigenvalues
print(np.linalg.norm(ntd_A - ntd_Atilde, ord=2))

print(np.diag(ntd_A))
print(np.diag(ntd_Atilde))

fig, axes = plot_conductivity_components(my_conductivity_A, title="A")

fig, axes = plot_conductivity_components(my_conductivity_Atilde, title="")
# plt.savefig("Figures/Analytic_example/non_uniqueness_example.pdf")

plt.show()
