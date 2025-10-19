import numpy as np

background_conductivity = [1, 0, 1]  # k11, k12, k22

ellipse_params = ((0.0, 0.0, 0.6, 0.4), (10, 0, 10))  # Center, axes
two_ellipses_params = (
    (-0.25, -0.25, 0.3, 0.2, 0.25, 0.25, 0.3, 0.2),
    (10, 0, 10),
)  # Centers, axes
triangle_params = (
    ((-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)),
    (10, 0, 10),
)  # Corner points
U_params = ((0.2, 0.6), (10, 0, 10))  # Thickness, radius


def conductivity_A0(x, y, background_conductivity=background_conductivity):
    """Background conductivity tensor"""
    inclusion_flag = 0
    return (
        background_conductivity[0],
        background_conductivity[1],
        background_conductivity[2],
        inclusion_flag,
    )


def in_ellipse(x, y, xc, yc, a, b):
    """Ellipse centered at (xc, yc) with semi-axes a, b"""
    return ((x - xc) / a) ** 2 + ((y - yc) / b) ** 2 <= 1


def in_two_ellipses(x, y, xc1, yc1, a1, b1, xc2, yc2, a2, b2):
    """Ellipse centered at (xc, yc) with semi-axes a, b"""
    in_ellipse1 = ((x - xc1) / a1) ** 2 + ((y - yc1) / b1) ** 2 <= 1
    in_ellipse2 = ((x - xc2) / a2) ** 2 + ((y - yc2) / b2) ** 2 <= 1

    return in_ellipse1 or in_ellipse2


def in_triangle(x, y, A, B, C):
    """Check if (x, y) is inside the triangle defined by A, B, C"""
    P = np.array([x, y])
    v0, v1, v2 = np.array(C) - np.array(A), np.array(B) - np.array(A), P - np.array(A)
    dot00, dot01, dot02 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v0, v2)
    dot11, dot12 = np.dot(v1, v1), np.dot(v1, v2)
    invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    return (u >= 0) and (v >= 0) and (u + v <= 1)


def in_U_shape(x, y, thickness=0.2, radius=0.5):
    """
    Check if (x, y) is inside a U-shaped region centered at (0,0),
    open upwards, with a rounded bottom.

    The U consists of:
      - Two vertical rectangular legs
      - A rounded bottom between two semicircles
    """

    # Outer and inner radii of the rounded part
    outer_r = radius
    inner_r = radius - thickness

    leg_x_inner = inner_r
    leg_x_outer = outer_r

    # --- Rounded bottom (between semicircles)
    dist = np.sqrt(x**2 + y**2)
    in_bottom_arc = (-outer_r <= y <= 0) and (inner_r <= dist <= outer_r)

    # --- Vertical legs (extending upward from the circular ends)
    in_left_leg = (-leg_x_outer <= x <= -leg_x_inner) and (0 <= y <= outer_r)
    in_right_leg = (leg_x_inner <= x <= leg_x_outer) and (0 <= y <= outer_r)

    # --- Combine all parts
    return in_bottom_arc or in_left_leg or in_right_leg


def conductivity_AD_ellipse(x, y, background_conductivity, ellipse_params):
    """Conductivity with elliptical inclusion"""
    (xc, yc, a, b), (k11_e, k12_e, k22_e) = ellipse_params
    k11, k12, k22, inclusion_flag = conductivity_A0(x, y, background_conductivity)
    if in_ellipse(x, y, xc, yc, a, b):
        k11, k12, k22 = k11_e, k12_e, k22_e
        inclusion_flag = 1
    return k11, k12, k22, inclusion_flag


def conductivity_AD_two_ellipses(x, y, background_conductivity, ellipse_params):
    """Conductivity with elliptical inclusion"""
    (xc1, yc1, a1, b1, xc2, yc2, a2, b2), (k11_e, k12_e, k22_e) = ellipse_params
    k11, k12, k22, inclusion_flag = conductivity_A0(x, y, background_conductivity)
    if in_two_ellipses(x, y, xc1, yc1, a1, b1, xc2, yc2, a2, b2):
        k11, k12, k22 = k11_e, k12_e, k22_e
        inclusion_flag = 1
    return k11, k12, k22, inclusion_flag


def conductivity_AD_triangle(x, y, background_conductivity, triangle_params):
    """Conductivity with triangular inclusion"""
    (A, B, C), (k11_t, k12_t, k22_t) = triangle_params
    k11, k12, k22, inclusion_flag = conductivity_A0(x, y, background_conductivity)
    if in_triangle(x, y, A, B, C):
        k11, k12, k22 = k11_t, k12_t, k22_t
        inclusion_flag = 1
    return k11, k12, k22, inclusion_flag


def conductivity_AD_Ushape(x, y, background_conductivity, U_params):
    """Conductivity with seamless curved U-shaped inclusion"""
    (thickness, radius), (k11_u, k12_u, k22_u) = U_params
    k11, k12, k22, inclusion_flag = conductivity_A0(x, y, background_conductivity)
    if in_U_shape(x, y, thickness=thickness, radius=radius):
        k11, k12, k22 = k11_u, k12_u, k22_u
        inclusion_flag = 1
    return k11, k12, k22, inclusion_flag


def conductivity_AD(x, y, background_conductivity, shape="ellipse", shape_params=None):
    if shape == "ellipse":
        if shape_params is None:
            shape_params = ellipse_params
        return conductivity_AD_ellipse(x, y, background_conductivity, shape_params)
    elif shape == "triangle":
        if shape_params is None:
            shape_params = triangle_params
        return conductivity_AD_triangle(x, y, background_conductivity, shape_params)
    elif shape == "U":
        if shape_params is None:
            shape_params = U_params
        return conductivity_AD_Ushape(x, y, background_conductivity, shape_params)
    elif shape == "two_ellipses":
        if shape_params is None:
            shape_params = two_ellipses_params
        return conductivity_AD_two_ellipses(x, y, background_conductivity, shape_params)
    else:
        raise ValueError(f"Unknown shape: {shape}")
