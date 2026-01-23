import numpy as np


def is_inside_ellipse(x, y, params):
    """Check if (x, y) is inside an ellipse defined by (center, semi-axes)."""

    x0, y0 = params[:2]
    a, b = params[2:]
    return ((x - x0) ** 2 / a**2 + (y - y0) ** 2 / b**2) <= 1.0


def is_inside_two_ellipses(x, y, params):
    """Return 1 or 2 if inside one of two ellipses, else 0."""
    cx1, cy1, a1, b1, cx2, cy2, a2, b2 = params
    if is_inside_ellipse(x, y, (cx1, cy1, a1, b1)):
        return 1
    elif is_inside_ellipse(x, y, (cx2, cy2, a2, b2)):
        return 2
    else:
        return 0


def is_inside_smiley(x, y, params=None):
    """
    Return region ID for smiley face parts:
    1: Right eye (Ellipse)
    2: Left eye (Square)
    3: Mouth (Smiling half-ellipse)
    0: Background
    """
    # Right eye: Ellipse
    ellipse_center_x, ellipse_center_y = 0.34, 0.32
    ellipse_axes = (0.3, 0.25)
    if (
        (x - ellipse_center_x) ** 2 / ellipse_axes[0] ** 2
        + (y - ellipse_center_y) ** 2 / ellipse_axes[1] ** 2
    ) <= 1.0:
        return 1

    # Left eye: Square
    square_center_x, square_center_y = -0.45, 0.45
    if abs(x - square_center_x) <= 0.125 and abs(y - square_center_y) <= 0.125:
        return 2

    # Mouth: Lower half of ellipse
    mouth_center_x, mouth_center_y = -0.25, -0.37

    if (
        (x - mouth_center_x) ** 2 / 0.5**2 + (y - mouth_center_y) ** 2 / 0.3**2
    ) <= 1.0 and (y < mouth_center_y):
        return 3

    return 0


def is_inside_L_shape(x, y, params=None):
    """
    NumPy version of L-shape membership test.
    Works with scalars or numpy arrays.
    """

    angle = -np.pi * 1.05
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    point = np.array([(x - 0.26) * 1.1, (y + 0.15) * 1.1])
    rotated_point = rot @ point
    X, Y = rotated_point[0], rotated_point[1]

    in_long = (X < 2 / 11) & (X > 0) & (np.abs(Y) < 1 / 3)
    in_short = (Y > (1 / 2 - 2 / 11)) & (Y < 1 / 2) & (X > 0) & (X < 1 / 2)

    return in_long | in_short


def is_inside_isotropy_test(x, y):
    """
    NumPy version of L-shape membership test.
    Works with scalars or numpy arrays.
    """
    angle = np.pi / 4  # 45 degrees
    rot = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )  # 45 degrees rotation

    point = np.array([x, y])
    rotated_point = rot @ point
    X, Y = rotated_point[0], rotated_point[1]

    in_long = (X < 2 / 11) & (X > 0) & (np.abs(Y) < 1 / 2)
    in_short = (Y > (1 / 2 - 2 / 11)) & (Y < 1 / 2) & (X > 0) & (X < 1 / 2)

    return in_long | in_short


def conductivity_AD(x, y, background_func, inclusion_funcs, shape_choice, shape_params):
    """
    Compute spatially varying conductivity tensor (a11, a12, a22)
    for background + inclusion shapes.

    Parameters
    ----------
    x, y : float
        Point coordinates.
    background_func : callable
        Function (x, y) -> (a11, a12, a22) for background region.
    inclusion_funcs : callable or tuple of callables
        One or several functions (x, y) -> (a11, a12, a22) for inclusions.
    shape_choice : str
        "ellipse", "two_ellipses", "L_shape", "smiley", etc.
    shape_params : tuple
        Parameters defining the chosen shape.
    """
    if shape_choice == "ellipse":
        if is_inside_ellipse(x, y, shape_params):
            return inclusion_funcs(x, y)
        else:
            return background_func(x, y)

    elif shape_choice == "two_ellipses":
        inside_idx = is_inside_two_ellipses(x, y, shape_params)
        if inside_idx == 1:
            return inclusion_funcs[0](x, y)
        elif inside_idx == 2:
            return inclusion_funcs[1](x, y)
        else:
            return background_func(x, y)

    elif shape_choice == "smiley":
        inside_idx = is_inside_smiley(x, y, shape_params)
        if inside_idx == 1:
            return inclusion_funcs[0](x, y)  # Right eye
        elif inside_idx == 2:
            return inclusion_funcs[1](x, y)  # Left eye
        elif inside_idx == 3:
            return inclusion_funcs[2](x, y)  # Mouth
        else:
            return background_func(x, y)

    elif shape_choice == "two_ellipses_and_L":
        ellipse1_params = shape_params[:4]
        ellipse2_params = shape_params[4:8]

        if is_inside_ellipse(x, y, ellipse1_params):
            return inclusion_funcs[0](x, y)
        elif is_inside_ellipse(x, y, ellipse2_params):
            return inclusion_funcs[1](x, y)
        elif is_inside_L_shape(x, y):
            return inclusion_funcs[2](x, y)
        else:
            return background_func(x, y)

    elif shape_choice == "L_shape":
        if is_inside_L_shape(x, y, shape_params):
            return inclusion_funcs(x, y)
        else:
            return background_func(x, y)

    elif shape_choice == "isotropy_test":
        if is_inside_isotropy_test(x, y):
            return inclusion_funcs(x, y)
        else:
            return background_func(x, y)

    else:
        raise ValueError(f"Unknown shape_choice '{shape_choice}'")
