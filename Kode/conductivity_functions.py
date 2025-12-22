def is_inside_ellipse(x, y, center, axes):
    """Check if (x, y) is inside an ellipse defined by (center, semi-axes)."""
    x0, y0 = center
    a, b = axes
    return ((x - x0) ** 2 / a**2 + (y - y0) ** 2 / b**2) <= 1.0


def is_inside_two_ellipses(x, y, params):
    """Return 1 or 2 if inside one of two ellipses, else 0."""
    cx1, cy1, a1, b1, cx2, cy2, a2, b2 = params
    if is_inside_ellipse(x, y, (cx1, cy1), (a1, b1)):
        return 1
    elif is_inside_ellipse(x, y, (cx2, cy2), (a2, b2)):
        return 2
    else:
        return 0


def is_inside_L_shape(x, y, params):
    """
    Check if (x, y) lies inside an L-shape centered at origin.
    params = (outer_size, thickness)
    """
    outer, thickness = params
    half = outer / 2
    t = thickness

    # Outside bounding box
    if not (-half <= x <= half and -half <= y <= half):
        return False

    # Remove top-right inner square (cutout)
    if (x > -half + t) and (y > -half + t):
        return False

    return True

import numpy as np

def is_inside_isotropy_test(x, y):
    """
    NumPy version of L-shape membership test.
    Works with scalars or numpy arrays.
    """
    angle = np.pi / 4  # 45 degrees
    rot = np.array([
        [np.cos(angle), -np.sin(angle)], 
        [np.sin(angle), np.cos(angle)]])  # 45 degrees rotation

    point = np.array([x, y])
    rotated_point = rot @ point
    X,Y = rotated_point[0], rotated_point[1]

    in_long = (X<2/11) & (X>0) & (np.abs(Y)<1/2)
    in_short = (Y>(1/2-2/11)) & (Y<1/2) & (X>0) & (X<1/2)

    return (in_long | in_short)



# ---------- Main conductivity definition ----------


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
        For "two_ellipses", must be a tuple (func1, func2).
    shape_choice : str
        "ellipse", "two_ellipses", or "L_shape".
    shape_params : tuple
        Parameters defining the chosen shape.
    """
    if shape_choice == "ellipse":
        center, axes = shape_params
        if is_inside_ellipse(x, y, center, axes):
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
