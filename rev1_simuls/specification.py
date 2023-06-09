from typing import List, Tuple

import numpy as np
from cupid_matching.utils import nprepeat_col, nprepeat_row

from rev1_simuls.utils import legendre_polynomials, quantile_transform


def monomial_name(var: str, deg: int) -> str:
    """create a string with a nice name for a univariate term"""
    if deg == 0:
        return 1
    if deg == 1:
        return var
    if deg > 1:
        return var + "^" + str(deg)


def make_base_name(deg_x: int, deg_y: int) -> str:
    """create a string with a nice name for a bivariate term

    Args:
        deg_x: degree in x
        deg_y: degree in y

    Returns:
        a string that looks like x^a y^b
    """
    if deg_x == 0:
        return monomial_name("y", deg_y)
    if deg_y == 0:
        return monomial_name("x", deg_x)
    # both degrees are positive
    return monomial_name("x", deg_x) + " " + monomial_name("y", deg_y)


def generate_bases(
    nx: np.ndarray,
    my: np.ndarray,
    degrees: List[Tuple[int, int]],
) -> Tuple[np.ndarray, List[str]]:
    """generates the bases for a semilinear specification

    Args:
        nx: the numbers of men of each type
        my: the numbers of women of each type
        degrees: the list of degrees for polynomials in `x` and `y`

    Returns:
         the matrix of base functions and their names
    """
    n_bases = 3 + len(degrees)
    n_types_men, n_types_women = nx.size, my.size
    types_men, types_women = np.arange(n_types_men), np.arange(n_types_women)
    base_funs = np.zeros((n_types_men, n_types_women, n_bases))
    base_names = [None] * n_bases
    base_funs[:, :, 0] = 1.0
    base_names[0] = "1"
    for y in types_women:
        base_funs[:, y, 1] = np.where(types_men > y, 1.0, 0.0)
        base_funs[:, y, 2] = np.where(types_men > y, types_men - y, 0)
    base_funs[:, :, 2] /= (n_types_men + n_types_women) / 2
    base_names[1] = "1(x>y)"
    base_names[2] = "max(x-y,0)"
    # we quantile-transform nx and my
    q_nx = quantile_transform(nx)
    q_my = quantile_transform(my)
    # and we use the Legendre polynomials on [0,1]
    max_deg_x = max(degree[0] for degree in degrees)
    max_deg_y = max(degree[1] for degree in degrees)
    polys_x = legendre_polynomials(q_nx, max_deg_x, a=0)
    polys_y = legendre_polynomials(q_my, max_deg_y, a=0)
    i_base = 3
    for deg_x, deg_y in degrees:
        poly_x = polys_x[:, deg_x]
        poly_y = polys_y[:, deg_y]
        base_funs[:, :, i_base] = np.outer(poly_x, poly_y)
        base_names[i_base] = make_base_name(deg_x, deg_y)
        i_base += 1
    return base_funs, base_names


def _generate_bases_firstsub(
    n_types_men: int,
    n_types_women: int,
) -> Tuple[np.ndarray, List[str]]:
    """generate the bases used in the first submission

    Args:
        n_types_men: the number of types of men
        n_types_women: the number of types of women

    Returns:
        the matrix of base functions and their names
    """
    n_bases = 8
    base_functions = np.zeros((n_types_men, n_types_women, n_bases))
    base_functions[:, :, 0] = 1.0
    vec_x = np.arange(n_types_men)
    vec_y = np.arange(n_types_women)
    base_functions[:, :, 1] = nprepeat_col(vec_x, n_types_women)
    base_functions[:, :, 2] = nprepeat_row(vec_y, n_types_men)
    base_functions[:, :, 3] = base_functions[:, :, 1] * base_functions[:, :, 1]
    base_functions[:, :, 4] = base_functions[:, :, 1] * base_functions[:, :, 2]
    base_functions[:, :, 5] = base_functions[:, :, 2] * base_functions[:, :, 2]
    for i in range(n_types_men):
        for j in range(i, n_types_women):
            base_functions[i, j, 6] = 1
            base_functions[i, j, 7] = i - j
    base_names = ["1", "x", "y", "x^2", "xy", "y^2", "1(x>y)", "max(x-y,0)"]
    return base_functions, base_names


def make_bases_cupid(
    nx: np.ndarray, my: np.ndarray, degrees: List[Tuple[int, int]]
) -> Tuple[np.ndarray, List[str]]:
    """create base functions

    Args:
        nx: numbers of men of each type
        my: numbers of women of each type
        degrees: degrees of the bivariate polynomials

    Returns:
        the values of the base functions and their names
    """
    base_functions, base_names = generate_bases(nx, my, degrees)
    n_bases = base_functions.shape[-1]
    print(f"We created {n_bases} bases:")
    for i_base, base_name in enumerate(base_names):
        print(f"{i_base+1}: {base_name}")
    return base_functions, base_names
