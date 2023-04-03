import sys
from pathlib import Path

import numpy as np

data_dir = Path("..") / "ChooSiow70nNdata"
results_dir = Path("..") / "Results"


def nprepeat_col(
    v: np.ndarray,
    n: int,
) -> np.ndarray:
    """create a matrix with `n` columns equal to the vector `v`

    Args:
        v: a 1-dim array of size `m`
        n:  the number of columns requested

    Returns:
        a 2-dim array of shape `(m, n)`
    """
    return np.repeat(v[:, np.newaxis], n, axis=1)


def nprepeat_row(
    v: np.ndarray,
    m: int,
) -> np.ndarray:
    """create a matrix with `m` rows equal to the vector `v`

    Args:
        v: a 1-dim array of size `n`
        n:  the number of rows requested

    Returns:
        a 2-dim array of shape `(m, n)`
    """
    return np.repeat(v[np.newaxis, :], m, axis=0)


def legendre_polynomials(
    x: np.ndarray,
    max_deg: int,
    a: float = -1.0,
    b: float = 1.0,
    no_constant: bool = False,
) -> np.ndarray:
    """evaluates the Legendre polynomials over x in the interval [a, b]

    Args:
        x: the points where the polynomials are to be evaluated
        max_deg: the maximum degree
        a: the start of the interval, classically -1
        b: the end of the interval, classically 1
        no_constant: if True, delete the constant polynomial

    Returns:
        an array of (max_deg+1) arrays of the shape of x
    """
    if a > np.min(x):
        sys.exit("legendre_polynomials: points below start of interval")
    if b < np.max(x):
        sys.exit("legendre_polynomials: points above end of interval")
    p = np.zeros((x.size, max_deg + 1))
    x_transf = 2.0 * (x - a) / (b - a) - 1.0
    p[:, 0] = np.ones_like(x)
    p[:, 1] = x_transf
    for deg in range(2, max_deg + 1):
        p2 = (2 * deg - 1) * (p[:, deg - 1] * x_transf) - (deg - 1) * p[:, deg - 2]
        p[:, deg] = p2 / deg
    polys_p = p[:, 1:] if no_constant else p
    return polys_p


def quantile_transform(v: np.ndarray) -> np.ndarray:
    """transform a vector of counts into the corresponding quantiles

    Args:
        v:  a vector of counts

    Returns:
         the corresponding quantiles
    """
    n = v.size
    q = np.zeros(n)
    for i in range(n):
        q[i] = np.sum(v <= v[i]) / (n + 1)
    return q
