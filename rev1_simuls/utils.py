import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cupid_matching.utils import bs_error_abort, test_matrix, test_vector

data_dir = Path("..") / "ChooSiow70nNdata"
results_dir = Path("..") / "Results"


@dataclass
class VarianceMatching:
    """the six matrix components of the variance of a Matching"""

    var_xyzt: np.ndarray
    var_xyz0: np.ndarray
    var_xy0t: np.ndarray
    var_x0z0: np.ndarray
    var_x00t: np.ndarray
    var_0y0t: np.ndarray

    def unpack(self):
        return (
            self.var_xyzt,
            self.var_xyz0,
            self.var_xy0t,
            self.var_x0z0,
            self.var_x00t,
            self.var_0y0t,
        )

    def __post__init__(self):
        XY, XY2 = test_matrix(self.var_xyzt)
        if XY2 != XY:
            bs_error_abort(
                f"var_xyzt should be a square matrix, not ({XY}, {XY2})"
            )
        XY3, X = test_matrix(self.var_xyz0)
        if XY3 != XY:
            bs_error_abort(f"var_xyz0 should have {XY} rows, not {XY3})")
        XY4, Y = test_matrix(self.var_xy0t)
        if XY4 != XY:
            bs_error_abort(f"var_xy0t should have {XY} rows, not {XY4})")
        if X * Y != XY:
            bs_error_abort(
                f"var_xyzt has {XY} rows, but varxyz0 has {X} columns and varxy0t has {Y}"
            )
        X2, X3 = test_matrix(self.var_x0z0)
        if X2 != X:
            bs_error_abort(f"var_x0z0 has {X2} rows, it should have {X}")
        if X3 != X:
            bs_error_abort(f"var_x0z0 has {X3} columns, it should have {X}")
        X4, Y2 = test_matrix(self.var_x00t)
        if X4 != X:
            bs_error_abort(f"var_x00t has {X4} rows, it should have {X}")
        if Y2 != Y:
            bs_error_abort(f"var_x00t has {Y2} columns, it should have {Y}")
        Y3, Y4 = test_matrix(self.var_0y0t)
        if Y3 != Y:
            bs_error_abort(f"var_x00t has {Y3} rows, it should have {Y}")
        if Y4 != Y:
            bs_error_abort(f"var_x00t has {Y4} columns, it should have {Y}")


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
        p2 = (2 * deg - 1) * (p[:, deg - 1] * x_transf) - (deg - 1) * p[
            :, deg - 2
        ]
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


def print_quantiles(
    v: np.ndarray | list[np.ndarray], quantiles: np.ndarray
) -> np.ndarray:
    """print these quantiles of the array(s)

    Args:
        v:  a vector or a list of vectors
        qtiles: the quantiles in [0,1]

    Returns:
         the corresponding quantiles as a vector or a matrix
    """
    nq = test_vector(quantiles)
    if isinstance(v, np.ndarray):
        qvals = np.quantile(v, quantiles)
        for q, qv in zip(quantiles, qvals):
            print(f"Quantile {q: .3f}: {qv: >10.3f}")
    elif isinstance(v, list):
        for v_i in v:
            _ = test_vector(v_i)
        nv = len(v)
        qvals = np.zeros((nq, nv))
        for i in range(nv):
            qvals[:, i] = np.quantile(v[i], quantiles)
        for iq, q in enumerate(quantiles):
            s = f"Quantile {q: .3f}: "
            qv = qvals[iq, :]
            for i in range(nv):
                s += f"  {qv[i]: >10.3f}"
            print(f"{s}")
    else:
        bs_error_abort("v must be  a vector or a list of vectors")

    return qvals
