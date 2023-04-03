from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from cupid_matching.matching_utils import Matching, _compute_margins


def ages_slice(
    age_start: int,
    age_end: int,
) -> slice:
    """returns the corresponding slice"""
    age0, age1 = age_start - 16, age_end - 15
    return slice(age0, age1)


def varmus_select_ages(
    varmus: np.ndarray,
    age_start: int,
    age_end: int,
) -> np.ndarray:
    """extracts the age-appropriate part of varmus"""
    if (age_start > 16) or (age_end < 40):
        nages = 25
        nages2 = nages * nages
        nb_ages = age_end - age_start + 1
        nb_ages2 = nb_ages * nb_ages
        vmus_dim = nb_ages2 + nb_ages + nb_ages
        vmus = np.zeros((vmus_dim, vmus_dim))
        age0, age1 = age_start - 16, age_end - 15
        for x1 in range(nb_ages):
            ox1 = x1 + age0
            for y1 in range(nb_ages):
                oy1 = y1 + age0
                varmus1 = varmus[ox1 * nages + oy1, :]
                vmus1 = np.zeros(vmus_dim)
                for x2 in range(nb_ages):
                    ox2 = x2 + age0
                    vmus1[(x2 * nb_ages) : (x2 * nb_ages + nb_ages)] = varmus1[
                        (ox2 * nages + age0) : (ox2 * nages + age1)
                    ]
                    vmus1[nb_ages2 : (nb_ages2 + nb_ages)] = varmus1[
                        (nages2 + age0) : (nages2 + age1)
                    ]
                    vmus1[(nb_ages2 + nb_ages) :] = varmus1[
                        (nages2 + nages + age0) : (nages2 + nages + age1)
                    ]
                    vmus[x1 * nb_ages + y1, :] = vmus1
        for x in range(nb_ages):
            ox = x + age0
            vmus[nb_ages2 + x, nb_ages2 : (nb_ages2 + nb_ages)] = varmus[
                nages2 + ox, (nages2 + age0) : (nages2 + age1)
            ]
            vmus[nb_ages2 + x, (nb_ages2 + nb_ages) :] = varmus[
                nages2 + ox, (nages2 + nages + age0) : (nages2 + nages + age1)
            ]
        for y in range(nb_ages):
            vmus[nb_ages2 + nb_ages + y, (nb_ages2 + nb_ages) :] = varmus[
                nages2 + nages + ox,
                (nages2 + nages + age0) : (nages2 + nages + age1),
            ]
        vmus[nb_ages2 : (nb_ages2 + nb_ages), :nb_ages2] = vmus[
            :nb_ages2, nb_ages2 : (nb_ages2 + nb_ages)
        ].T
        vmus[(nb_ages2 + nb_ages) :, :nb_ages2] = vmus[
            :nb_ages2, (nb_ages2 + nb_ages) :
        ].T
        vmus[(nb_ages2 + nb_ages) :, nb_ages2 : (nb_ages2 + nb_ages)] = vmus[
            nb_ages2 : (nb_ages2 + nb_ages), (nb_ages2 + nb_ages) :
        ].T
        return vmus
    else:
        return varmus


def read_margins(
    data_dir: Path,  # the data directory
    age_start: Optional[int] = 16,  # we exclude younger ages
    age_end: Optional[int] = 40,  # we exclude older ages
) -> Tuple[np.ndarray, np.ndarray]:
    """reads and returns the margins for men and for women"""
    nx = np.loadtxt(data_dir / "nx70n.txt")
    my = np.loadtxt(data_dir / "my70n.txt")
    ages = ages_slice(age_start, age_end)
    return nx[ages], my[ages]


def read_marriages(
    data_dir: Path,  # the data directory
    age_start: Optional[int] = 16,  # we exclude younger ages
    age_end: Optional[int] = 40,  # we exclude older ages
) -> Tuple[np.ndarray, np.ndarray]:
    """reads and returns the marriages and the variances"""
    muxy = np.loadtxt(data_dir / "muxy70nN.txt")
    varmus = np.loadtxt(data_dir / "varmus70nN.txt")
    vmus = varmus_select_ages(varmus, age_start, age_end)
    ages = ages_slice(age_start, age_end)
    return muxy[ages, ages], vmus


def reshape_varcov(
    varmus: np.ndarray,
    mus: Matching,
    n_households: float,
) -> tuple[np.ndarray]:
    """splits the variance-covariance matrix
    and renormalizes for a requested total number of households

    Args:
        varmus:  muxy row major, then  mux0, then mu0y packed in both dimensions
        mus: the original Matching
        n_households:  the number of households we want

    Returns:
         the 6 constituent blocks of the normalized variance-covariance
    """
    muxy, mux0, mu0y, *_ = mus.unpack()
    ncat_men, ncat_women = muxy.shape
    n_prod_categories = ncat_men * ncat_women
    # first we reshape
    varmus_xyzt = varmus[:n_prod_categories, :n_prod_categories]
    varmus_xyz0 = varmus[
        :n_prod_categories, n_prod_categories : (n_prod_categories + ncat_men)
    ]
    varmus_xy0t = varmus[:n_prod_categories, (n_prod_categories + ncat_men) :]
    varmus_x0z0 = varmus[
        n_prod_categories : (n_prod_categories + ncat_men),
        n_prod_categories : (n_prod_categories + ncat_men),
    ]
    varmus_x00y = varmus[
        n_prod_categories : (n_prod_categories + ncat_men),
        (n_prod_categories + ncat_men) :,
    ]
    varmus_0y0t = varmus[
        (n_prod_categories + ncat_men) :, (n_prod_categories + ncat_men) :
    ]
    varcovs = (
        varmus_xyzt,
        varmus_xyz0,
        varmus_xy0t,
        varmus_x0z0,
        varmus_x00y,
        varmus_0y0t,
    )
    # then we rescale
    n_households_mus = np.sum(muxy) + np.sum(mux0) + np.sum(mu0y)
    rescale_factor = n_households / n_households_mus
    rescale_factor2 = rescale_factor * rescale_factor
    varcovs = tuple(v * rescale_factor2 for v in varcovs)
    return varcovs


def rescale_mus(
    mus: Matching,  # muxy, mux0, mu0y
    n_households: float,  # the number of households we want
) -> Matching:  # the normalized Matching after rescaling
    """normalizes the marriages and margins to a requested total number of households"""
    muxy, mux0, mu0y, nx, my = mus.unpack()
    n_households_mus = np.sum(muxy) + np.sum(mux0) + np.sum(mu0y)
    rescale_factor = n_households / n_households_mus
    muxy_norm = muxy * rescale_factor
    nx_norm = nx * rescale_factor
    my_norm = my * rescale_factor
    mus_norm = Matching(muxy_norm, nx_norm, my_norm)
    return mus_norm


def _get_zeros_mu(
    mu: np.ndarray, eps: float = 1e-9
) -> Tuple[bool, np.ndarray, float]:
    mu_size = mu.size
    nonzero_mu = mu[mu > eps]
    min_nonzero = np.min(nonzero_mu)
    n_zeros_mu = mu_size - nonzero_mu.size
    mu_has_zeros = n_zeros_mu > 0
    return mu_has_zeros, mu_size, min_nonzero


def remove_zero_cells(
    mus: Matching,  # muxy, mux0, mu0y, n, m
    coeff: int = 100,  # default scale factor for delta
) -> Matching:  # the transformed muxy, mux0, mu0y, nx, my
    """if `coeff` is not 0, add small number `delta` to 0-cells to avoid numerical issues"""
    muxy, mux0, mu0y, *_ = mus.unpack()
    zeros_muxy, muxy_size, min_muxy = _get_zeros_mu(muxy)
    zeros_mux0, mux0_size, min_mux0 = _get_zeros_mu(mux0)
    zeros_mu0y, mu0y_size, min_mu0y = _get_zeros_mu(mu0y)
    some_zeros = zeros_muxy or zeros_mux0 or zeros_mu0y
    if not some_zeros or coeff == 0:
        return mus
    else:
        delta = min(min_muxy, min_mux0, min_mu0y) / coeff
        muxy_fixed = muxy.astype(float)
        mux0_fixed = mux0.astype(float)
        mu0y_fixed = mu0y.astype(float)
        n_cells = 0
        if zeros_muxy:
            muxy_fixed += delta
            n_cells += muxy_size
        if zeros_mux0:
            mux0_fixed += delta
            n_cells += mux0_size
        if zeros_mu0y:
            mu0y_fixed += delta
            n_cells += mu0y_size
        n_households = np.sum(muxy) + np.sum(mux0) + np.sum(mu0y)
        scale_factor = n_households / (n_households + delta * n_cells)
        muxy_fixed *= scale_factor
        mux0_fixed *= scale_factor
        mux0_fixed *= scale_factor
        nx_fixed, my_fixed = _compute_margins(
            muxy_fixed, mux0_fixed, mu0y_fixed
        )
        mus_fixed = Matching(muxy_fixed, nx_fixed, my_fixed)
        return mus_fixed
