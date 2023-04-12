from pathlib import Path
from typing import Optional, Tuple
import pickle

import numpy as np
from cupid_matching.matching_utils import Matching, _compute_margins

from rev1_simuls.utils import VarianceMatching


def ages_slice(
    age_start: int,
    age_end: int,
) -> slice:
    """returns the corresponding slice"""
    age0, age1 = age_start - 16, age_end - 15
    return slice(age0, age1)


def varmus_select_ages(
    varmus: VarianceMatching,
    age_start: int,
    age_end: int,
) -> VarianceMatching:
    """extracts the age-appropriate part of varmus"""
    if (age_start > 16) or (age_end < 40):
        (
            varmus_xyzt,
            varmus_xyz0,
            varmus_xy0t,
            varmus_x0z0,
            varmus_x00t,
            varmus_0y0t,
        ) = varmus.unpack()
        old_nages = 25
        new_nages = age_end - age_start + 1
        new_nages2 = new_nages * new_nages
        vmus_xyzt = np.zeros((new_nages2, new_nages2))
        vmus_xyz0 = np.zeros((new_nages2, new_nages))
        vmus_xy0t = np.zeros((new_nages2, new_nages))
        vmus_x0z0 = np.zeros((new_nages, new_nages))
        vmus_x00t = np.zeros((new_nages, new_nages))
        vmus_0y0t = np.zeros((new_nages, new_nages))
        age0, age1 = age_start - 16, age_end - 15
        for x1 in range(new_nages):
            ox1 = x1 + age0
            for y1 in range(new_nages):
                oy1 = y1 + age0
                oxy1 = ox1 * old_nages + oy1
                varmus_xyzt1 = varmus_xyzt[oxy1, :]
                varmus_xyz01 = varmus_xyz0[oxy1, :]
                varmus_xy0t1 = varmus_xy0t[oxy1, :]
                for x2 in range(new_nages):
                    ox2 = x2 + age0
                    vmus_xyzt[
                        oxy1, (x2 * new_nages) : (x2 * new_nages + new_nages)
                    ] = varmus_xyzt1[
                        (ox2 * old_nages + age0) : (ox2 * old_nages + age1)
                    ]
                vmus_xyz0[oxy1, :] = varmus_xyz01[age0:age1]
                vmus_xy0t[oxy1, :] = varmus_xy0t1[age0:age1]
        for x in range(new_nages):
            ox = x + age0
            vmus_x0z0[x, :] = varmus_x0z0[ox, age0:age1]
            vmus_x00t[x, :] = varmus_x00t[ox, age0:age1]
        for y in range(new_nages):
            oy = y + age0
            vmus_0y0t[y, :] = varmus[oy, age0:age1]
        return VarianceMatching(
            var_xyzt=vmus_xyzt,
            var_xyz0=vmus_xyz0,
            var_xy0t=vmus_xy0t,
            var_x0z0=vmus_x0z0,
            var_x00t=vmus_x00t,
            var_0y0t=vmus_0y0t,
        )
    else:
        return varmus


def read_margins(
    data_dir: Path,  # the data directory
    sample_size: str,  # large or small sample
    age_start: Optional[int] = 16,  # we exclude younger ages
    age_end: Optional[int] = 40,  # we exclude older ages
) -> Tuple[np.ndarray, np.ndarray]:
    """reads and returns the margins for men and for women"""
    nx = np.loadtxt(data_dir / f"{sample_size}_nx.txt")
    my = np.loadtxt(data_dir / f"{sample_size}_my.txt")
    ages = ages_slice(age_start, age_end)
    return nx[ages], my[ages]


def read_marriages(
    data_dir: Path,  # the data directory
    sample_size: str,  # large or small sample
    age_start: Optional[int] = 16,  # we exclude younger ages
    age_end: Optional[int] = 40,  # we exclude older ages
) -> Tuple[np.ndarray, np.ndarray]:
    """reads and returns the marriages and the variances"""
    muxy = np.loadtxt(data_dir / f"{sample_size}_muxy.txt")
    with open(data_dir / f"{sample_size}_varmus.pkl", "rb") as f:
        varmus = pickle.load(f)
    vmus = varmus_select_ages(varmus, age_start, age_end)
    ages = ages_slice(age_start, age_end)
    return muxy[ages, ages], vmus


def reshape_varcov(
    varmus: np.ndarray,
    mus: Matching,
    n_households: float,
) -> VarianceMatching:
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
    # then we rescale
    n_households_mus = np.sum(muxy) + np.sum(mux0) + np.sum(mu0y)
    rescale_factor = n_households / n_households_mus
    rescale_factor2 = rescale_factor * rescale_factor

    varcovs = VarianceMatching(
        var_xyzt=varmus_xyzt * rescale_factor2,
        var_xyz0=varmus_xyz0 * rescale_factor2,
        var_xy0t=varmus_xy0t * rescale_factor2,
        var_x0z0=varmus_x0z0 * rescale_factor2,
        var_x00y=varmus_x00y * rescale_factor2,
        var_0y0t=varmus_0y0t * rescale_factor2,
    )

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
