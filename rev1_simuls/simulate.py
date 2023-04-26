""" runs one simulation of the estimates """
from typing import Tuple

import numpy as np
from cupid_matching.entropy import EntropyFunctions
from cupid_matching.min_distance import estimate_semilinear_mde
from cupid_matching.min_distance_utils import MDEResults
from cupid_matching.model_classes import ChooSiowPrimitives
from cupid_matching.poisson_glm import choo_siow_poisson_glm
from cupid_matching.poisson_glm_utils import PoissonGLMResults

from rev1_simuls.config import n_simulations_done
from rev1_simuls.read_data import remove_zero_cells


def _run_simul(
    i_sim: int,
    seed: int,
    choo_siow_true: ChooSiowPrimitives,
    n_households_obs: int,
    base_functions: np.ndarray,
    entropy: EntropyFunctions,
    zero_guard: int,
    do_simuls_mde: bool,
    do_simuls_poisson: bool,
    verbose: int = 0,
) -> Tuple[
    MDEResults | PoissonGLMResults | Tuple[MDEResults, PoissonGLMResults],
    np.ndarray,
]:
    """runs one simulation

    Args:
        i_sim: the index of the simulation
        seed: the seed for its random draws
        choo_siow_true: the DGP we simulate from
        n_households_obs: the number of households
        base_functions:  the bases
        entropy:  the entropy
        zero_guard: the divider
        do_simuls_mde: run the MDE simulation
        do_simuls_poisson:  run the Poisson simulation
        verbose:  how verbose: 1 print simulation number,
                                2 print steps

    Returns:
        the Results object(s) from the simulation
        and the nonparametric estimate of the joint surplus
    """
    do_both = do_simuls_mde and do_simuls_poisson
    mus_sim = choo_siow_true.simulate(n_households_obs, seed=seed)
    mus_sim_non0 = remove_zero_cells(mus_sim, coeff=zero_guard)
    if verbose >= 1:
        print(f"Doing simul {i_sim}")
    muxy, mux0, mu0y, *_ = mus_sim_non0.unpack()
    phi_nonparam = 2.0 * np.log(muxy) - np.log(mu0y)
    phi_nonparam -= np.log(mux0).reshape((-1, 1))
    if do_simuls_mde:
        if verbose == 2:
            print(f"    Doing MDE {i_sim}")
        mde_results_sim = estimate_semilinear_mde(
            mus_sim_non0, base_functions, entropy
        )
        estim_coeffs_mde = mde_results_sim.estimated_coefficients
        if verbose == 2:
            print(f"    Done MDE {i_sim}")
    if do_simuls_poisson:
        if verbose == 2:
            print(f"    Doing Poisson {i_sim}")
        poisson_results_sim = choo_siow_poisson_glm(
            mus_sim, base_functions, verbose=0
        )
        estim_coeffs_poisson = poisson_results_sim.estimated_beta
        if verbose == 2:
            print(f"    Done Poisson {i_sim}")
    if verbose >= 1:
        print(f"        Done simul {i_sim}")

    with n_simulations_done.get_lock():
        n_simulations_done.value += 1
    print(f"       Did {n_simulations_done.value} simulations")

    if do_both:
        return ((estim_coeffs_mde, estim_coeffs_poisson), phi_nonparam)
    elif do_simuls_mde:
        return ((estim_coeffs_mde,), phi_nonparam)
    elif do_simuls_poisson:
        return ((estim_coeffs_poisson,), phi_nonparam)
