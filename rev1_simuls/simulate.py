""" simulate the model """
from typing import Tuple
import numpy as np

from cupid_matching.entropy import EntropyFunctions
from cupid_matching.min_distance_utils import MDEResults
from cupid_matching.poisson_glm_utils import PoissonGLMResults
from cupid_matching.model_classes import ChooSiowPrimitives
from cupid_matching.min_distance import estimate_semilinear_mde
from cupid_matching.poisson_glm import choo_siow_poisson_glm
from rev1_simuls.read_data import remove_zero_cells


def _run_simul(
    i_sim: int,
    seed: int,
    choo_siow_true: ChooSiowPrimitives,
    n_households_sim: float,
    base_functions: np.ndarray,
    entropy: EntropyFunctions,
    value_coeff: float,
    do_simuls_mde: bool,
    do_simuls_poisson: bool,
    verbose: int = 0,
) -> MDEResults | PoissonGLMResults | Tuple[MDEResults, PoissonGLMResults]:
    """runs one simulation

    Args:
        i_sim: the index of the simulation
        seed: the seed for its random draws
        choo_siow_true: the DGP we simulate from
        n_households_sim: the number of households in the simulation
        base_functions:  the bases
        entropy:  the entropy
        value_coeff: the divider
        do_simuls_mde: run the MDE simulation
        do_simuls_poisson:  run the Poisson simulation
        verbose:  how verbose: 1 print simulation number,
                                2 print steps

    Returns:
        the Results object(s) from the simulation
    """
    global n_simuls_done
    do_both = do_simuls_mde and do_simuls_poisson
    mus_sim = choo_siow_true.simulate(n_households_sim, seed=seed)
    mus_sim_non0 = remove_zero_cells(mus_sim, coeff=value_coeff)
    if verbose >= 1:
        print(f"Doing simul {i_sim}")
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
            mus_sim_non0, base_functions, verbose=0
        )
        estim_coeffs_poisson = poisson_results_sim.estimated_beta
        if verbose == 2:
            print(f"    Done Poisson {i_sim}")
    n_simuls_done += 1
    print(f"        Done {n_simuls_done} simuls")
    if verbose >= 1:
        print(f"        Done simul {i_sim}")
    if do_both:
        return estim_coeffs_mde, estim_coeffs_poisson
    elif do_simuls_mde:
        return estim_coeffs_mde
    elif do_simuls_poisson:
        return estim_coeffs_poisson
