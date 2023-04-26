import dataclasses as dc
import multiprocessing as mp
import pickle
import time
from multiprocessing.pool import ThreadPool as Pool
from typing import List, Tuple

import numpy as np
from cupid_matching.choo_siow import (entropy_choo_siow,
                                      entropy_choo_siow_corrected)
from cupid_matching.entropy import EntropyFunctions
from cupid_matching.matching_utils import Matching
from cupid_matching.min_distance import estimate_semilinear_mde
from cupid_matching.model_classes import ChooSiowPrimitives
from cupid_matching.poisson_glm import choo_siow_poisson_glm

from rev1_simuls.config import (age_end, age_start, degrees, do_simuls_mde,
                                do_simuls_poisson, model_name, n_sim,
                                plot_simuls, renormalize_true_coeffs,
                                sample_size, shrink_factor, use_mde_correction,
                                zero_guard)
from rev1_simuls.plots import plot_simulation_results
from rev1_simuls.read_data import prepare_data_cupid
from rev1_simuls.simulate import _run_simul
from rev1_simuls.specification import make_bases_cupid
from rev1_simuls.utils import (VarianceMatching, data_dir, print_quantiles,
                               results_dir)


def name_and_pickle_primitives(
    mus: Matching, varmus: VarianceMatching, base_functions: np.ndarray
) -> str:
    """pickles the population matching, its variance, and the base functions,
    and creates a qualified name

    Args:
        mus: the population Matching
        varmus: the corresponding variances
        base_functions: the values of the base functions

    Returns:
        the qualified name
    """
    full_model_name = f"{model_name}_{sample_size}"
    if shrink_factor != 1:
        full_model_name = f"{full_model_name}_f{shrink_factor}"
    if use_mde_correction:
        full_model_name = f"{full_model_name}_corrected"
    if (age_start > 16) or (age_end < 40):
        full_model_name = f"{full_model_name}_a{age_start}_{age_end}"
    with open(
        data_dir / f"{full_model_name}_mus_{zero_guard}.pkl",
        "wb",
    ) as f:
        pickle.dump(mus, f)
    with open(data_dir / f"{full_model_name}_varmus.pkl", "wb") as f:
        pickle.dump(varmus, f)
    with open(data_dir / f"{full_model_name}_base_functions.pkl", "wb") as f:
        pickle.dump(base_functions, f)
    return full_model_name


def estimate_original(
    full_model_name: str,
    entropy: EntropyFunctions,
    mus: Matching,
    base_functions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """estimates Choo and Siow on the original data

    Args:
        full_model_name: the qualified name
        mus: the population Matching
        base_functions: the values of the base functions

    Returns:
        the estimated coefficients, their estimated variance,
        their estimated standard errors, and the estimated joint surplus
    """
    if do_simuls_mde:
        mde_results = estimate_semilinear_mde(mus, base_functions, entropy)
        estim_Phi = mde_results.estimated_Phi
        estim_coeffs = mde_results.estimated_coefficients
        varcov_coeffs = mde_results.varcov_coefficients
        std_coeffs = mde_results.stderrs_coefficients
    else:
        poisson_results = choo_siow_poisson_glm(mus, base_functions)
        estim_Phi = poisson_results.estimated_Phi
        estim_coeffs = poisson_results.estimated_beta
        var_gamma = poisson_results.variance_gamma
        n_bases = base_functions.shape[2]
        varcov_coeffs = var_gamma[-n_bases:, -n_bases:]
        std_coeffs = np.sqrt(np.diag(varcov_coeffs))
    print("original estimates:")
    for base_name, coeff, stderr in zip(base_names, estim_coeffs, std_coeffs):
        print(f"{base_name: <15}: {coeff: >10.3f}  ({stderr: >10.3f})")

    if shrink_factor != 1:
        # experiment with smaller Phi
        estim_Phi /= shrink_factor
        estim_coeffs /= shrink_factor
        varcov_coeffs /= shrink_factor * shrink_factor
        std_coeffs /= shrink_factor

    if renormalize_true_coeffs:
        n_bases = base_functions.shape[2]
        for i_base in range(n_bases):
            base_functions[:, :, i_base] *= estim_coeffs[i_base]
        estim_coeffs1 = 1.0 / estim_coeffs
        varcov_coeffs = varcov_coeffs * np.outer(estim_coeffs1, estim_coeffs1)
        std_coeffs *= np.abs(estim_coeffs1)
        estim_coeffs = np.ones(n_bases)

    if do_simuls_mde:
        mde_true = dc.replace(
            mde_results,
            estimated_coefficients=estim_coeffs,
            varcov_coefficients=varcov_coeffs,
            stderrs_coefficients=std_coeffs,
        )
        print(mde_true)
        with open(data_dir / f"{full_model_name}_mde_true.pkl", "wb") as f:
            pickle.dump(mde_true, f)

    elif do_simuls_poisson:
        poisson_true = dc.replace(
            poisson_results,
            estimated_beta=estim_coeffs,
            variance_gamma=varcov_coeffs,
            stderrs_beta=std_coeffs,
        )
        print(poisson_true)
        with open(data_dir / f"{full_model_name}_poisson_true.pkl", "wb") as f:
            pickle.dump(poisson_true, f)

    return estim_coeffs, varcov_coeffs, std_coeffs, estim_Phi


if __name__ == "__main__":
    do_both = do_simuls_mde and do_simuls_poisson

    mus, varmus = prepare_data_cupid(sample_size)
    muxy, mux0, mu0y, nx, my = mus.unpack()
    n_households_obs = np.sum(nx) + np.sum(my) - np.sum(muxy)
    base_functions, base_names = make_bases_cupid(nx, my, degrees)
    n_bases = len(base_names)
    full_model_name = name_and_pickle_primitives(mus, varmus, base_functions)
    entropy = (
        entropy_choo_siow_corrected
        if use_mde_correction
        else entropy_choo_siow
    )
    (
        estim_coeffs,
        varcov_coeffs,
        std_coeffs,
        estim_Phi,
    ) = estimate_original(full_model_name, entropy, mus, base_functions)

    # we use the Phi and the margins we got from the Cupid dataset
    choo_siow_true = ChooSiowPrimitives(estim_Phi, nx, my)

    with open(data_dir / f"{full_model_name}_phi_true.pkl", "wb") as f:
        pickle.dump(estim_Phi, f)

    # generate random seeds
    rng = np.random.default_rng(130962)
    seeds = rng.integers(100_000, size=n_sim)
    verbose = 0

    # run simulation
    list_args = [
        [
            i_sim,
            seeds[i_sim],
            choo_siow_true,
            n_households_obs,
            base_functions,
            entropy,
            zero_guard,
            do_simuls_mde,
            do_simuls_poisson,
            verbose,
        ]
        for i_sim in range(n_sim)
    ]
    nb_cpus = mp.cpu_count() - 2
    start_simuls = time.time()
    with Pool(nb_cpus) as pool:
        results = pool.starmap(_run_simul, list_args)
    end_simuls = time.time()

    # unpack simulation results
    if do_both:
        estim_coeffs_mde = np.zeros((n_sim, n_bases))
        estim_coeffs_poisson = np.zeros((n_sim, n_bases))
        for i_sim in range(n_sim):
            estim_coeffs_mde[i_sim, :] = results[i_sim][0][0]
            estim_coeffs_poisson[i_sim, :] = results[i_sim][0][1]
        simul_results = {
            "Base names": base_names,
            "Base functions": base_functions,
            "True coeffs": estim_coeffs,
            "MDE": estim_coeffs_mde,
            "Poisson": estim_coeffs_poisson,
        }
    elif do_simuls_mde:
        estim_coeffs_mde = np.zeros((n_sim, n_bases))
        for i_sim in range(n_sim):
            estim_coeffs_mde[i_sim, :] = results[i_sim][0]
        simul_results = {
            "Base names": base_names,
            "Base functions": base_functions,
            "True coeffs": estim_coeffs,
            "MDE": estim_coeffs_mde,
        }
    elif do_simuls_poisson:
        estim_coeffs_poisson = np.zeros((n_sim, n_bases))
        for i_sim in range(n_sim):
            estim_coeffs_poisson[i_sim, :] = results[i_sim][0]
        simul_results = {
            "Base names": base_names,
            "Base functions": base_functions,
            "True coeffs": estim_coeffs,
            "Poisson": estim_coeffs_poisson,
        }
    simul_results["Cupid stderrs"] = std_coeffs
    simul_results["Cupid varcov"] = varcov_coeffs
    phi_nonparam = np.zeros((n_sim, nx.size, my.size))
    for isim in range(n_sim):
        phi_nonparam[isim, :, :] = results[isim][1]
    simul_results["Phi non param"] = phi_nonparam

    # and save them
    with open(
        results_dir / f"{full_model_name}_{zero_guard}.pkl",
        "wb",
    ) as f:
        pickle.dump(simul_results, f)

    if plot_simuls:
        plot_simulation_results(
            full_model_name,
            n_sim,
            zero_guard,
            do_simuls_mde=do_simuls_mde,
            do_simuls_poisson=do_simuls_poisson,
        )

    seed = 75694
    mus_sim = choo_siow_true.simulate(n_households_obs, seed=seed)
    quantiles = np.arange(10, 30) / 100.0
    print("Quantiles of population and simulated mus:")
    print_quantiles([mus.muxy.flatten(), mus_sim.muxy.flatten()], quantiles)
    print(
        f"Numbers of marriages: {np.sum(mus.muxy)} and {np.sum(mus_sim.muxy)}"
    )
    print(
        f"\n\n**** {n_sim} simulations took {end_simuls - start_simuls: >10.3f} seconds on {nb_cpus} CPUs****"
    )
