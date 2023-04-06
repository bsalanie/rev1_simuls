import dataclasses as dc
import pickle
from multiprocessing.pool import ThreadPool as Pool
from typing import Tuple, List

import numpy as np
from cupid_matching.choo_siow import (
    entropy_choo_siow,
    entropy_choo_siow_corrected,
)
from cupid_matching.matching_utils import Matching
from cupid_matching.min_distance import MDEResults, estimate_semilinear_mde
from cupid_matching.model_classes import ChooSiowPrimitives
from cupid_matching.poisson_glm import PoissonGLMResults, choo_siow_poisson_glm

from rev1_simuls.simulate import _run_simul
from rev1_simuls.plots import plot_simulation_results
from rev1_simuls.read_data import (
    read_margins,
    read_marriages,
    remove_zero_cells,
    rescale_mus,
    reshape_varcov,
)
from rev1_simuls.specification import _generate_bases_firstsub, generate_bases
from rev1_simuls.utils import data_dir, results_dir
from rev1_simuls.config import (
    age_start,
    age_end,
    do_simuls_mde,
    do_simuls_poisson,
    n_households_cupid_popu,
    use_rescale,
    n_sim,
    zero_guard,
    model_name,
    shrink_factor,
    use_mde_correction,
    renormalize_true_coeffs,
    n_households_sim,
    degrees,
    plot_simuls,
)


def prepare_data_cupid():
    nx, my = read_margins(data_dir, age_start=age_start, age_end=age_end)
    muxy, varmus = read_marriages(
        data_dir, age_start=age_start, age_end=age_end
    )
    n_types_men, n_types_women = muxy.shape
    print(
        f"""
    \nThe data has {n_types_men} types of men and {n_types_women} types of women.
    """
    )
    quantiles = np.arange(1, 10) / 100.0
    mus = Matching(muxy, nx, my)
    qmus = np.quantile(muxy, quantiles)
    mus_norm = (
        rescale_mus(mus, n_households_cupid_popu) if use_rescale else mus
    )
    varmus_norm = (
        reshape_varcov(varmus, mus, n_households_cupid_popu)
        if use_rescale
        else varmus
    )

    mus_norm_fixed = remove_zero_cells(mus_norm, coeff=zero_guard)
    muxy_norm_fixed = mus_norm_fixed.muxy
    qmus_fixed = np.quantile(muxy_norm_fixed, quantiles)
    print("   quantiles of raw and fixed muxy:")
    for q, qm, qmf in zip(quantiles, qmus, qmus_fixed):
        print(f"{q: .3f}: {qm: .2e}    {qmf: .2e}")
    return mus_norm_fixed, varmus_norm


def make_bases(
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


def name_and_pickle_primitives(
    mus_popu: Matching, varmus: np.ndarray, base_functions: np.ndarray
) -> str:
    """pickles the population matching, its variance, and the base functions,
    and creates a qualified name

    Args:
        mus_popu: the population Matching
        varmus: the corresponding variances
        base_functions: the values of the base functions

    Returns:
        the qualified name
    """
    full_model_name = model_name
    if shrink_factor != 1:
        full_model_name = f"{full_model_name}_f{shrink_factor}"
    if use_mde_correction:
        full_model_name = f"{full_model_name}_corrected"
    if (age_start > 16) or (age_end < 40):
        full_model_name = f"{full_model_name}_a{age_start}_{age_end}"
    with open(
        data_dir / f"{full_model_name}_mus_popu_{int(zero_guard)}.pkl",
        "wb",
    ) as f:
        pickle.dump(mus_popu, f)
    with open(data_dir / f"{full_model_name}_varmus_norm.pkl", "wb") as f:
        pickle.dump(varmus, f)
    with open(data_dir / f"{full_model_name}_base_functions.pkl", "wb") as f:
        pickle.dump(base_functions, f)
    return full_model_name


def estimate_choosiow(
    full_model_name: str, mus_popu: Matching, base_functions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """estimates Choo and Siow on the population data

    Args:
        full_model_name: the qualified name
        mus_popu: the population Matching
        base_functions: the values of the base functions

    Returns:
        the estimated coefficients, theri estimated variance,
        their estimated standard errors, and the estimated joint surplus
    """
    if do_simuls_mde:
        mde_results = estimate_semilinear_mde(
            mus_popu, base_functions, entropy_choo_siow
        )
        estim_Phi = mde_results.estimated_Phi
        estim_coeffs = mde_results.estimated_coefficients
        varcov_coeffs = mde_results.varcov_coefficients
        std_coeffs = mde_results.stderrs_coefficients
    else:
        poisson_results = choo_siow_poisson_glm(mus_popu, base_functions)
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

    if model_name == "choo_siow_cupid":
        mus_popu, varmus_popu = prepare_data_cupid()
        muxy_popu, mux0_popu, mu0y_popu, nx_popu, my_popu = mus_popu.unpack()
        base_functions, base_names = make_bases(nx_popu, my_popu, degrees)
        n_bases = len(base_names)
        full_model_name = name_and_pickle_primitives(
            mus_popu, varmus_popu, base_functions
        )
        (
            estim_coeffs,
            varcov_coeffs,
            std_coeffs,
            estim_Phi,
        ) = estimate_choosiow(full_model_name, mus_popu, base_functions)

        # we use the Phi and the margins we got from the Cupid dataset
        choo_siow_true = ChooSiowPrimitives(estim_Phi, nx_popu, my_popu)
    elif model_name.startswith(
        "choo_siow_firstsub"
    ):  # we regenerate the simulation in the first submitted version
        n_types_men = n_types_women = 20
        theta1 = np.array([1.0, 0.0, 0.0, -0.01, 0.02, -0.01, 0.5, 0.0])
        if model_name == "choo_siow_firstsub10":
            theta1 *= 10
        n_bases = theta1.size
        base_functions, base_names = _generate_bases_firstsub(
            n_types_men, n_types_women
        )
        Phi1 = base_functions @ theta1
        t = 0.2
        nx1 = np.logspace(
            start=0, base=1 - t, stop=n_types_men - 1, num=n_types_men
        )
        my1 = np.logspace(
            start=0, base=1 - t, stop=n_types_women - 1, num=n_types_women
        )
        choo_siow_true = ChooSiowPrimitives(Phi1, nx1, my1)
        estim_coeffs = theta1

    # generate random seeds
    rng = np.random.default_rng(130962)
    seeds = rng.integers(100_000, size=n_sim)
    verbose = 0

    entropy = (
        entropy_choo_siow_corrected
        if use_mde_correction
        else entropy_choo_siow
    )

    # run simulation
    list_args = [
        [
            i_sim,
            seeds[i_sim],
            choo_siow_true,
            n_households_sim,
            base_functions,
            entropy,
            zero_guard,
            do_simuls_mde,
            do_simuls_poisson,
            verbose,
        ]
        for i_sim in range(n_sim)
    ]
    nb_cpus = 4
    with Pool(nb_cpus) as pool:
        results = pool.starmap(_run_simul, list_args)

    # unpack simulation results
    if do_both:
        estim_coeffs_mde = np.zeros((n_sim, n_bases))
        estim_coeffs_poisson = np.zeros((n_sim, n_bases))
        for i_sim in range(n_sim):
            estim_coeffs_mde[i_sim, :] = results[i_sim][0]
            estim_coeffs_poisson[i_sim, :] = results[i_sim][1]
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
            estim_coeffs_mde[i_sim, :] = results[i_sim]
        simul_results = {
            "Base names": base_names,
            "Base functions": base_functions,
            "True coeffs": estim_coeffs,
            "MDE": estim_coeffs_mde,
        }
    elif do_simuls_poisson:
        estim_coeffs_poisson = np.zeros((n_sim, n_bases))
        for i_sim in range(n_sim):
            estim_coeffs_poisson[i_sim, :] = results[i_sim]
        simul_results = {
            "Base names": base_names,
            "Base functions": base_functions,
            "True coeffs": estim_coeffs,
            "Poisson": estim_coeffs_poisson,
        }
    if model_name == "choo_siow_cupid":
        simul_results["Cupid stderrs"] = std_coeffs
        simul_results["Cupid varcov"] = varcov_coeffs

    # and save them
    with open(
        results_dir
        / f"{full_model_name}_{n_households_sim}_{int(zero_guard)}.pkl",
        "wb",
    ) as f:
        pickle.dump(simul_results, f)

    if plot_simuls:
        n_households_popu = (
            n_households_cupid_popu
            if full_model_name.startswith("choo_siow_cupid")
            else None
        )
        plot_simulation_results(
            full_model_name,
            n_households_sim,
            n_sim,
            zero_guard,
            do_simuls_mde=do_simuls_mde,
            do_simuls_poisson=do_simuls_poisson,
            n_households_popu=n_households_popu,
        )

    seed = 75694
    mus_sim = choo_siow_true.simulate(n_households_sim, seed=seed)
    quantiles = np.arange(1, 10) / 100.0
    qmus = np.quantile(mus_sim.muxy, quantiles)
    print("Quantiles of simulated mus:")
    for q, qm in zip(quantiles, qmus):
        print(f"{q: .3f}: {qm: 10.3f}")
