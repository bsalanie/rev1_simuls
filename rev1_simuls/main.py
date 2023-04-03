import dataclasses as dc
import pickle
from multiprocessing.pool import ThreadPool as Pool
from typing import Tuple

import numpy as np
from cupid_matching.choo_siow import (entropy_choo_siow,
                                      entropy_choo_siow_corrected)
from cupid_matching.entropy import EntropyFunctions
from cupid_matching.matching_utils import Matching, _get_singles
from cupid_matching.min_distance import MDEResults, estimate_semilinear_mde
from cupid_matching.model_classes import ChooSiowPrimitives
from cupid_matching.poisson_glm import PoissonGLMResults, choo_siow_poisson_glm

from rev1_simuls.plots import plot_simulation_results
from rev1_simuls.read_data import (read_margins, read_marriages,
                                   remove_zero_cells, rescale_mus,
                                   reshape_varcov)
from rev1_simuls.specification import _generate_bases_firstsub, generate_bases
from rev1_simuls.utils import data_dir, results_dir

renormalize_true_coeffs = True  # renormalize so that all "true" coeffs = 1

do_simuls_mde = True
do_simuls_poisson = True
do_both = do_simuls_mde and do_simuls_poisson

plot_simuls = True

# model_name = "choo_siow_firstsub"
model_name = "choo_siow_cupid"
# model_name = "choo_siow_firstsub10"

use_rescale = False  # rescale the sample
use_mde_correction = False  # use the `corrected` version of MDE

n_households_cupid_pop = 13_274_041  # number of households in the Cupid population
n_households_cupid_obs = 75_265  # number of households in the Cupid sample

age_start, age_end = 16, 40  # we select ages

#  number of households in the simulation:
n_households_sim = n_households_cupid_obs
# n_households_sim =  1_000_000
# n_households_sim =  n_households_cupid_pop

n_sim = 5  # number of simulations
value_coeff = 1  # we set the zeros at the smallest positive value
# divided by value_coeff,
# except if value_coeff is 0

shrink_factor = 1  # we shrink the Choo-Siow estimates by a multiplicative integer


if model_name == "choo_siow_cupid":
    nx, my = read_margins(data_dir, age_start=age_start, age_end=age_end)
    muxy, varmus = read_marriages(data_dir, age_start=age_start, age_end=age_end)
    n_types_men, n_types_women = muxy.shape
    mux0, mu0y = _get_singles(muxy, nx, my)
    print(
        f"""
    \nThe data has {n_types_men} types of men and {n_types_women} types of women.
    """
    )


if model_name == "choo_siow_cupid":
    mus = Matching(muxy, nx, my)
    qmus = np.quantile(muxy, np.arange(1, 100) / 100.0)
    mus_norm = rescale_mus(mus, n_households_cupid_obs) if use_rescale else mus
    varmus_norm = (
        reshape_varcov(varmus, mus, n_households_cupid_obs) if use_rescale else varmus
    )

if model_name == "choo_siow_cupid":
    mus_norm_fixed = remove_zero_cells(mus_norm, coeff=value_coeff)
    (
        muxy_norm_fixed,
        mux0_norm_fixed,
        mu0y_norm_fixed,
        nx_norm_fixed,
        my_norm_fixed,
    ) = mus_norm_fixed.unpack()
    qmus_fixed = np.quantile(muxy_norm_fixed, np.arange(1, 100) / 100.0)
    print("   quantiles of raw and fixed muxy:")
    print(np.column_stack((qmus, qmus_fixed)))

if model_name == "choo_siow_cupid":
    degrees = [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    base_functions, base_names = generate_bases(nx_norm_fixed, my_norm_fixed, degrees)
    n_bases = base_functions.shape[-1]
    print(f"We created {n_bases} bases:")
    print(f"{base_names}")

if model_name == "choo_siow_cupid":
    full_model_name = model_name
    if shrink_factor != 1:
        full_model_name = f"{full_model_name}_f{shrink_factor}"
    if use_mde_correction:
        full_model_name = f"{full_model_name}_corrected"
    if (age_start > 16) or (age_end < 40):
        full_model_name = f"{full_model_name}_a{age_start}_{age_end}"
    with open(
        data_dir / f"{full_model_name}_mus_norm_fixed_{int(value_coeff)}.pkl",
        "wb",
    ) as f:
        pickle.dump(mus_norm_fixed, f)
    with open(data_dir / f"{full_model_name}_varmus_norm.pkl", "wb") as f:
        pickle.dump(varmus_norm, f)
    with open(data_dir / f"{full_model_name}_base_functions.pkl", "wb") as f:
        pickle.dump(base_functions, f)

if model_name == "choo_siow_cupid":
    if do_simuls_mde:
        mde_results = estimate_semilinear_mde(
            mus_norm_fixed, base_functions, entropy_choo_siow
        )
        estim_Phi = mde_results.estimated_Phi
        estim_coeffs = mde_results.estimated_coefficients
        varcov_coeffs = mde_results.varcov_coefficients
        std_coeffs = mde_results.stderrs_coefficients
        print("original estimates:")
        print(np.column_stack((estim_coeffs, std_coeffs)))
    else:
        poisson_results = choo_siow_poisson_glm(mus_norm_fixed, base_functions)
        estim_Phi = poisson_results.estimated_Phi
        estim_coeffs = poisson_results.estimated_beta
        var_gamma = poisson_results.variance_gamma
        varcov_coeffs = var_gamma[-n_bases:, -n_bases:]
        std_coeffs = np.sqrt(np.diag(varcov_coeffs))


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


if (
    model_name == "choo_siow_cupid"
):  # we use the Phi and the margins we got from the Cupid dataset
    choo_siow_estim = ChooSiowPrimitives(estim_Phi, nx_norm_fixed, my_norm_fixed)
elif model_name.startswith(
    "choo_siow_firstsub"
):  # we regenerate the simulation in the first submitted version
    n_types_men = n_types_women = 20
    theta1 = np.array([1.0, 0.0, 0.0, -0.01, 0.02, -0.01, 0.5, 0.0])
    if model_name == "choo_siow_firstsub10":
        theta1 *= 10
    n_bases = theta1.size
    base_functions, base_names = _generate_bases_firstsub(n_types_men, n_types_women)
    Phi1 = base_functions @ theta1
    t = 0.2
    nx1 = np.logspace(start=0, base=1 - t, stop=n_types_men - 1, num=n_types_men)
    my1 = np.logspace(start=0, base=1 - t, stop=n_types_women - 1, num=n_types_women)
    choo_siow_estim = ChooSiowPrimitives(Phi1, nx1, my1)
    estim_coeffs = theta1


def _run_simul(
    i_sim: int,  # the index of the simulation
    seed: int,  # the seed for its random draws
    n_households_sim: float,  # the number of households in the simulation
    base_functions: np.ndarray,  # the bases
    entropy: EntropyFunctions,  # the entropy
    value_coeff: float,  # the divider
    do_simuls_mde: bool,  # run the MDE simulation
    do_simuls_poisson: bool,  # run the Poisson simulation
    verbose: int = 0,  # how verbose: 1 print simulation number, 2 print steps
) -> MDEResults | PoissonGLMResults | Tuple[MDEResults, PoissonGLMResults]:
    """runs one simulation"""
    global n_simuls_done
    do_both = do_simuls_mde and do_simuls_poisson
    mus_sim = choo_siow_estim.simulate(n_households_sim, seed=seed)
    mus_sim_non0 = remove_zero_cells(mus_sim, coeff=value_coeff)
    if verbose >= 1:
        print(f"Doing simul {i_sim}")
    if do_simuls_mde:
        if verbose == 2:
            print(f"    Doing MDE {i_sim}")
        mde_results_sim = estimate_semilinear_mde(mus_sim_non0, base_functions, entropy)
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


# generate random seeds
rng = np.random.default_rng(130962)
seeds = rng.integers(100_000, size=n_sim)
verbose = 0

entropy = entropy_choo_siow_corrected if use_mde_correction else entropy_choo_siow

# run simulation
list_args = [
    [
        i_sim,
        seeds[i_sim],
        n_households_sim,
        base_functions,
        entropy,
        value_coeff,
        do_simuls_mde,
        do_simuls_poisson,
        verbose,
    ]
    for i_sim in range(n_sim)
]
nb_cpus = 4
n_simuls_done = 0
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
    results_dir / f"{full_model_name}_{n_households_sim}_{int(value_coeff)}.pkl",
    "wb",
) as f:
    pickle.dump(simul_results, f)


if plot_simuls:
    n_households_obs = (
        n_households_cupid_obs
        if full_model_name.startswith("choo_siow_cupid")
        else None
    )
    plot_simulation_results(
        full_model_name,
        n_households_sim,
        n_sim,
        value_coeff,
        do_simuls_mde=do_simuls_mde,
        do_simuls_poisson=do_simuls_poisson,
        n_households_obs=n_households_obs,
    )


seed = 75694
mus_sim = choo_siow_estim.simulate(n_households_sim, seed=seed)
qmus = np.quantile(mus_sim.muxy, np.arange(1, 20) / 20.0)
print("Quantiles of simulated mus:")
print(qmus)

np.sum(mus_sim.mu0y)
