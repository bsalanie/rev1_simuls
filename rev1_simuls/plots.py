import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cupid_matching.choo_siow import entropy_choo_siow
from cupid_matching.matching_utils import Matching
from cupid_matching.model_classes import ChooSiowPrimitives

from rev1_simuls.utils import results_dir


def _discard_outliers(
    betas: np.ndarray,  # simulation estimates
    method: str,  # name of estimator
    nstd: float = 4.0,  # number of standard deviations allowed
) -> np.ndarray:
    """discards simulation outliers"""
    n_sim = betas.shape[0]
    m, s = np.mean(betas, 0), np.std(betas, 0)
    outliers = np.any(
        abs(betas - m) > nstd * s, 1
    )  # True if simulation has an outlier
    n_outliers = np.sum(outliers)
    print(
        f"""
    We have a total of {n_outliers} outliers for {method}
    out of {n_sim} simulations.
    """
    )
    return outliers


def _dataframe_results(
    full_model_name: str,  #  the model we simulate
    base_names: List[str],  #  the names of the bases
    estims: List[np.ndarray],  #  the simulation results
    do_simuls_mde: bool = True,  #  whether we simulate MDE
    do_simuls_poisson: bool = True,  #  whether we simulate Poisson
) -> None:
    """constructs the dataframe to plot the simulation results"""
    do_both = do_simuls_mde and do_simuls_poisson
    n_kept, n_bases = estims[0].shape
    nkb = n_kept * n_bases
    n_estims = len(estims)
    nekb = n_estims * nkb
    simulation = np.zeros(nekb)
    i = 0
    for i_sim in range(n_kept):
        i_sim_vec = np.full(n_bases, i_sim)
        for i_e in range(n_estims):
            simulation[(i_e * nkb + i) : (i_e * nkb + i + n_bases)] = i_sim_vec
        i += n_bases

    if full_model_name.startswith(
        "choo_siow_cupid"
    ):  # we have an 'Expected' curve in the plot
        estimator_names = ["Expected"]
        estimates = estims[0].reshape(nkb)
        if do_simuls_mde:
            estimator_names += ["MDE"]
            estimates = np.concatenate((estimates, estims[1].reshape(nkb)))
            if do_simuls_poisson:
                estimator_names += ["Poisson"]
                estimates = np.concatenate((estimates, estims[2].reshape(nkb)))
        elif do_simuls_poisson:
            estimator_names += ["Poisson"]
            estimates = np.concatenate((estimates, estims[1].reshape(nkb)))
    else:  # firstsub has no 'Expected' curve
        if do_simuls_mde:
            estimator_names = ["MDE"]
            estimate = estims[0].reshape(nkb)
            if do_simuls_poisson:
                estimator_names += ["Poisson"]
                estimate = np.concatenate((estimate, estims[1].reshape(nkb)))
        elif do_simuls_poisson:
            estimator_names = ["Poisson"]
            estimate = estims[0].reshape(nkb)

    estimator = np.repeat(np.array(estimator_names), nkb)
    coefficient_names = base_names * n_kept * n_estims

    return pd.DataFrame(
        {
            "Simulation": simulation,
            "Estimator": estimator,
            "Parameter": coefficient_names,
            "Estimate": estimates,
        }
    )


def plot_simulation_results(
    full_model_name: str,  # the type of model we are estimating
    n_households_sim: float,  # the number of households in the simulation
    n_sim: int,  # the number of simulation runs
    value_coeff: int,  # the divider of the smallest positive mu
    do_simuls_mde: bool = True,  # do we simulate MDE
    do_simuls_poisson: bool = True,  # do we simulate Poisson
    n_households_obs: float = None,  # the number of observed households in the Cupid dataset
) -> None:
    """plots the simulation results"""
    results_file = (
        results_dir
        / f"{full_model_name}_{n_households_sim}_{int(value_coeff)}.pkl"
    )
    with open(results_file, "rb") as f:
        results = pickle.load(f)
    true_coeffs = results["True coeffs"]
    n_bases = true_coeffs.size
    if full_model_name.startswith("choo_siow_cupid"):
        varcov_coeffs = results["Cupid varcov"]
        varcov_rescaled = varcov_coeffs * n_households_obs / n_households_sim

    base_names = results["Base names"]

    outliers_mask = [False] * n_sim
    if do_simuls_mde:
        estim_mde = results["MDE"]
        outliers_mde = _discard_outliers(estim_mde, "MDE", nstd=4.0)
        outliers_mask = outliers_mask | outliers_mde
    if do_simuls_poisson:
        estim_poisson = results["Poisson"]
        outliers_poisson = _discard_outliers(
            estim_poisson, "Poisson", nstd=4.0
        )
        outliers_mask = outliers_mask | outliers_poisson

    kept = [True] * n_sim
    if any(outliers_mask):
        n_discards = 0
        for i in range(n_sim):
            if outliers_mask[i]:
                kept[i] = False
                n_discards += 1
        print(f"We are discarding {n_discards} outlier samples")
    else:
        print(f"We have found no outlier samples")

    if full_model_name.startswith("choo_siow_cupid"):
        rng = np.random.default_rng(67569)
        n_kept = len(kept)
        nkb = n_kept * n_bases
        expected = np.zeros((n_kept, n_bases))
        for i_sim in range(n_kept):
            expected[i_sim, :] = rng.multivariate_normal(
                mean=true_coeffs, cov=varcov_rescaled
            )
        estims = [expected]
        if do_simuls_mde:
            estims.append(estim_mde)
            if do_simuls_poisson:
                estims.append(estim_poisson)
        elif do_simuls_poisson:
            estims.append(estim_poisson)
    else:
        if do_simuls_mde:
            estims = estim_mde
            if do_simuls_poisson:
                estims.append(estim_poisson)
        elif do_simuls_poisson:
            estims = estim_poisson

    df_simul_results = _dataframe_results(
        full_model_name,
        base_names,
        estims,
        do_simuls_mde=do_simuls_mde,
        do_simuls_poisson=do_simuls_poisson,
    )

    g = sns.FacetGrid(
        data=df_simul_results,
        sharex=False,
        sharey=False,
        hue="Estimator",
        col="Parameter",
        col_wrap=2,
    )
    g.map(sns.kdeplot, "Estimate")
    #     g.set_xlim([-1.0, 3.0])
    g.set_titles("{col_name}")
    for true_val, ax in zip(true_coeffs, g.axes.ravel()):
        ax.vlines(true_val, *ax.get_ylim(), color="k", linestyles="dashed")
    g.add_legend()

    plt.savefig(
        results_dir
        / f"{full_model_name}_simul_{n_households_sim}_{int(value_coeff)}.png"
    )
