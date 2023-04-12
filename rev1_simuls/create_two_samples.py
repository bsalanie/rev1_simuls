import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import pickle

from rev1_simuls.utils import bs_error_abort, VarianceMatching
from rev1_simuls.config import intermediate_data_dir, output_data_dir


def make_marriages(
    age_start: int, age_end: int, husband_reform: int, wife_reform: int
) -> List:
    """counts marriages by age cells

    with partners of ages between age_start and age_end at the date of the Census,
    and in states with a given reform status
    """

    marriages = pd.read_csv(
        intermediate_data_dir / "ChooSiowMarriages.csv", dtype=int
    )
    # we need to de-age each partner by 1 or 2 years, depending on when the marriage is observed
    year_observed = marriages["year"].values
    aging_factor = year_observed - 10 * np.floor(year_observed / 10)
    marriages["husband_age"] = marriages["husbagemarr"] - aging_factor
    marriages["wife_age"] = marriages["wifeagemarr"] - aging_factor

    n_ages = age_end - age_start + 1
    ages_range = range(age_start, age_end + 1)
    # only if both partners are in a non-reform state
    state_selection = (marriages["husbreform"] == husband_reform) & (
        marriages["wifereform"] == wife_reform
    )
    age_selection = (marriages["husband_age"].isin(ages_range)) & (
        marriages["wife_age"].isin(ages_range)
    )
    marriages_sel = marriages[state_selection & age_selection]

    # go through age cells
    n_marriages = np.zeros((n_ages, n_ages))
    n_weighted_marriages = np.zeros((n_ages, n_ages))
    sumweights_sq = np.zeros((n_ages, n_ages))
    i_age = 0
    for h_age in ages_range:
        marr_h = marriages_sel[marriages_sel["husband_age"] == h_age]
        if marr_h.shape[0] > 0:
            j_age = 0
            for w_age in ages_range:
                marr_hw = marr_h[marr_h["wife_age"] == w_age]
                if marr_hw.shape[0] > 0:
                    weights = marr_hw["samplingweight"].values
                    n_weighted_marriages[i_age, j_age] = np.sum(weights)
                    n_marriages[i_age, j_age] = marr_hw.shape[0]
                    sumweights_sq[i_age, j_age] = np.sum(weights * weights)
                j_age += 1
        i_age += 1

    return n_weighted_marriages, n_marriages, sumweights_sq


def make_availables70n(age_start: int, age_end: int) -> np.ndarray:
    """create the series for available men and women in this age range.

    here we do it only for the (1970, non-reform) subset we use
    """
    # we need these next two to correct the number of availables
    n_weighted_marriages_70nR, n_marriages_70nR, sumw2_70nR = make_marriages(
        16, 40, 0, 1
    )
    n_weighted_marriages_70rN, n_marriages_70rN, sumw2_70rN = make_marriages(
        16, 40, 1, 0
    )

    availables = pd.read_csv(
        intermediate_data_dir / "ChooSiowAvailables.csv", dtype=int
    )

    n_ages = age_end - age_start + 1
    ages_range = list(range(age_start, age_end + 1))

    # only if the person is in a non-reform state
    state_selection = availables["reform"] == 0
    age_selection = availables["age"].isin(ages_range)
    availables_sel = availables[state_selection & age_selection]

    n_available_men = np.zeros(n_ages)
    n_available_women = np.zeros(n_ages)
    n_weighted_available_men = np.zeros(n_ages)
    n_weighted_available_women = np.zeros(n_ages)
    sumweights_sq_men = np.zeros(n_ages)
    sumweights_sq_women = np.zeros(n_ages)

    i_age = 0
    for age in ages_range:
        available_age = availables_sel[availables_sel["age"] == age]
        available_h = available_age[available_age["sex"] == 1]
        weights_h = available_h["weight"].values
        # we need to subtract men who married a woman from a reform state
        n_available_men[i_age] = available_h.shape[0] - np.sum(
            n_marriages_70nR[i_age, :]
        )
        n_weighted_available_men[i_age] = np.sum(weights_h) - np.sum(
            n_weighted_marriages_70nR[i_age, :]
        )
        sumweights_sq_men[i_age] = np.sum(weights_h * weights_h) - np.sum(
            sumw2_70nR[i_age, :]
        )

        available_w = available_age[available_age["sex"] == 2]
        weights_w = available_w["weight"].values
        # we need to subtract women who married a man from a reform state
        n_available_women[i_age] = available_w.shape[0] - np.sum(
            n_marriages_70rN[:, i_age]
        )
        n_weighted_available_women[i_age] = np.sum(weights_w) - np.sum(
            n_weighted_marriages_70rN[:, i_age]
        )
        sumweights_sq_women[i_age] = np.sum(weights_w * weights_w) - np.sum(
            sumw2_70rN[:, i_age]
        )

        i_age += 1

    return (
        n_available_men,
        n_weighted_available_men,
        sumweights_sq_men,
        n_available_women,
        n_weighted_available_women,
        sumweights_sq_women,
    )


def make_sample(sample_size: str) -> None:
    n_weighted_marriages_70nN, n_marriages_70nN, sumw2_70nN = make_marriages(
        16, 40, 0, 0
    )
    (
        n_available_men,
        n_weighted_available_men,
        sumweights_sq_men,
        n_available_women,
        n_weighted_available_women,
        sumweights_sq_women,
    ) = make_availables70n(16, 40)

    average_weight_marriages = np.sum(n_weighted_marriages_70nN) / np.sum(
        n_marriages_70nN
    )
    average_weight_availables = (
        np.sum(n_weighted_available_men) + np.sum(n_weighted_available_women)
    ) / (np.sum(n_available_men) + np.sum(n_available_women))
    relative_weight_marriages = (
        average_weight_marriages / average_weight_availables
    )

    if sample_size == "large":
        # we inflate availables
        n_available_men = (n_available_men / relative_weight_marriages).astype(
            int
        )
        n_available_women = (
            n_available_women / relative_weight_marriages
        ).astype(int)
        n_marriages = n_marriages_70nN.astype(int)
    elif sample_size == "small":
        # we deflate marriages
        n_marriages = (n_marriages_70nN * relative_weight_marriages).astype(
            int
        )
        n_available_women = n_available_women.astype(int)
        n_available_men = n_available_men.astype(int)
    else:
        bs_error_abort(f"sample_size = {sample_size} is invalid.")

    print(f"Making a {sample_size} sample:")
    print(f"   with {np.sum(n_marriages)} marriages")
    print(f"     {np.sum(n_available_men)} available men")
    print(f"     and {np.sum(n_available_women)} available women.")
    np.savetxt(output_data_dir / f"{sample_size}_muxy.txt", n_marriages)
    np.savetxt(output_data_dir / f"{sample_size}_nx.txt", n_available_men)
    np.savetxt(output_data_dir / f"{sample_size}_my.txt", n_available_women)

    # we also need to compute the variance-covariance of the matching patterns
    # again, we only do it for the 1970 Census in non-reform states

    # total number of households in the sample
    n_households_obs = (
        np.sum(n_available_men)
        + np.sum(n_available_women)
        - np.sum(n_marriages)
    )
    print(f"We have a total of {n_households_obs} households.")

    n_ages = n_available_men.size
    n_ages2 = n_ages * n_ages

    # matching patterns
    muxy = n_marriages
    mux0 = n_available_men - np.sum(muxy, 1)
    mu0y = n_available_women - np.sum(muxy, 0)

    # the variance-covariance matrix
    var_xyzt = np.zeros((n_ages2, n_ages2))
    var_xyz0 = np.zeros((n_ages2, n_ages))
    var_xy0t = np.zeros((n_ages2, n_ages))
    var_x0z0 = np.zeros((n_ages, n_ages))
    var_x00t = np.zeros((n_ages, n_ages))
    var_0y0t = np.zeros((n_ages, n_ages))

    muxy_vec = muxy.reshape(n_ages2)

    for x in range(n_ages):
        for y in range(n_ages):
            xy = x * n_ages + y
            vari = -muxy[x, y] * muxy_vec / n_households_obs
            vari[xy] += muxy[x, y]
            var_xyzt[xy, :] = vari
            var_xyz0[xy, :] = -muxy[x, y] * mux0 / n_households_obs
            var_xy0t[xy, :] = -muxy[x, y] * mu0y / n_households_obs
    for x in range(n_ages):
        var_x0z0[x, :] = -mux0[x] * mux0 / n_households_obs
        var_x0z0[x, x] += mux0[x]
        var_x00t[x, :] = -mux0[x] * mu0y / n_households_obs
    for y in range(n_ages):
        var_0y0t[y, :] = -mu0y[y] * mu0y / n_households_obs
        var_0y0t[y, y] += mu0y[y]

    varmus = VarianceMatching(
        var_xyzt=var_xyzt,
        var_xyz0=var_xyz0,
        var_xy0t=var_xy0t,
        var_x0z0=var_x0z0,
        var_x00t=var_x00t,
        var_0y0t=var_0y0t,
    )

    with open(output_data_dir / f"{sample_size}_varmus.pkl", "wb") as f:
        pickle.dump(varmus, f)

    quantiles = np.arange(1, 100) / 100.0
    q_vals = np.quantile(n_marriages, quantiles)

    with open(
        output_data_dir / f"{sample_size}_quantiles_marriages.txt", "w"
    ) as f:
        for q, qval in zip(quantiles, q_vals):
            f.write(f"Quantile {q: .3f}: {int(qval): >10d}\n")
            print(f"Quantile {q: .3f}: {int(qval): >10d}")
    return


if __name__ == "__main__":
    make_sample("large")
    make_sample("small")
