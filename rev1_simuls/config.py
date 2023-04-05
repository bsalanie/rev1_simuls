""" configuration for the simulations"""

renormalize_true_coeffs = True  # renormalize so that all "true" coeffs = 1

do_simuls_mde = True
do_simuls_poisson = True

plot_simuls = True

# model_name = "choo_siow_firstsub"
model_name = "choo_siow_cupid"
degrees = [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
age_start, age_end = 16, 40  # we select ages
shrink_factor = (
    1  # we shrink the Choo-Siow estimates by a multiplicative integer
)
n_households_cupid_pop = (
    13_274_041  # number of households in the Cupid population
)
n_households_cupid_obs = 75_265  # number of households in the Cupid sample

use_rescale = False  # rescale the sample
use_mde_correction = False  # use the `corrected` version of MDE

#  number of households in the simulation:
n_households_sim = n_households_cupid_obs
# n_households_sim =  1_000_000
# n_households_sim =  n_households_cupid_pop

n_sim = 5  # number of simulations
zero_guard = 1  # we set the zeros at the smallest positive value
# divided by zero_guard,
# except if zero_guard is 0
