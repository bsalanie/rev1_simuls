""" configuration for the simulations"""

do_simuls_mde = True
do_simuls_poisson = True

plot_simuls = True

model_name = "choo_siow_cupid"
# model_name = "choo_siow_firstsub"

degrees = [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
age_start, age_end = 16, 40  # we select ages
shrink_factor = (
    1  # we shrink the Choo-Siow estimates by a multiplicative integer
)


n_households_cupid_popu = (
    13_274_041  # number of households in the Cupid population
)
n_households_cupid_sample = 75_265  # number of households in the Cupid sample

use_rescale = False  # rescale the sample
use_mde_correction = False  # use the `corrected` version of MDE

renormalize_true_coeffs = True  # make all true coeffs equal 1

#  number of households in the simulation:
n_households_sim = n_households_cupid_sample
# n_households_sim =  1_000_000
# n_households_sim =  n_households_cupid_popu

n_sim = 5  # number of simulations
#  if zero_guard is not 0,
#     we set the zero cells at the size of the smallest positive cell
#      divided by zero_guard.
zero_guard = 1
