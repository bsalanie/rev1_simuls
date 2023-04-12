""" configuration for the simulations"""

from pathlib import Path

dropbox_home = Path.home() / "Dropbox"
cupid_pub_dir = dropbox_home / "GalichonSalanie" / "Cupid" / "RestudPubli"
intermediate_data_dir = (
    cupid_pub_dir / "ReplicationPackagev3/Data/Intermediate"
)
output_data_dir = Path("..") / "ChooSiow70nNdata"

do_simuls_mde = True
do_simuls_poisson = True

plot_simuls = True

model_name = "choo_siow_cupid"
sample_size = "small"
# sample_size = "large"

# degrees of polynomials for the base functions
degrees = [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]

age_start, age_end = 16, 40  # we select ages
shrink_factor = (
    1  # we shrink the Choo-Siow estimates by a multiplicative integer
)

n_households_cupid_popu = (
    13_274_041  # number of households in the Cupid population
)
use_mde_correction = False  # use the `corrected` version of MDE

renormalize_true_coeffs = True  # make all true coeffs equal 1

n_sim = 10  # number of simulations
#  if zero_guard is not 0,
#     we set the zero cells at the size of the smallest positive cell
#      divided by zero_guard.
zero_guard = 1
