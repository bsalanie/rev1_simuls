"""check the quality of the nonparametric estimatites of Phi"""

import pickle
import numpy as np

from rev1_simuls.utils import data_dir, results_dir, print_quantiles

for sample_size in ["small", "large"]:
    full_model_name = f"choo_siow_cupid_{sample_size}"

    with open(data_dir / f"{full_model_name}_phi_true.pkl", "rb") as f:
        phi_true = pickle.load(f)

    with open(results_dir / f"{full_model_name}_1.pkl", "rb") as f:
        simul_results = pickle.load(f)

    phi_est = simul_results["Phi non param"]

    mean_error = np.mean(phi_est, 0) - phi_true
    mean_rel_error = mean_error/phi_true

    print("Quantiles of the mean error (level and relative) of phi non param:")
    print_quantiles((mean_error.flatten(), mean_rel_error.flatten()),
                    np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))



        
