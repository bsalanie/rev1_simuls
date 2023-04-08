import numpy as np

from rev1_simuls.utils import print_quantiles


def test_print_quantiles_vec():
    v = np.arange(101.0)
    q = np.arange(10.0) / 10.0
    qvals = print_quantiles(v, q)
    qvals_expected = np.arange(start=0.0, stop=100.0, step=10.0)
    assert np.allclose(qvals, qvals_expected)


def test_print_quantiles_list():
    v1 = np.arange(101.0)
    v2 = np.arange(101.0, 202.0)
    lst = [v1, v2]
    q = np.arange(10.0) / 10.0
    qvals = print_quantiles(lst, q)
    qvals_expected = np.column_stack(
        (
            np.arange(start=0.0, stop=100.0, step=10.0),
            np.arange(start=101.0, stop=200.0, step=10.0),
        )
    )
    assert np.allclose(qvals, qvals_expected)
