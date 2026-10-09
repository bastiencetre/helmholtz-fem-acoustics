import numpy as np

from helmfem.porous import MATERIALS, AbsorptionModel, fit_alpha


def test_error_is_zero_for_unexcited_modes():
    model = AbsorptionModel(MATERIALS["glass_wool"], omega=2000.0)
    e = model.modal_errors(20 - 30j)
    assert np.all(e[model.k != 0] == 0)


def test_fit_alpha_reaches_a_small_residual():
    alpha, err = fit_alpha(MATERIALS["melamine_foam"], omega=2 * np.pi * 1000)
    assert err < 1e-5
    assert alpha.real > 0 and alpha.imag < 0
