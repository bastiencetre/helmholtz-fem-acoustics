"""Equivalent absorbing boundary condition for a porous wall.

Air (medium 0) is in contact with a porous layer of thickness L (medium 1, equivalent
fluid model). We look for the complex coefficient alpha of a Robin condition

    mu_0 du/dn + alpha u = 0      on the wall,

such that the air field of the simplified problem stays as close as possible to the
air field of the full air + porous problem. Decomposing both problems in Fourier modes
k along the wall gives a closed-form error e(alpha) = sum_k e_k(alpha), which is then
minimised numerically for every angular frequency omega.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class PorousMaterial:
    name: str
    porosity: float          # phi
    resistivity: float       # sigma [Pa.s/m^2]
    tortuosity: float        # alpha_h
    gamma_p: float = 7.0 / 5.0
    rho_0: float = 1.2       # air density [kg/m^3]
    c_0: float = 340.0       # speed of sound in air [m/s]


MATERIALS = {
    m.name: m
    for m in [
        PorousMaterial("melamine_foam", porosity=0.99, resistivity=14_000.0, tortuosity=1.02),
        PorousMaterial("glass_wool", porosity=0.97, resistivity=15_000.0, tortuosity=1.80),
        PorousMaterial("birch_lt", porosity=0.529, resistivity=151_429.0, tortuosity=1.37),
        PorousMaterial("sandstone", porosity=0.25, resistivity=1.0e6, tortuosity=1.60),
    ]
}


def plane_wave_excitation(k: np.ndarray) -> np.ndarray:
    """Only the k = 0 mode (normal incidence) is excited."""
    return (k == 0).astype(float)


class AbsorptionModel:
    """Closed-form modal error e(alpha) for one material and one angular frequency."""

    def __init__(self, material: PorousMaterial, omega: float, thickness=0.01,
                 n_modes=12, weight_a=1.0, weight_b=1.0, excitation=plane_wave_excitation):
        m = material
        self.L = thickness
        self.A, self.B = weight_a, weight_b
        self.mu0, self.xi0 = 1.0, 1.0 / m.c_0**2
        self.mu1 = m.porosity / m.tortuosity
        self.xi1 = m.porosity * m.gamma_p / m.c_0**2
        self.a = m.resistivity * m.porosity**2 * m.gamma_p / (m.c_0**2 * m.rho_0 * m.tortuosity)

        self.k = np.arange(-n_modes, n_modes + 1) * np.pi / thickness
        self.g = excitation(self.k)
        self.evanescent = self.k**2 >= omega**2 * self.xi0 / self.mu0
        self.lam0 = self._lambda0(omega)
        self.lam1 = self._lambda1(omega)

    def _lambda0(self, omega):
        """Vertical propagation constant in air (real if evanescent, imaginary otherwise)."""
        d = self.k**2 - omega**2 * self.xi0 / self.mu0
        return np.where(self.evanescent, np.sqrt(np.abs(d)), 1j * np.sqrt(np.abs(d)))

    def _lambda1(self, omega):
        """Complex propagation constant in the dissipative porous medium."""
        t1 = omega**2 * self.xi1 / self.mu1
        t2 = np.sqrt((self.k**2 - t1) ** 2 + (self.a * omega / self.mu1) ** 2)
        return (np.sqrt(self.k**2 - t1 + t2) - 1j * np.sqrt(t1 - self.k**2 + t2)) / np.sqrt(2.0)

    def _f(self, x):
        l0, L = self.lam0, self.L
        return (l0 * self.mu0 - x) * np.exp(-l0 * L) + (l0 * self.mu0 + x) * np.exp(l0 * L)

    def modal_errors(self, alpha: complex) -> np.ndarray:
        l0, l1, L, A, B = self.lam0, self.lam1, self.L, self.A, self.B
        f1, fa = self._f(l1 * self.mu1), self._f(alpha)
        chi = self.g * ((l0 * self.mu0 - l1 * self.mu1) / f1 - (l0 * self.mu0 - alpha) / fa)
        eta = self.g * ((l0 * self.mu0 + l1 * self.mu1) / f1 - (l0 * self.mu0 + alpha) / fa)

        exp_m, exp_p = np.exp(-2 * l0 * L), np.exp(2 * l0 * L)
        c2, e2 = np.abs(chi) ** 2, np.abs(eta) ** 2
        weight = A + B * np.abs(self.k) ** 2

        with np.errstate(divide="ignore", invalid="ignore"):
            # Evanescent modes in air: lambda_0 is real.
            growth = c2 * (1 - exp_m) + e2 * (exp_p - 1)
            cross = np.real(chi * np.conj(eta))
            evan = (weight * (growth / (2 * l0) + 2 * L * cross)
                    + B * np.abs(l0) / 2 * growth - 2 * B * l0**2 * L * cross)
            # Propagating modes in air: lambda_0 is purely imaginary.
            osc = np.imag(chi * np.conj(eta * (1 - exp_m)))
            prop = (weight * (L * (c2 + e2) + 1j / l0 * osc)
                    + B * L * np.abs(l0) ** 2 * (c2 + e2) + 1j * B * l0 * osc)
        return np.where(self.evanescent, evan, prop)

    def error(self, alpha: complex) -> float:
        return float(np.real(np.sum(self.modal_errors(alpha))))


def fit_alpha(material: PorousMaterial, omega: float, alpha0=40.0 - 40.0j, **kwargs):
    """Minimise e(alpha) over the complex plane. Returns (alpha, e(alpha))."""
    model = AbsorptionModel(material, omega, **kwargs)
    objective = lambda z: model.error(complex(z[0], z[1]))
    res = minimize(objective, [alpha0.real, alpha0.imag], tol=1e-4)
    alpha = complex(res.x[0], res.x[1])
    return alpha, model.error(alpha)
