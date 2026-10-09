"""Study 4 - Equivalent absorbing boundary condition for four porous wall materials.

For each angular frequency omega, fit the complex Robin coefficient alpha(omega) that best
reproduces the air field of an air + porous layer problem (see helmfem/porous.py), for
melamine foam, glass wool, birch plywood and sandstone. Re(alpha) is the resistive part of
the equivalent surface impedance, Im(alpha) its reactive part; the residual e(alpha) tells
how well a single boundary coefficient can replace the full porous layer.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from helmfem.plotting import save  # noqa: E402
from helmfem.porous import MATERIALS, fit_alpha  # noqa: E402

LABELS = {"melamine_foam": "melamine foam", "glass_wool": "glass wool",
          "birch_lt": "birch plywood", "sandstone": "sandstone"}


def main(out: Path, n_freq: int):
    omegas = np.linspace(2 * np.pi, 10_000 * np.pi, n_freq)
    freqs = omegas / (2 * np.pi)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    print(f"{'material':>15} | {'alpha at 1 kHz':>22} | max residual")
    for name, material in MATERIALS.items():
        fits = [fit_alpha(material, w) for w in omegas]
        alpha = np.array([a for a, _ in fits])
        err = np.array([e for _, e in fits])
        axes[0].plot(freqs, alpha.real, label=LABELS[name])
        axes[1].plot(freqs, alpha.imag, label=LABELS[name])
        axes[2].semilogy(freqs, err, label=LABELS[name])
        a_1k = fit_alpha(material, 2 * np.pi * 1000)[0]
        print(f"{LABELS[name]:>15} | {a_1k.real:9.2f} {a_1k.imag:+9.2f} i | {err.max():.1e}")

    for ax, title in zip(axes, (r"resistive part Re($\alpha$)", r"reactive part Im($\alpha$)",
                                r"fit residual e($\alpha$)")):
        ax.set_title(title)
        ax.set_xlabel("frequency [Hz]")
        ax.grid(True, ls=":", alpha=0.6)
    axes[0].legend()
    fig.suptitle("Equivalent boundary coefficient of a 1 cm porous layer", fontweight="bold")
    save(fig, out / "study4_porous_alpha.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--n-freq", type=int, default=40)
    args = parser.parse_args()
    main(args.out, args.n_freq)
