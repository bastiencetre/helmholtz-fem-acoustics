"""Study 3 - Controlling sound with geometry: pre-fractal walls and Helmholtz resonators.

Domain carved out of the unit square, Dirichlet u = 0 on the left side x = 0, homogeneous
Neumann (rigid wall) everywhere else, Gaussian point-like source in the air:

    -Laplace(u) - k^2 u = f.

1. Pre-fractal top wall (sum of sines, levels 1 and 2): the irregular boundary breaks the
   regular standing-wave pattern of the square.
2. Helmholtz resonator (thin wall with one neck): we sweep k and measure the share of the
   acoustic energy trapped in the cavity. Peaks line up with the Laplace eigenvalues whose
   modes live inside the cavity, which we compute with a shift-invert Lanczos solver.
3. Two resonators in series and a midpoint-displacement fractal wall.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import scipy.sparse.linalg as spla

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from helmfem import apply_dirichlet, helmholtz_system, keep_elements, mass_matrix, square_mesh, stiffness_matrix  # noqa: E402
from helmfem.mesh import (double_resonator, helmholtz_resonator, mesh_under_profile,  # noqa: E402
                          midpoint_fractal_profile, sine_prefractal)
from helmfem.plotting import plot_field, plot_mesh, save  # noqa: E402


def gaussian(x0, y0, sigma=0.03):
    return lambda x, y: np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma**2)


def left_side(mesh, tol=1e-12):
    b = mesh.boundary_nodes()
    return b[np.abs(mesh.nodes[b, 0]) < tol]


def solve(mesh, k, source):
    A, b, _, _ = helmholtz_system(mesh, k, source)
    A, b = apply_dirichlet(A, b, left_side(mesh))
    return spla.spsolve(A.tocsc(), b)


def energy_share(mesh, u, region):
    """Fraction of the L2 energy stored in ``region``: (u^H M_region u) / (u^H M u)."""
    c = mesh.centroids()
    M_region = mass_matrix(mesh, coef=region(c[:, 0], c[:, 1]).astype(float))
    M = mass_matrix(mesh)
    return float(np.real(np.vdot(u, M_region @ u)) / np.real(np.vdot(u, M @ u)))


def laplace_modes(mesh, n_modes):
    """Smallest eigenpairs of K phi = lambda M phi with phi = 0 on x = 0."""
    K, M = stiffness_matrix(mesh), mass_matrix(mesh)
    free = np.setdiff1d(np.arange(mesh.n_nodes), left_side(mesh))
    Kf, Mf = K[free][:, free], M[free][:, free]
    lam, vec = spla.eigsh(Kf.tocsc(), k=n_modes, M=Mf.tocsc(), sigma=0.0, which="LM")
    order = np.argsort(lam)
    modes = np.zeros((mesh.n_nodes, n_modes))
    modes[free] = vec[:, order]
    return lam[order], modes


def prefractal_fields(out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for row, level in enumerate((1, 2)):
        mesh = keep_elements(square_mesh(96, 96), sine_prefractal(level))
        for col, k in enumerate((np.pi, 2 * np.pi, 4 * np.pi)):
            u = solve(mesh, k, gaussian(0.3, 0.5))
            plot_field(mesh, u, f"level {level}, k = {k / np.pi:.0f} pi", ax=axes[row, col])
    fig.suptitle("Re(u) under a pre-fractal top wall (source at x = 0.3)", fontweight="bold")
    save(fig, out / "study3_prefractal_fields.png")


def resonator(out):
    inside, cavity = helmholtz_resonator(wall_x=0.65, thickness=0.04, neck_center=0.5, neck_height=0.06)
    mesh = keep_elements(square_mesh(80, 80), inside)
    source = gaussian(0.35, 0.5)

    lam, modes = laplace_modes(mesh, n_modes=24)
    shares = np.array([energy_share(mesh, modes[:, j], cavity) for j in range(lam.size)])
    trapped = np.flatnonzero(shares > 0.5)
    print("  Laplace modes living in the cavity (k = sqrt(lambda), share of energy):")
    for j in trapped:
        print(f"    k = {np.sqrt(lam[j]):6.3f}   share = {shares[j]:.2f}")

    ks = np.linspace(0.5, 4.5 * np.pi, 400)
    response = [energy_share(mesh, solve(mesh, k, source), cavity) for k in ks]

    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(grid[0, :])
    ax.semilogy(ks, response, color="tab:blue", lw=1.2)
    for j in range(lam.size):
        ax.axvline(np.sqrt(lam[j]), color="tab:red" if j in trapped else "lightgray",
                   lw=1.0 if j in trapped else 0.6, zorder=0)
    ax.set_xlim(ks[0], ks[-1])
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("energy in cavity / total")
    ax.set_title("Frequency sweep: red lines = eigenmodes localised in the cavity, grey = other modes")
    ax.grid(True, which="both", ls=":", alpha=0.6)

    # The lowest trapped mode is the Helmholtz "breathing" mode (air oscillating in the neck);
    # modes with a node at the neck are trapped too but the source barely excites them.
    for i, j in enumerate(trapped[:4]):
        plot_field(mesh, modes[:, j], f"mode {j}: k = {np.sqrt(lam[j]):.2f}, share {shares[j]:.2f}",
                   ax=fig.add_subplot(grid[1, i]))
    fig.suptitle("Helmholtz resonator: wall at x = 0.65 (thickness 0.04) with a 0.06-high neck", fontweight="bold")
    save(fig, out / "study3_resonator.png")


def double_and_fractal(out):
    inside, cavity = double_resonator()
    mesh = keep_elements(square_mesh(80, 80), inside)
    lam, modes = laplace_modes(mesh, n_modes=16)
    shares = [energy_share(mesh, modes[:, j], cavity) for j in range(4)]

    x, y = midpoint_fractal_profile(level=2, base_segments=16, amplitude=0.20, decay=0.6)
    wall = mesh_under_profile(x, y, ny=24, refine=3)
    k = np.pi
    A, b, _, _ = helmholtz_system(wall, k)
    border = wall.boundary_nodes()
    exact = np.exp(1j * k * wall.nodes[:, 0])
    A, b = apply_dirichlet(A, b, border, exact[border])
    u = spla.spsolve(A.tocsc(), b)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for i in range(3):
        plot_field(mesh, modes[:, i], f"two resonators, mode {i} (cavity share {shares[i]:.2f})", ax=axes[0, i])
    plot_mesh(mesh_under_profile(x, y, ny=8, refine=1), axes[1, 0], lw=0.5)  # coarser copy, for readability
    axes[1, 0].set_title("level-2 midpoint fractal wall (boundary-fitted mesh)", fontsize=10)
    plot_field(wall, u.real, "plane wave, Re(u_h)", ax=axes[1, 1])
    plot_field(wall, u.imag, "plane wave, Im(u_h)", ax=axes[1, 2])
    save(fig, out / "study3_double_resonator_and_fractal_wall.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures"))
    out = parser.parse_args().out
    prefractal_fields(out)
    resonator(out)
    double_and_fractal(out)
