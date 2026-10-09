"""Study 1 - Discretisation error and pollution effect of P1 finite elements.

Problem: -Laplace(u) - k^2 u = 0 on the unit square with u = exp(i k x) on the border, whose
exact solution is the plane wave u*(x, y) = exp(i k x). We measure ||u* - u_h||_L2 when the
mesh size h and the wavenumber k vary, on a regular mesh and on a mesh where every other
row of nodes is shifted by 30 % of a cell.

Expected behaviour: err ~ h^2 at fixed k (optimal P1 order in L2), but err grows much faster
than linearly with k at fixed h: this is the pollution (numerical dispersion) effect.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from scipy.sparse.linalg import spsolve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from helmfem import apply_dirichlet, helmholtz_system, l2_error, shifted_square_mesh, square_boundary_nodes, square_mesh  # noqa: E402
from helmfem.plotting import fit_loglog, plot_convergence, save, tidy_log_axes  # noqa: E402

GRIDS = (16, 32, 64, 128)                    # nx; ny = nx // 4 as in the assignment
WAVENUMBERS = (np.pi, 2 * np.pi, 4 * np.pi, 8 * np.pi)


def solve_plane_wave(mesh, k):
    A, b, _, M = helmholtz_system(mesh, k)
    exact = np.exp(1j * k * mesh.nodes[:, 0])
    border = square_boundary_nodes(mesh)
    A, b = apply_dirichlet(A, b, border, exact[border])
    u = spsolve(A.tocsc(), b)
    return l2_error(M, u, exact)


def mesh_size(mesh, nx, shifted):
    # Regular mesh: h = 1/nx. Shifted mesh: mean element diameter (cells are no longer equal).
    return mesh.mean_h() if shifted else 1.0 / nx


def run(shifted: bool, out: Path):
    make = (lambda nx, ny: shifted_square_mesh(nx, ny)) if shifted else square_mesh
    tag = "shifted" if shifted else "regular"
    print(f"\n=== {tag} mesh ===")

    h_err = []
    for nx in GRIDS:
        mesh = make(nx, nx // 4)
        h_err.append((mesh_size(mesh, nx, shifted), solve_plane_wave(mesh, np.pi)))
    k_err = [(k, solve_plane_wave(make(64, 16), k)) for k in WAVENUMBERS]
    kh_err = {k: [] for k in WAVENUMBERS}
    for nx in GRIDS:
        mesh = make(nx, nx // 4)
        h = mesh_size(mesh, nx, shifted)
        for k in WAVENUMBERS:
            kh_err[k].append((k * h, solve_plane_wave(mesh, k)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    slopes = []
    for ax, data, xlabel in zip(axes[:2], (h_err, k_err),
                                ("mesh size h" if not shifted else "mean element diameter h", "wavenumber k")):
        x, y = map(np.array, zip(*sorted(data)))
        slopes.append(plot_convergence(ax, x, y, "error"))
        ax.set_xlabel(xlabel)
        ax.legend()

    # One curve per k: at equal k*h the error is larger for larger k (pollution).
    all_points = sorted(p for pts in kh_err.values() for p in pts)
    slopes.append(fit_loglog(*map(np.array, zip(*all_points)))[0])
    for k, pts in kh_err.items():
        x, y = map(np.array, zip(*pts))
        axes[2].loglog(x, y, "o-", label=f"k = {k / np.pi:.0f} pi")
    tidy_log_axes(axes[2])
    axes[2].set_xlabel("k h")
    axes[2].legend(title=f"global slope {slopes[2]:.2f}", fontsize=8)

    axes[0].set_ylabel(r"$\|u^* - u_h\|_{L^2}$")
    axes[0].set_title("k = pi, h varies")
    axes[1].set_title("mesh 64 x 16, k varies")
    axes[2].set_title("same k h, different k")
    fig.suptitle(f"P1 error for a plane wave, {tag} mesh", fontweight="bold")
    save(fig, out / f"study1_error_{tag}.png")

    print(f"  slope vs h  : {slopes[0]:.3f}   (theory: 2)")
    print(f"  slope vs k  : {slopes[1]:.3f}   (pollution: faster than k^2)")
    print(f"  slope vs kh : {slopes[2]:.3f}")
    return slopes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    run(shifted=False, out=args.out)
    run(shifted=True, out=args.out)
