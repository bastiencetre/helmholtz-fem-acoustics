"""Study 2 - Krylov solvers for the (indefinite, complex) Helmholtz FEM system.

Same plane-wave problem as study 1. The system A u = b with A = K - k^2 M is complex,
symmetric but not Hermitian, and indefinite as soon as k^2 exceeds the first Laplace
eigenvalue: conjugate gradient does not apply, so we compare non-symmetric Krylov methods.

Fair comparison: SciPy counts "iterations" differently for each method (GMRES inner steps,
LGMRES outer cycles, BiCGStab two products per step...), so we count the number of
matrix-vector products with A needed to reach ||Au - b|| / ||b|| <= 1e-8.

Two preconditioners are added on top of the plain methods:
* ILU  -- incomplete LU of A (scipy spilu, drop tolerance 1e-3, about 3.5x the nnz of A).
  Cheap ILU helps at low frequency but breaks down when A becomes strongly indefinite;
* shifted Laplacian -- exact inverse of K - (1 - 0.5i) k^2 M, the classical Helmholtz
  preconditioner (Erlangga, Vuik & Oosterlee, 2004), here factorised once with a sparse LU.
"""

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import scipy.sparse.linalg as spla

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from helmfem import apply_dirichlet, helmholtz_system, square_boundary_nodes, square_mesh  # noqa: E402
from helmfem.plotting import save, tidy_log_axes  # noqa: E402

RTOL = 1e-8
MATVEC_BUDGET = 20_000
GRIDS = (16, 32, 64)
WAVENUMBERS = (np.pi, 2 * np.pi, 4 * np.pi, 8 * np.pi)


def plane_wave_system(n, k):
    mesh = square_mesh(n, n)
    A, b, K, M = helmholtz_system(mesh, k)
    exact = np.exp(1j * k * mesh.nodes[:, 0])
    border = square_boundary_nodes(mesh)
    A, b = apply_dirichlet(A, b, border, exact[border])
    return A, b, K, M, border


def shifted_laplacian(K, M, k, border, beta=(1.0, -0.5)):
    P = (K - (beta[0] + 1j * beta[1]) * k**2 * M).astype(complex)
    P, _ = apply_dirichlet(P, np.zeros(P.shape[0], complex), border)
    lu = spla.splu(P.tocsc())
    return spla.LinearOperator(P.shape, matvec=lu.solve, dtype=complex)


def ilu(A):
    fact = spla.spilu(A.tocsc(), drop_tol=1e-3, fill_factor=4)
    return spla.LinearOperator(A.shape, matvec=fact.solve, dtype=complex)


SOLVERS = {
    "GMRES(50)": lambda A, b, P: spla.gmres(A, b, rtol=RTOL, atol=0, restart=50, maxiter=MATVEC_BUDGET // 50),
    "LGMRES": lambda A, b, P: spla.lgmres(A, b, rtol=RTOL, atol=0, maxiter=MATVEC_BUDGET // 30),
    "BiCGStab": lambda A, b, P: spla.bicgstab(A, b, rtol=RTOL, atol=0, maxiter=MATVEC_BUDGET // 2),
    "CGS": lambda A, b, P: spla.cgs(A, b, rtol=RTOL, atol=0, maxiter=MATVEC_BUDGET // 2),
    "GMRES + ILU": lambda A, b, P: spla.gmres(A, b, rtol=RTOL, atol=0, restart=50, maxiter=200, M=P["ilu"]),
    "GMRES + shifted Laplacian": lambda A, b, P: spla.gmres(A, b, rtol=RTOL, atol=0, restart=50,
                                                            maxiter=200, M=P["shifted"]),
}


class CountingOperator(spla.LinearOperator):
    """Wraps A and counts how many times it is applied."""

    def __init__(self, A):
        super().__init__(A.dtype, A.shape)
        self.A, self.count = A, 0

    def _matvec(self, x):
        self.count += 1
        return self.A @ x


def benchmark(n, k):
    A, b, K, M, border = plane_wave_system(n, k)
    preconditioners = {"ilu": ilu(A), "shifted": shifted_laplacian(K, M, k, border)}
    rows = {}
    for name, solve in SOLVERS.items():
        op = CountingOperator(A)
        t0 = time.perf_counter()
        x, info = solve(op, b, preconditioners)
        elapsed = time.perf_counter() - t0
        residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        rows[name] = dict(matvecs=op.count, residual=residual, converged=residual <= 10 * RTOL, time=elapsed)
    return rows


def print_table(title, results, key_label):
    print(f"\n{title}")
    names = list(SOLVERS)
    print(f"{key_label:>8} | " + " | ".join(f"{n:>26}" for n in names))
    for key, rows in results.items():
        cells = []
        for n in names:
            r = rows[n]
            mark = "" if r["converged"] else " (no conv.)"
            cells.append(f"{r['matvecs']:>6d} matvecs{mark:>11}"[:26].rjust(26))
        print(f"{key:>8.3f} | " + " | ".join(cells))


def plot(ax, xs, results, xlabel):
    for name in SOLVERS:
        y = [results[x][name]["matvecs"] for x in xs]
        ok = [results[x][name]["converged"] for x in xs]
        line, = ax.semilogy(xs, y, "o-", label=name)
        bad = [(x, v) for x, v, c in zip(xs, y, ok) if not c]
        if bad:
            ax.plot(*zip(*bad), "x", color=line.get_color(), ms=11, mew=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("matrix-vector products to reach 1e-8")
    tidy_log_axes(ax)


def main(out: Path):
    by_h = {1.0 / n: benchmark(n, np.pi) for n in GRIDS}
    print_table("k = pi, mesh refinement (key = h)", by_h, "h")
    by_k = {k: benchmark(64, k) for k in WAVENUMBERS}
    print_table("mesh 64 x 64, increasing frequency (key = k)", by_k, "k")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    plot(axes[0], sorted(by_h), by_h, "mesh size h  (k = pi)")
    axes[0].invert_xaxis()
    plot(axes[1], list(WAVENUMBERS), by_k, "wavenumber k  (mesh 64 x 64)")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Cost of Krylov solvers on the Helmholtz FEM system (x = budget exhausted)", fontweight="bold")
    save(fig, out / "study2_iterative_solvers.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures"))
    main(parser.parse_args().out)
