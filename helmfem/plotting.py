"""Small plotting helpers shared by the study scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from .mesh import TriMesh


def fit_loglog(x, y):
    """Least-squares fit of log(y) = slope * log(x) + intercept."""
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    return float(slope), float(intercept)


def plot_convergence(ax, x, y, label, show_fit=True):
    """Log-log curve plus the dashed regression line, slope written in the legend."""
    slope, intercept = fit_loglog(x, y)
    ax.loglog(x, y, "o-", label=f"{label} (slope {slope:.2f})")
    if show_fit:
        x = np.asarray(x, dtype=float)
        ax.loglog(x, np.exp(intercept) * x**slope, "--", lw=1, color="gray")
    tidy_log_axes(ax)
    return slope


def tidy_log_axes(ax):
    """Grid on, and no labels on minor ticks (they overlap on short log ranges)."""
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())


def plot_field(mesh: TriMesh, values, title, path=None, symmetric=True, ax=None):
    """Colour map of a nodal field."""
    tri = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.triangles)
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    values = np.real(values)
    vmax = max(float(np.abs(values).max()), 1e-12) if symmetric else None
    kw = dict(vmin=-vmax, vmax=vmax, cmap="RdBu_r") if symmetric else dict(cmap="viridis")
    im = ax.tripcolor(tri, values, shading="gouraud", **kw)
    ax.figure.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if own_figure:
        save(ax.figure, path)
    return ax


def plot_mesh(mesh: TriMesh, ax, color="tab:orange", lw=0.3):
    ax.triplot(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.triangles, color=color, lw=lw)
    ax.set_aspect("equal")
    ax.axis("off")


def save(fig, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)
