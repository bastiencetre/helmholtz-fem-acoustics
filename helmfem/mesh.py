"""Triangular meshes: generation, geometry filtering and boundary extraction.

A mesh is stored as two arrays:

* ``nodes``     -- (n_nodes, 2) float array of coordinates,
* ``triangles`` -- (n_elems, 3) int array of node indices (counter-clockwise).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Indicator = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class TriMesh:
    nodes: np.ndarray
    triangles: np.ndarray

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elems(self) -> int:
        return self.triangles.shape[0]

    def centroids(self) -> np.ndarray:
        return self.nodes[self.triangles].mean(axis=1)

    def areas(self) -> np.ndarray:
        p = self.nodes[self.triangles]
        e1 = p[:, 1] - p[:, 0]
        e2 = p[:, 2] - p[:, 0]
        return 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])

    def diameters(self) -> np.ndarray:
        """Longest edge of every triangle."""
        p = self.nodes[self.triangles]
        edges = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]], axis=1)
        return np.linalg.norm(edges, axis=2).max(axis=1)

    def mean_h(self) -> float:
        """Average element diameter, used as the mesh size of irregular meshes."""
        return float(self.diameters().mean())

    def boundary_edges(self) -> np.ndarray:
        """Edges that belong to exactly one triangle, as a (n_edges, 2) array."""
        t = self.triangles
        edges = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
        edges = np.sort(edges, axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        return unique[counts == 1]

    def boundary_nodes(self) -> np.ndarray:
        return np.unique(self.boundary_edges())


def square_mesh(nx: int, ny: int, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0) -> TriMesh:
    """Structured mesh of a rectangle: every cell is split along its diagonal."""
    xs = np.linspace(xmin, xmax, nx + 1)
    ys = np.linspace(ymin, ymax, ny + 1)
    X, Y = np.meshgrid(xs, ys)
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    i, j = i.ravel(), j.ravel()
    sw = j * (nx + 1) + i
    se = sw + 1
    nw = sw + (nx + 1)
    ne = nw + 1
    lower = np.column_stack([sw, se, ne])
    upper = np.column_stack([sw, ne, nw])
    triangles = np.vstack([lower, upper])
    return TriMesh(nodes, triangles)


def shifted_square_mesh(nx: int, ny: int, shift_ratio: float = 0.30) -> TriMesh:
    """Unit-square mesh where every odd row of interior nodes is shifted along x.

    The shift is deterministic (``shift_ratio * h``) so two runs give the same mesh.
    Boundary nodes are left in place so the domain is still the unit square.
    """
    mesh = square_mesh(nx, ny)
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    row = np.rint(y * ny).astype(int)
    on_boundary = np.isclose(x, 0) | np.isclose(x, 1) | np.isclose(y, 0) | np.isclose(y, 1)
    shifted = (~on_boundary) & (row % 2 == 1)
    mesh.nodes[shifted, 0] += shift_ratio / nx
    return mesh


def square_boundary_nodes(mesh: TriMesh, tol: float = 1e-12) -> np.ndarray:
    """Nodes lying on the border of the unit square."""
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    on = (np.abs(x) < tol) | (np.abs(x - 1) < tol) | (np.abs(y) < tol) | (np.abs(y - 1) < tol)
    return np.flatnonzero(on)


def keep_elements(mesh: TriMesh, inside: Indicator) -> TriMesh:
    """Keep the triangles whose centroid satisfies ``inside(x, y)``, then drop unused nodes.

    This is how the resonator and pre-fractal geometries are carved out of a square mesh.
    """
    c = mesh.centroids()
    kept = mesh.triangles[inside(c[:, 0], c[:, 1])]
    used = np.unique(kept)
    old_to_new = np.full(mesh.n_nodes, -1, dtype=int)
    old_to_new[used] = np.arange(used.size)
    return TriMesh(mesh.nodes[used], old_to_new[kept])


# --- Geometries ---------------------------------------------------------------------------

def sine_prefractal(level: int, amplitude=0.15, decay=0.55, base_frequency=6) -> Indicator:
    """Domain under y = 1 - sum_j amplitude * decay^j * sin(2 pi m0 2^j x), j < level."""

    def top(x):
        y = np.ones_like(x, dtype=float)
        amp, freq = amplitude, base_frequency
        for _ in range(level):
            y -= amp * np.sin(2 * np.pi * freq * x)
            amp *= decay
            freq *= 2
        return y

    return lambda x, y: (y <= top(x)) & (x >= 0) & (x <= 1) & (y >= 0)


def helmholtz_resonator(wall_x=0.65, thickness=0.04, neck_center=0.5, neck_height=0.06):
    """Square split by a vertical wall with a single opening (the neck).

    Returns ``(inside, in_cavity)`` indicators; the cavity is everything right of the wall.
    The wall must be at least two mesh cells thick: elements are removed by centroid, so a
    thinner strip leaves triangles that still connect both sides and the wall leaks.
    """
    y1, y2 = neck_center - neck_height / 2, neck_center + neck_height / 2

    def inside(x, y):
        in_wall = (x >= wall_x) & (x <= wall_x + thickness) & ~((y >= y1) & (y <= y2))
        return (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1) & ~in_wall

    def in_cavity(x, y):
        return (x > wall_x + thickness) & (y >= 0) & (y <= 1)

    return inside, in_cavity


def double_resonator():
    """Two walls in series, each with its own neck."""
    inside1, cavity1 = helmholtz_resonator(0.55, 0.04, 0.35, 0.06)
    inside2, cavity2 = helmholtz_resonator(0.75, 0.04, 0.70, 0.06)
    return (lambda x, y: inside1(x, y) & inside2(x, y),
            lambda x, y: cavity1(x, y) | cavity2(x, y))


def midpoint_fractal_profile(level: int, base_segments=8, amplitude=0.25, decay=0.5):
    """Pre-fractal top boundary built by midpoint displacement.

    At every level each segment gets a midpoint pushed alternately up and down by an
    amplitude that shrinks geometrically. The profile is normalised to a mean height of 1.
    """
    x = np.linspace(0.0, 1.0, base_segments + 1)
    y = np.ones_like(x)
    for lvl in range(level):
        amp = amplitude * decay**lvl
        signs = np.where(np.arange(x.size - 1) % 2 == 0, 1.0, -1.0)
        xm = 0.5 * (x[:-1] + x[1:])
        ym = np.maximum(0.05, 0.5 * (y[:-1] + y[1:]) + signs * amp)
        x = np.insert(x, np.arange(1, x.size), xm)
        y = np.insert(y, np.arange(1, y.size), ym)
    return x, y / y.mean()


def mesh_under_profile(top_x, top_y, ny=16, refine=2) -> TriMesh:
    """Boundary-fitted mesh of the region between y = 0 and a piecewise-linear profile.

    A structured mesh of the unit square is stretched column by column so that its top row
    lies exactly on the profile. Every profile vertex is a mesh node, so the fractal
    boundary is represented without any geometric error. ``refine`` splits each profile
    segment into that many columns.
    """
    x = np.concatenate([np.linspace(a, b, refine, endpoint=False) for a, b in zip(top_x[:-1], top_x[1:])])
    x = np.append(x, top_x[-1])
    mesh = square_mesh(x.size - 1, ny)
    col = np.rint(mesh.nodes[:, 0] * (x.size - 1)).astype(int)
    mesh.nodes[:, 0] = x[col]
    mesh.nodes[:, 1] *= np.interp(x[col], top_x, top_y)
    return mesh
