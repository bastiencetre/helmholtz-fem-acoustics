import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from helmfem import (apply_dirichlet, helmholtz_system, keep_elements, l2_error, mass_matrix,
                     shifted_square_mesh, square_boundary_nodes, square_mesh, stiffness_matrix)
from helmfem.mesh import helmholtz_resonator, mesh_under_profile, midpoint_fractal_profile


def test_square_mesh_counts():
    mesh = square_mesh(4, 3)
    assert mesh.n_nodes == 5 * 4
    assert mesh.n_elems == 2 * 4 * 3
    assert np.isclose(mesh.areas().sum(), 1.0)
    assert len(mesh.boundary_nodes()) == 2 * (4 + 3)


def test_mass_matrix_integrates_constants():
    mesh = shifted_square_mesh(8, 8)
    M = mass_matrix(mesh)
    ones = np.ones(mesh.n_nodes)
    assert np.isclose(ones @ M @ ones, 1.0)


def test_stiffness_kills_constants_and_is_symmetric():
    mesh = square_mesh(6, 5)
    K = stiffness_matrix(mesh)
    assert np.allclose(K @ np.ones(mesh.n_nodes), 0.0)
    assert abs(K - K.T).max() < 1e-12


def test_stiffness_energy_of_linear_function():
    # For u = x, integral |grad u|^2 = area = 1, and P1 reproduces linear functions exactly.
    mesh = shifted_square_mesh(7, 7)
    u = mesh.nodes[:, 0]
    assert np.isclose(u @ stiffness_matrix(mesh) @ u, 1.0)


def test_dirichlet_reproduces_boundary_values():
    mesh = square_mesh(5, 5)
    A, b, _, _ = helmholtz_system(mesh, k=1.0)
    nodes = square_boundary_nodes(mesh)
    A, b = apply_dirichlet(A, b, nodes, values=2.0)
    u = np.linalg.solve(A.toarray(), b)
    assert np.allclose(u[nodes], 2.0)


@pytest.mark.parametrize("mesh_fn", [square_mesh, shifted_square_mesh])
def test_plane_wave_converges_at_order_two(mesh_fn):
    k, errors, sizes = np.pi, [], []
    for n in (16, 32, 64):
        mesh = mesh_fn(n, n // 4)
        A, b, _, M = helmholtz_system(mesh, k)
        exact = np.exp(1j * k * mesh.nodes[:, 0])
        nodes = square_boundary_nodes(mesh)
        A, b = apply_dirichlet(A, b, nodes, exact[nodes])
        u = np.linalg.solve(A.toarray(), b)
        errors.append(l2_error(M, u, exact))
        sizes.append(mesh.mean_h())
    slope = np.polyfit(np.log(sizes), np.log(errors), 1)[0]
    assert 1.8 < slope < 2.3


def test_resonator_geometry_removes_the_wall():
    inside, in_cavity = helmholtz_resonator()
    mesh = keep_elements(square_mesh(40, 40), inside)
    assert mesh.areas().sum() < 1.0
    assert in_cavity(np.array([0.9]), np.array([0.5]))[0]
    assert not inside(np.array([0.66]), np.array([0.1]))[0]


def test_fractal_mesh_area_matches_profile():
    x, y = midpoint_fractal_profile(level=2, base_segments=16, amplitude=0.2, decay=0.6)
    mesh = mesh_under_profile(x, y, ny=12)
    assert np.isclose(mesh.areas().sum(), np.trapezoid(y, x), rtol=1e-6)


def _n_components(mesh):
    n, _ = connected_components(abs(stiffness_matrix(mesh)) > 0, directed=False)
    return n


def test_closed_wall_really_splits_the_domain():
    # Regression test: a wall thinner than two cells used to leak through shared nodes.
    inside, _ = helmholtz_resonator(neck_height=0.0)
    assert _n_components(keep_elements(square_mesh(64, 64), inside)) == 2
    inside, _ = helmholtz_resonator()
    assert _n_components(keep_elements(square_mesh(64, 64), inside)) == 1
