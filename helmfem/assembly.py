"""P1 finite elements on triangles: sparse assembly and Dirichlet conditions.

For the Helmholtz problem  -div(grad u) - k^2 u = f  the weak form gives the linear
system  (K - k^2 M) U = M F  where K is the stiffness matrix and M the mass matrix.
Everything is vectorised over elements, so meshes with 10^5 nodes assemble in well
under a second.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .mesh import TriMesh

# Gradients of the three reference basis functions on the unit triangle.
_REF_GRADS = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
# Exact P1 mass matrix of a triangle of area 1.
_REF_MASS = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 12.0


def _element_geometry(mesh: TriMesh):
    p = mesh.nodes[mesh.triangles]                       # (ne, 3, 2)
    jac = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]], axis=2)  # columns = edges
    det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
    area = 0.5 * np.abs(det)
    inv_jac = np.linalg.inv(jac)                          # (ne, 2, 2)
    grads = _REF_GRADS @ inv_jac                          # (ne, 3, 2)
    return area, grads


def _scatter(mesh: TriMesh, local: np.ndarray) -> sp.csr_matrix:
    rows = np.repeat(mesh.triangles, 3, axis=1).ravel()
    cols = np.tile(mesh.triangles, (1, 3)).ravel()
    n = mesh.n_nodes
    return sp.csr_matrix((local.ravel(), (rows, cols)), shape=(n, n))


def stiffness_matrix(mesh: TriMesh, coef=None) -> sp.csr_matrix:
    """K_ij = sum_T coef_T * integral_T grad(phi_i) . grad(phi_j)."""
    area, grads = _element_geometry(mesh)
    local = area[:, None, None] * grads @ grads.transpose(0, 2, 1)
    if coef is not None:
        local = local * np.asarray(coef)[:, None, None]
    return _scatter(mesh, local)


def mass_matrix(mesh: TriMesh, coef=None) -> sp.csr_matrix:
    """M_ij = sum_T coef_T * integral_T phi_i phi_j.

    ``coef`` is an optional per-element weight; a 0/1 weight gives the mass matrix of a
    sub-region, which is how the energy stored in a resonator cavity is measured.
    """
    area, _ = _element_geometry(mesh)
    local = area[:, None, None] * _REF_MASS[None]
    if coef is not None:
        local = local * np.asarray(coef)[:, None, None]
    return _scatter(mesh, local)


def helmholtz_system(mesh: TriMesh, k: float, source=None):
    """Return (A, b, K, M) for  -Laplace(u) - k^2 u = source  with natural (Neumann) BCs.

    ``source(x, y)`` is interpolated at the nodes and integrated with the mass matrix.
    """
    K = stiffness_matrix(mesh)
    M = mass_matrix(mesh)
    A = (K - k**2 * M).astype(np.complex128)
    if source is None:
        b = np.zeros(mesh.n_nodes, dtype=np.complex128)
    else:
        b = M @ source(mesh.nodes[:, 0], mesh.nodes[:, 1]).astype(np.complex128)
    return A.tocsr(), b, K, M


def apply_dirichlet(A: sp.spmatrix, b: np.ndarray, nodes: np.ndarray, values=0.0):
    """Impose u[nodes] = values by symmetric elimination.

    The known values are moved to the right-hand side, the matching rows and columns are
    zeroed and replaced by the identity. The system keeps its size, which is convenient
    for comparing iterative solvers on the full matrix.
    """
    n = A.shape[0]
    g = np.zeros(n, dtype=np.result_type(A.dtype, np.asarray(values).dtype, b.dtype))
    g[nodes] = values
    b = b - A @ g
    b[nodes] = g[nodes]

    free = np.ones(n)
    free[nodes] = 0.0
    keep = sp.diags(free)
    A = keep @ A @ keep + sp.diags(1.0 - free)
    return A.tocsr(), b


def l2_error(M: sp.spmatrix, u_h: np.ndarray, u_exact: np.ndarray) -> float:
    """Discrete L2 norm of the error, ||e||^2 = e^H M e."""
    e = u_h - u_exact
    return float(np.sqrt(np.real(np.vdot(e, M @ e))))
