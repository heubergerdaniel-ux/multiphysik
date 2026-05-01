"""3-D FEM for linear elasticity on a regular voxel grid.

Each voxel is one 8-node hexahedral element (Hex8).  No external FEM
library is required -- the solver uses only NumPy and SciPy sparse.

Coordinate convention
---------------------
mask[ix, iy, iz] == True  ->  solid element
Physical position of node (ix, iy, iz): (ix*h, iy*h, iz*h) in mm.
DOF layout for node n: [3n, 3n+1, 3n+2] = [u_x, u_y, u_z].

Performance notes
-----------------
Assembly vectorises over all solid elements simultaneously.  For
~25 000 solid elements (holder at 3 mm) the full assembly+solve takes
roughly 5-15 s on a typical desktop CPU.  Void elements are excluded
from the COO data; a small diagonal regularisation E_min keeps the
global K non-singular for DOFs that touch only void elements.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve

# ---------------------------------------------------------------------------
# Isoparametric node coordinates for Hex8 in [-1,+1]^3
# Standard ordering: 4 nodes on zeta=-1 face, then zeta=+1 face.
# ---------------------------------------------------------------------------
_NODE_ISO = np.array([
    [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
    [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
], dtype=float)

# Physical corner offsets (dx,dy,dz in {0,1}) matching _NODE_ISO ordering
_DX = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int32)
_DY = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int32)
_DZ = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)


def _shape_derivs(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Shape-function derivatives in isoparametric coords. Returns (3, 8)."""
    n = _NODE_ISO
    return np.array([
        n[:, 0] * (1 + n[:, 1] * eta)  * (1 + n[:, 2] * zeta) / 8,
        n[:, 1] * (1 + n[:, 0] * xi)   * (1 + n[:, 2] * zeta) / 8,
        n[:, 2] * (1 + n[:, 0] * xi)   * (1 + n[:, 1] * eta)  / 8,
    ])


def _elasticity_matrix(E: float, nu: float) -> np.ndarray:
    """6x6 Voigt elasticity matrix for linear isotropic material.

    Voigt order: [s_xx, s_yy, s_zz, s_xy, s_yz, s_xz].
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


@lru_cache(maxsize=64)
def element_stiffness(h: float, E: float, nu: float) -> np.ndarray:
    """24x24 stiffness matrix for a cubic Hex8 element of side *h* [mm].

    Result is cached (same h/E/nu -> reused across BESO iterations).
    """
    C = _elasticity_matrix(E, nu)
    gp = [-1 / np.sqrt(3), 1 / np.sqrt(3)]      # 2-point Gauss
    jac = h / 2.0                                 # J = jac * I  (regular cube)
    det_J = jac ** 3

    KE = np.zeros((24, 24))
    for xi in gp:
        for eta in gp:
            for zeta in gp:
                dN = _shape_derivs(xi, eta, zeta) / jac   # (3,8) physical

                B = np.zeros((6, 24))
                for i in range(8):
                    B[0, 3*i]   = dN[0, i]
                    B[1, 3*i+1] = dN[1, i]
                    B[2, 3*i+2] = dN[2, i]
                    B[3, 3*i]   = dN[1, i];  B[3, 3*i+1] = dN[0, i]
                    B[4, 3*i+1] = dN[2, i];  B[4, 3*i+2] = dN[1, i]
                    B[5, 3*i]   = dN[2, i];  B[5, 3*i+2] = dN[0, i]

                KE += B.T @ C @ B * det_J   # weight = 1 for 2-pt Gauss

    return KE


def element_dof_indices(Nx: int, Ny: int, Nz: int) -> np.ndarray:
    """Global DOF indices for every element.

    Returns
    -------
    edofs : (Nx*Ny*Nz, 24)  int64
        edofs[e, k] is the global DOF index for local DOF k of element e.
        Elements are ordered C-style with iz varying fastest.
    """
    ex, ey, ez = np.meshgrid(
        np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij"
    )
    ex, ey, ez = ex.ravel(), ey.ravel(), ez.ravel()      # (Nel,)

    # Node indices for each of the 8 corners: shape (Nel, 8)
    node = (
        (ex[:, None] + _DX[None, :])
        + (ey[:, None] + _DY[None, :]) * (Nx + 1)
        + (ez[:, None] + _DZ[None, :]) * (Nx + 1) * (Ny + 1)
    )

    # DOF indices: 3 per node [ux, uy, uz]
    edofs = np.empty((len(ex), 24), dtype=np.int64)
    for i in range(8):
        edofs[:, 3*i]   = 3 * node[:, i]
        edofs[:, 3*i+1] = 3 * node[:, i] + 1
        edofs[:, 3*i+2] = 3 * node[:, i] + 2

    return edofs


def assemble_K(
    mask: np.ndarray,
    edofs: np.ndarray,
    KE: np.ndarray,
    E_min: float = 1e-9,
) -> "scipy.sparse.csr_matrix":
    """Assemble global stiffness matrix.

    Solid elements (mask==True) use E_factor=1.  Void elements are
    excluded from the COO data; a diagonal E_min term keeps the system
    non-singular for void-only DOFs.

    Parameters
    ----------
    mask   : (Nx, Ny, Nz) bool
    edofs  : (Nel, 24) int64
    KE     : (24, 24)  -- stiffness for unit E
    E_min  : regularisation stiffness for void DOFs
    """
    Nx, Ny, Nz = mask.shape
    ndof = 3 * (Nx + 1) * (Ny + 1) * (Nz + 1)
    solid = mask.ravel()

    if not np.any(solid):
        return E_min * eye(ndof, format="csr")

    es = edofs[solid]                               # (Ns, 24) solid elements only

    # COO triplets: (Ns, 24, 24) -> ravel
    rows = np.tile(es[:, :, None], (1, 1, 24)).ravel()
    cols = np.tile(es[:, None, :], (1, 24, 1)).ravel()
    data = np.tile(KE[None, :, :], (es.shape[0], 1, 1)).ravel()

    K = coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()
    K += E_min * eye(ndof, format="csr")            # void-DOF regularisation
    return K


def apply_bcs(
    K: "scipy.sparse.csr_matrix",
    f: np.ndarray,
    fixed_dofs: np.ndarray,
) -> Tuple["scipy.sparse.csr_matrix", np.ndarray, np.ndarray]:
    """Eliminate fixed DOFs from K and f (penalty / elimination method).

    Returns (K_free, f_free, free_dofs) where free_dofs lists the active DOFs.
    """
    ndof = K.shape[0]
    all_dofs  = np.arange(ndof)
    fixed_set = set(fixed_dofs.tolist())
    free_dofs = np.array([d for d in all_dofs if d not in fixed_set], dtype=np.int64)

    K_free = K[np.ix_(free_dofs, free_dofs)]
    f_free = f[free_dofs]
    return K_free, f_free, free_dofs


def fem_solve(
    mask: np.ndarray,
    edofs: np.ndarray,
    KE: np.ndarray,
    fixed_dofs: np.ndarray,
    force_vec: np.ndarray,
    E_min: float = 1e-9,
) -> np.ndarray:
    """Assemble, apply BCs, and solve K*u = f.

    Returns
    -------
    u : (ndof,) displacement vector (zero at fixed DOFs)
    """
    K = assemble_K(mask, edofs, KE, E_min=E_min)
    K_free, f_free, free_dofs = apply_bcs(K, force_vec, fixed_dofs)

    # Solve -- spsolve uses UMFPACK or SuperLU depending on SciPy build
    u_free = spsolve(K_free.tocsc(), f_free)

    ndof = K.shape[0]
    u = np.zeros(ndof)
    u[free_dofs] = u_free
    return u


def element_strain_energy(
    u: np.ndarray,
    edofs: np.ndarray,
    KE: np.ndarray,
) -> np.ndarray:
    """Strain energy per element: c_e = 0.5 * u_e^T * KE * u_e.

    Used as the BESO sensitivity number (higher = element carries more load).

    Returns
    -------
    ce : (Nel,)
    """
    u_e = u[edofs]                                  # (Nel, 24)
    # c_e = 0.5 * sum_j (KE @ u_e[e]) * u_e[e]
    Ku  = u_e @ KE                                  # (Nel, 24)
    ce  = 0.5 * np.einsum("ei,ei->e", Ku, u_e)     # (Nel,)
    return ce
