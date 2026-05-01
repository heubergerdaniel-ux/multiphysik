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
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import cg, LinearOperator, spilu, spsolve

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


def fem_solve(
    mask: np.ndarray,
    edofs: np.ndarray,
    KE: np.ndarray,
    fixed_dofs: np.ndarray,
    force_vec: np.ndarray,
    E_min: float = 1e-9,
) -> np.ndarray:
    """Assemble, apply BCs, and solve K*u = f.

    DOF compression: only DOFs that belong to at least one solid element
    are included in the linear system.  This is critical for BESO on sparse
    geometries where solid elements are a small fraction of the grid.

    Force snapping: if a loaded DOF is not connected to any solid element
    (e.g. because the arm tip voxel falls outside the voxelised mask at
    coarse resolution), the load is transferred to the nearest DOF in the
    active set so the force is always applied.

    Solver: conjugate gradient with diagonal (Jacobi) preconditioner.
    CG avoids the SuperLU fill-in memory explosions that plague spsolve on
    near-singular systems and is fast enough for ~25K DOF FEM systems.

    Returns
    -------
    u : (ndof,) displacement vector (zero at fixed and void-only DOFs)
    """
    ndof = 3 * np.prod([s + 1 for s in mask.shape])

    # --- 1. Find active DOFs (connected to >=1 solid element or fixed) ---
    solid = mask.ravel()
    if np.any(solid):
        active = np.unique(edofs[solid].ravel())
    else:
        return np.zeros(ndof)

    # Include only fixed DOFs that are connected to at least one solid element.
    # Orphaned fixed DOFs (within the BC region but not touching any solid element
    # -- e.g. at the voxel-grid boundary of the base circle) add isolated rows
    # with only E_min on the diagonal, which breaks AMG and degrades CG.
    solid_connected = set(active.tolist())
    valid_fixed = fixed_dofs[np.isin(fixed_dofs, list(solid_connected))]
    active = np.unique(np.concatenate([active, valid_fixed]))

    # --- 2. Snap force DOFs that are not in active to nearest active DOF ---
    #   This handles the case where the load point (arm tip) falls outside the
    #   voxelised domain at coarse topopt resolution.
    #   Use physical (grid-coordinate) distance, not DOF-index distance,
    #   so the force stays near the intended application point.
    force_vec = force_vec.copy()
    loaded_global = np.flatnonzero(force_vec)
    if len(loaded_global) > 0:
        Nx_s, Ny_s, Nz_s = mask.shape
        stride_y = Nx_s + 1
        stride_z = (Nx_s + 1) * (Ny_s + 1)
        active_set = set(active.tolist())
        for fd in loaded_global:
            if fd not in active_set:
                axis_fd   = int(fd % 3)
                node_fd   = int(fd // 3)
                iz_fd     = node_fd // stride_z
                rem_fd    = node_fd % stride_z
                iy_fd     = rem_fd  // stride_y
                ix_fd     = rem_fd  % stride_y

                # Candidate: active DOFs on the same displacement axis
                same_axis = active[active % 3 == axis_fd]
                if len(same_axis) > 0:
                    nodes_sa  = same_axis // 3
                    iz_sa     = nodes_sa // stride_z
                    rem_sa    = nodes_sa % stride_z
                    iy_sa     = rem_sa // stride_y
                    ix_sa     = rem_sa % stride_y
                    dist2 = ((ix_sa - ix_fd)**2
                             + (iy_sa - iy_fd)**2
                             + (iz_sa - iz_fd)**2)
                    nearest = same_axis[int(np.argmin(dist2))]
                else:
                    nearest = active[int(np.argmin(np.abs(active - fd)))]
                force_vec[nearest] += force_vec[fd]
                force_vec[fd] = 0.0
                n_near = int(nearest // 3)
                iz_n = n_near // stride_z
                rem_n = n_near % stride_z
                iy_n = rem_n // stride_y
                ix_n = rem_n % stride_y
                print(f"  [FEM] Force DOF {fd} (ix={ix_fd},iy={iy_fd},iz={iz_fd}) "
                      f"snapped to DOF {nearest} (ix={ix_n},iy={iy_n},iz={iz_n})")

    # --- 3. Build compressed index map: global DOF -> compressed index ---
    compress = np.full(ndof, -1, dtype=np.int64)
    compress[active] = np.arange(len(active), dtype=np.int64)

    n_active = len(active)

    # --- 4. Assemble compressed K from solid elements only ---
    es   = edofs[solid]                                  # (Ns, 24)
    es_c = compress[es]                                  # compressed indices

    rows = np.tile(es_c[:, :, None], (1, 1, 24)).ravel()
    cols = np.tile(es_c[:, None, :], (1, 24, 1)).ravel()
    data = np.tile(KE[None, :, :], (es_c.shape[0], 1, 1)).ravel()

    K_c = coo_matrix((data, (rows, cols)),
                     shape=(n_active, n_active)).tocsr()
    # Small diagonal for numerical stability (covers fixed DOFs etc.)
    K_c += E_min * eye(n_active, format="csr")

    # --- 5. Build compressed force vector ---
    f_c = force_vec[active]

    # --- 6. Eliminate fixed DOFs from the compressed system ---
    fixed_c = compress[fixed_dofs]
    fixed_c = fixed_c[fixed_c >= 0]                      # only those in active
    free_mask = np.ones(n_active, dtype=bool)
    free_mask[fixed_c] = False
    free_idx = np.where(free_mask)[0]

    K_free = K_c[np.ix_(free_idx, free_idx)]
    f_free = f_c[free_idx]

    # --- 7. Solve: AMG-preconditioned CG, with direct-solver fallback ---
    #   Strategy (in order of preference):
    #   a) Smoothed-aggregation AMG (PyAMG) + CG  -- O(10-50) iters, fast
    #   b) If CG diverges: SuperLU direct solver   -- exact, handles ill-cond.
    #      The compressed system is O(15-25K DOFs) which SuperLU handles in
    #      memory without issue (the earlier crash was on the *uncompressed*
    #      430K-DOF system).
    #   c) If spsolve also fails: Jacobi-CG best-effort (warns user).
    u_c = np.zeros(n_active)
    if len(free_idx) > 0:
        K_csc = K_free.tocsc()

        # Build preconditioner
        try:
            import pyamg
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ml = pyamg.smoothed_aggregation_solver(
                    K_csc,
                    B=np.ones((K_free.shape[0], 1)),   # near-null: constant
                )
            M = ml.aspreconditioner(cycle="V")
        except Exception:
            try:
                ilu = spilu(K_csc, drop_tol=1e-4, fill_factor=20)
                M = LinearOperator(K_free.shape, ilu.solve)
            except (MemoryError, RuntimeError):
                diag_vals = K_free.diagonal()
                diag_vals = np.where(np.abs(diag_vals) < 1e-30, 1.0, diag_vals)
                M = diags(1.0 / diag_vals)

        u_free, info = cg(K_csc, f_free, M=M, rtol=1e-6, maxiter=2000)

        if info != 0:
            # CG failed -- fall back to direct solver (SuperLU).
            # The compressed system is small enough (~15-25K DOFs) that
            # SuperLU fill-in is well within RAM even for sparse geometries.
            if info > 0:
                print(f"  [FEM] CG did not converge ({info} iters) "
                      "-- direct solver fallback")
            else:
                print(f"  [FEM] CG breakdown (info={info}) "
                      "-- direct solver fallback")
            try:
                u_free = spsolve(K_csc, f_free)
                if not np.all(np.isfinite(u_free)):
                    raise RuntimeError("spsolve produced non-finite values")
                print(f"  [FEM] Direct solve OK "
                      f"({K_csc.shape[0]:,} DOFs)")
            except Exception as e_direct:
                print(f"  [FEM] Direct solver also failed ({e_direct}) "
                      "-- result approximate")
                # u_free already holds the last CG iterate; keep it.

        u_c[free_idx] = u_free

    # --- 8. Map back to full DOF vector ---
    u = np.zeros(ndof)
    u[active] = u_c
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
