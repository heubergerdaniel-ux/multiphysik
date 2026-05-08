"""D2Q9 Lattice-Boltzmann solver (incompressible Navier-Stokes, 2D).

D2Q5 advection-diffusion solver for temperature transport (CFD-B).

All quantities are in non-dimensional *lattice units* unless a _m or _s suffix
is present.  The caller (solver.py) handles physical-to-lattice conversion.

Coordinate convention
---------------------
mask shape  : (Ny, Nx)   -- rows = y, columns = x
f shape     : (9, Ny, Nx)
flow axis   : +x (columns)
inlet       : left face  (x = 0)
outlet      : right face (x = Nx-1)
top/bottom  : zero-gradient (Neumann)
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# D2Q9 lattice constants
# ---------------------------------------------------------------------------
#
# Velocity index layout (viewed as 2D stencil):
#   6  2  5
#   3  0  1
#   7  4  8

_E9 = np.array([
    [ 0,  0],   # 0 rest
    [ 1,  0],   # 1 right
    [ 0,  1],   # 2 up
    [-1,  0],   # 3 left
    [ 0, -1],   # 4 down
    [ 1,  1],   # 5 right-up
    [-1,  1],   # 6 left-up
    [-1, -1],   # 7 left-down
    [ 1, -1],   # 8 right-down
], dtype=np.int32)

_W9 = np.array([
    4/9,
    1/9, 1/9, 1/9, 1/9,
    1/36, 1/36, 1/36, 1/36,
], dtype=np.float64)

_OPP9 = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

# Velocity components as float arrays for broadcasting
_CX9 = _E9[:, 0].astype(np.float64)   # (9,)
_CY9 = _E9[:, 1].astype(np.float64)   # (9,)

# ---------------------------------------------------------------------------
# D2Q5 lattice constants  (advection-diffusion)
# ---------------------------------------------------------------------------
_E5 = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1]], dtype=np.int32)
_W5 = np.array([1/3, 1/6, 1/6, 1/6, 1/6], dtype=np.float64)
_OPP5 = np.array([0, 3, 4, 1, 2], dtype=np.int32)
_CX5 = _E5[:, 0].astype(np.float64)
_CY5 = _E5[:, 1].astype(np.float64)


# ---------------------------------------------------------------------------
# D2Q9 equilibrium and macroscopic quantities
# ---------------------------------------------------------------------------

def _feq9(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 Maxwell-Boltzmann equilibrium.  Returns shape (9, Ny, Nx)."""
    # eu[i, y, x] = cx[i]*ux[y,x] + cy[i]*uy[y,x]
    eu = _CX9[:, None, None] * ux + _CY9[:, None, None] * uy
    usq = ux * ux + uy * uy
    return _W9[:, None, None] * rho[None] * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*usq)


def _macro9(f: np.ndarray):
    """Density and velocity from distribution f (9, Ny, Nx)."""
    rho = f.sum(axis=0)
    ux = (_CX9[:, None, None] * f).sum(axis=0) / rho
    uy = (_CY9[:, None, None] * f).sum(axis=0) / rho
    return rho, ux, uy


def _stream9(f: np.ndarray) -> np.ndarray:
    """Advect distributions one lattice step in their velocity direction."""
    out = np.empty_like(f)
    for i in range(9):
        out[i] = np.roll(np.roll(f[i], int(_E9[i, 0]), axis=1),
                         int(_E9[i, 1]), axis=0)
    return out


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def _zou_he_outlet_p(f: np.ndarray, rho_out: float = 1.0) -> None:
    """Zou-He pressure BC at right face (x=Nx-1), rho=rho_out.

    Prescribes density (pressure) at the outlet.  ux is extrapolated from mass
    conservation; uy is assumed ~0 (valid for flow-aligned outlet).
    Modifies f[:, :, -1] in-place.
    Ref: Zou & He 1997, Phys. Fluids 9(6).
    """
    f0 = f[0, :, -1]; f1 = f[1, :, -1]; f2 = f[2, :, -1]
    f4 = f[4, :, -1]; f5 = f[5, :, -1]; f8 = f[8, :, -1]

    ux_out = (2.0*(f1 + f5 + f8) + f0 + f2 + f4) / rho_out - 1.0

    f[3, :, -1] = f1 - (2.0/3.0) * rho_out * ux_out
    f[7, :, -1] = f5 - (1.0/6.0) * rho_out * ux_out + 0.5*(f2 - f4)
    f[6, :, -1] = f8 - (1.0/6.0) * rho_out * ux_out - 0.5*(f2 - f4)


def _zou_he_inlet(f: np.ndarray, U_lb: float, fluid_inlet: np.ndarray) -> None:
    """Zou-He velocity BC at fluid nodes on the left face (x=0), ux=U_lb, uy=0.

    fluid_inlet : (Ny,) bool, True = fluid node at x=0.
    Modifies f[:, fluid_inlet, 0] in-place.
    Ref: Zou & He 1997, Phys. Fluids 9(6).
    """
    f0 = f[0, fluid_inlet, 0]
    f2 = f[2, fluid_inlet, 0]
    f4 = f[4, fluid_inlet, 0]
    f3 = f[3, fluid_inlet, 0]
    f6 = f[6, fluid_inlet, 0]
    f7 = f[7, fluid_inlet, 0]

    rho_in = (f0 + f2 + f4 + 2.0*(f3 + f6 + f7)) / (1.0 - U_lb)

    f[1, fluid_inlet, 0] = f3 + (2.0/3.0) * rho_in * U_lb
    f[5, fluid_inlet, 0] = f7 - 0.5*(f2 - f4) + (1.0/6.0) * rho_in * U_lb
    f[8, fluid_inlet, 0] = f6 + 0.5*(f2 - f4) + (1.0/6.0) * rho_in * U_lb


# ---------------------------------------------------------------------------
# Drag calculation
# ---------------------------------------------------------------------------

def compute_drag_x(mask: np.ndarray, fstar: np.ndarray) -> float:
    """Momentum exchange drag in x-direction (lattice units).

    fstar : (9, Ny, Nx) post-collision, pre-streaming distributions.
    mask  : (Ny, Nx) bool, True = solid.

    Uses the momentum exchange method: for each solid-fluid interface link,
    the force contribution is 2 * cx * f_fluid[i].
    """
    fluid = ~mask
    F_x = 0.0
    for i in range(9):
        cx = int(_E9[i, 0])
        if cx == 0:
            continue
        cy = int(_E9[i, 1])
        # solid_neighbor[y, x] = mask[y+cy, x+cx] (periodic roll, but
        # boundary cells are fluid by construction of the padded domain)
        solid_neighbor = np.roll(np.roll(mask, -cy, axis=0), -cx, axis=1)
        link = fluid & solid_neighbor          # fluid cell next to solid in direction i
        F_x += 2.0 * cx * float(fstar[i][link].sum())
    return F_x


# ---------------------------------------------------------------------------
# Main D2Q9 simulation
# ---------------------------------------------------------------------------

def run_d2q9(
    mask:      np.ndarray,
    U_lb:      float,
    nu_lb:     float,
    max_steps: int,
    tol:       float = 1e-5,
    check_every: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run D2Q9 BGK-LBM until convergence or max_steps.

    Parameters
    ----------
    mask      : (Ny, Nx) bool, True = solid obstacle.
    U_lb      : Inlet velocity in lattice units (Ma = U_lb / cs, cs = 1/sqrt(3)).
    nu_lb     : Kinematic viscosity in lattice units.
    max_steps : Maximum number of time steps.
    tol       : Convergence criterion on max |du/dt|.
    check_every: Steps between convergence checks.

    Returns
    -------
    ux, uy, rho : (Ny, Nx) arrays, macroscopic quantities at final step.
    fstar       : (9, Ny, Nx) post-collision distributions (for drag).
    """
    # Clamp tau to avoid s = 1/tau approaching 2 (marginal stability).
    # For high-Re cases nu_lb << 1; tau < 0.6 causes instability with curved
    # geometries.  The clamped tau sets an effective floor Re.
    tau = max(3.0 * nu_lb + 0.5, 0.60)
    Ny, Nx = mask.shape

    # Initialise: uniform flow (solid cells zeroed below)
    rho0 = np.ones((Ny, Nx))
    ux0  = np.where(mask, 0.0, U_lb)
    uy0  = np.zeros((Ny, Nx))
    f    = _feq9(rho0, ux0, uy0)

    fstar = f.copy()
    ux_prev = ux0.copy()
    fluid_inlet = ~mask[:, 0]  # fluid nodes at x=0

    for step in range(max_steps):
        # 1. Macroscopic
        rho, ux, uy = _macro9(f)
        ux[mask] = 0.0
        uy[mask] = 0.0

        # 2. Collision (BGK)
        fstar = f - (f - _feq9(rho, ux, uy)) / tau

        # 3. Streaming
        f = _stream9(fstar)

        # 4. Bounce-back at solid nodes (uses pre-streaming fstar)
        for i in range(9):
            f[_OPP9[i]][mask] = fstar[i][mask]

        # 5. Top/bottom zero-gradient (Neumann, before BCs so corners are overridden)
        f[:, 0, :] = f[:, 1, :]
        f[:, -1, :] = f[:, -2, :]

        # 6. Outlet BC (Zou-He pressure, fixes rho=1 at x=Nx-1 to stop mass drift)
        _zou_he_outlet_p(f)

        # 7. Inlet BC (Zou-He velocity, left face x=0, fluid nodes only) — last
        _zou_he_inlet(f, U_lb, fluid_inlet)

        # 8. Convergence check
        if step > 0 and step % check_every == 0:
            err = float(np.abs(ux[~mask] - ux_prev[~mask]).max())
            if err < tol:
                break
            ux_prev = ux.copy()

    rho, ux, uy = _macro9(f)
    ux[mask] = 0.0
    uy[mask] = 0.0
    return ux, uy, rho, fstar


# ---------------------------------------------------------------------------
# D2Q5 temperature transport (advection-diffusion)
# ---------------------------------------------------------------------------

def _geq5(T: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q5 equilibrium for temperature. cs2 = 1/2 for D2Q5.

    Works for any trailing shape: (N,) -> (5, N), (Ny, Nx) -> (5, Ny, Nx).
    """
    T  = np.asarray(T,  dtype=np.float64)
    ux = np.asarray(ux, dtype=np.float64)
    uy = np.asarray(uy, dtype=np.float64)
    extra = (1,) * T.ndim
    cx = _CX5.reshape(-1, *extra)
    cy = _CY5.reshape(-1, *extra)
    W  = _W5.reshape(-1, *extra)
    eu = cx * ux + cy * uy
    return W * T * (1.0 + 2.0 * eu)


def run_d2q5(
    mask:      np.ndarray,
    ux_lb:     np.ndarray,
    uy_lb:     np.ndarray,
    alpha_lb:  float,
    T_inlet:   float,
    heat_flux_lb: float,
    max_steps: int,
    tol:       float = 1e-5,
    check_every: int = 200,
) -> np.ndarray:
    """Run D2Q5 advection-diffusion for temperature field.

    Parameters
    ----------
    mask          : (Ny, Nx) bool, True = solid.
    ux_lb, uy_lb  : Velocity field from run_d2q9 (lattice units).
    alpha_lb      : Thermal diffusivity in lattice units.
    T_inlet       : Inlet temperature (arbitrary units, e.g. Kelvin).
    heat_flux_lb  : Heat source per lattice cell per step at solid surface.
    max_steps     : Max timesteps.

    Returns
    -------
    T : (Ny, Nx) temperature field.
    """
    tau_T = max(3.0 * alpha_lb + 0.5, 0.60)
    Ny, Nx = mask.shape

    T = np.full((Ny, Nx), T_inlet)
    g = _geq5(T, ux_lb, uy_lb)
    T_prev = T.copy()

    # Precompute solid-fluid interface mask (fluid cells adjacent to solid)
    interface = np.zeros((Ny, Nx), dtype=bool)
    for i in range(1, 5):
        neighbor = np.roll(np.roll(mask, -int(_E5[i, 1]), axis=0),
                           -int(_E5[i, 0]), axis=1)
        interface |= (~mask & neighbor)

    for step in range(max_steps):
        # 1. Temperature macroscopic
        T = g.sum(axis=0)
        T[mask] = T_inlet  # reset solid nodes to ambient initially
        # Apply heat flux as source at interface nodes
        T[interface] += heat_flux_lb

        # 2. Collision
        gstar = g - (g - _geq5(T, ux_lb, uy_lb)) / tau_T

        # 3. Streaming
        g_new = np.empty_like(gstar)
        for i in range(5):
            g_new[i] = np.roll(np.roll(gstar[i], int(_E5[i, 0]), axis=1),
                                int(_E5[i, 1]), axis=0)

        # 4. Bounce-back at solid
        for i in range(5):
            g_new[_OPP5[i]][mask] = gstar[i][mask]

        # 5. Inlet T = T_inlet (left face) — apply to g_new, not g
        g_new[:, :, 0] = _geq5(
            np.full((Ny,), T_inlet), ux_lb[:, 0], uy_lb[:, 0]
        )

        # 6. Outlet zero-gradient
        g_new[:, :, -1] = g_new[:, :, -2]

        # 7. Top/bottom ambient
        g_new[:, 0, :] = _geq5(np.full((Nx,), T_inlet),
                                 ux_lb[0, :], uy_lb[0, :])
        g_new[:, -1, :] = _geq5(np.full((Nx,), T_inlet),
                                  ux_lb[-1, :], uy_lb[-1, :])

        g = g_new

        # 8. Convergence
        if step > 0 and step % check_every == 0:
            T_new = g.sum(axis=0)
            err = float(np.abs(T_new[~mask] - T_prev[~mask]).max())
            if err < tol:
                break
            T_prev = T_new.copy()

    return g.sum(axis=0)
