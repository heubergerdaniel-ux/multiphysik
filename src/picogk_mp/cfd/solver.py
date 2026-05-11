"""High-level CFD pipeline: run_flow (CFD-A) and run_thermal (CFD-B).

run_flow   -- D2Q9 LBM external-flow simulation -> drag coefficient + velocity field
run_thermal -- D2Q5 advection-diffusion -> convective heat transfer coefficient
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .domain import FlowDomain, build_domain
from .lbm_core import compute_drag_x, run_d2q9, run_d2q5


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FlowResult:
    """Output of run_flow."""
    ux_lb: np.ndarray      # (Ny, Nx) x-velocity in lattice units
    uy_lb: np.ndarray      # (Ny, Nx) y-velocity in lattice units
    rho_lb: np.ndarray     # (Ny, Nx) density in lattice units
    fstar: np.ndarray      # (9, Ny, Nx) post-collision distributions
    domain: FlowDomain
    Cd: float              # drag coefficient (dimensionless)
    Re: float
    elapsed_s: float


@dataclass
class ThermalResult:
    """Output of run_thermal."""
    T_field: np.ndarray    # (Ny, Nx) temperature [physical units]
    T_max: float           # maximum temperature at interface [physical units]
    T_surface_avg: float   # average interface temperature
    h_conv: float          # convective heat transfer coefficient [W/m2K]
    domain: FlowDomain
    elapsed_s: float


# ---------------------------------------------------------------------------
# CFD-A: external flow
# ---------------------------------------------------------------------------

def run_flow(
    stl_path: str | Path,
    velocity_m_s: float = 0.5,
    flow_direction: str = "x",
    resolution_mm: float = 3.0,
    max_steps: int = 3000,
    nu_air_m2s: float = 1.5e-5,
    tol: float = 1e-5,
) -> FlowResult:
    """Compute 2D external flow around an STL part using D2Q9 LBM.

    Parameters
    ----------
    stl_path        : Input STL (any geometry from the pipeline).
    velocity_m_s    : Freestream velocity [m/s].
    flow_direction  : "x", "y", or "z" -- main flow axis.
    resolution_mm   : Lattice cell size [mm].  3 mm is a good start.
    max_steps       : Maximum LBM time steps (converges earlier if tol met).
    nu_air_m2s      : Kinematic viscosity of the fluid [m2/s]
                      (air at 20 C: 1.5e-5).
    tol             : Convergence tolerance on max velocity change per step.

    Returns
    -------
    FlowResult with Cd, Re, velocity field and domain geometry.
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    flow_axis = axis_map.get(flow_direction.lower(), 0)

    t0 = time.time()
    domain = build_domain(
        stl_path=stl_path,
        velocity_m_s=velocity_m_s,
        flow_axis=flow_axis,
        resolution_mm=resolution_mm,
        nu_air_m2s=nu_air_m2s,
    )

    ux, uy, rho, fstar = run_d2q9(
        mask=domain.mask,
        U_lb=domain.U_lb,
        nu_lb=domain.nu_lb,
        max_steps=max_steps,
        tol=tol,
    )

    # Drag coefficient
    F_x_lb = compute_drag_x(domain.mask, fstar)
    D_lb   = domain.char_length_m / domain.dx_m   # char length in cells
    Cd     = float(
        F_x_lb / (0.5 * 1.0 * domain.U_lb**2 * D_lb)
    ) if D_lb > 0 else 0.0

    return FlowResult(
        ux_lb=ux,
        uy_lb=uy,
        rho_lb=rho,
        fstar=fstar,
        domain=domain,
        Cd=Cd,
        Re=domain.Re,
        elapsed_s=round(time.time() - t0, 2),
    )


# ---------------------------------------------------------------------------
# CFD-B: thermal (convective cooling)
# ---------------------------------------------------------------------------

def run_thermal(
    flow_result: FlowResult,
    heat_flux_W_m2: float = 1000.0,
    T_inlet_C: float = 20.0,
    max_steps: int = 3000,
    tol: float = 1e-5,
) -> ThermalResult:
    """Compute convective heat transfer using D2Q5 advection-diffusion.

    Couples with the velocity field from run_flow.  Models constant heat flux
    from the solid surface and computes the convective heat transfer coefficient
    h_conv = q / (T_surface_avg - T_inlet).

    Parameters
    ----------
    flow_result     : Output of run_flow (provides velocity field + domain).
    heat_flux_W_m2  : Surface heat flux [W/m2].
    T_inlet_C       : Inlet air temperature [C].
    max_steps       : Maximum D2Q5 time steps.
    tol             : Convergence tolerance.

    Returns
    -------
    ThermalResult with temperature field, h_conv [W/m2K], T_max [C].
    """
    t0 = time.time()
    domain = flow_result.domain

    # Thermal diffusivity of air ~ 2.1e-5 m2/s at 20 C (Pr = nu/alpha ~ 0.71)
    alpha_air = domain.nu_air_m2s / 0.71
    dt = domain.dx_m * domain.U_lb / domain.velocity_m_s
    alpha_lb = alpha_air * dt / domain.dx_m**2

    # Convert heat flux to lattice source per interface cell per step
    # q [W/m2] * dx [m] * dt [s] / (rho*cp*dx^3) [kg/m3 * J/kg/K * m3]
    # For air: rho*cp ~ 1200 J/m3/K
    rho_cp_air = 1200.0  # J/m3/K
    heat_flux_lb = (heat_flux_W_m2 * domain.dx_m * dt) / (rho_cp_air * domain.dx_m**2)

    T_field_K = run_d2q5(
        mask=domain.mask,
        ux_lb=flow_result.ux_lb,
        uy_lb=flow_result.uy_lb,
        alpha_lb=alpha_lb,
        T_inlet=T_inlet_C,
        heat_flux_lb=heat_flux_lb,
        max_steps=max_steps,
        tol=tol,
    )

    # Interface mask: fluid cells touching solid
    mask = domain.mask
    from .lbm_core import _E5
    interface = np.zeros_like(mask)
    for i in range(1, 5):
        nbr = np.roll(np.roll(mask, -int(_E5[i, 1]), axis=0),
                      -int(_E5[i, 0]), axis=1)
        interface |= (~mask & nbr)

    T_interface = T_field_K[interface]
    T_surface_avg = float(T_interface.mean()) if T_interface.size > 0 else T_inlet_C
    T_max = float(T_interface.max()) if T_interface.size > 0 else T_inlet_C

    delta_T = T_surface_avg - T_inlet_C
    h_conv = float(heat_flux_W_m2 / delta_T) if abs(delta_T) > 1e-6 else 0.0

    return ThermalResult(
        T_field=T_field_K,
        T_max=T_max,
        T_surface_avg=T_surface_avg,
        h_conv=h_conv,
        domain=domain,
        elapsed_s=round(time.time() - t0, 2),
    )
