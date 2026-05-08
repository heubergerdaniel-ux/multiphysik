"""CFD sub-package: Lattice-Boltzmann flow and thermal simulation.

CFD-A (external flow)
---------------------
from picogk_mp.cfd import run_flow
result = run_flow("part.stl", velocity_m_s=0.5, resolution_mm=3.0)
print(result.Cd, result.Re)

CFD-B (convective cooling)
--------------------------
from picogk_mp.cfd import run_thermal
thermal = run_thermal(result, heat_flux_W_m2=500.0, T_inlet_C=20.0)
print(thermal.h_conv, thermal.T_max)

Solver: custom D2Q9 (Navier-Stokes) + D2Q5 (advection-diffusion) in NumPy.
No C compiler required.  pylbm installed as an optional extension point.
"""
from .solver import FlowResult, ThermalResult, run_flow, run_thermal
from .postprocess import save_velocity_png, save_temperature_png
from .checks import DragCheck

__all__ = [
    "run_flow",
    "run_thermal",
    "FlowResult",
    "ThermalResult",
    "save_velocity_png",
    "save_temperature_png",
    "DragCheck",
]
