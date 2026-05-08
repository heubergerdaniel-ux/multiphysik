"""Example 08 -- CFD flow and thermal analysis.

Demonstrates Phase CFD of the multiphysik pipeline:
  1. Generate a simple box-shaped part via trimesh
  2. CFD-A: compute external flow (Cd, Re, velocity field)
  3. CFD-B: compute convective cooling (h_conv, temperature field)

No picogk context required -- runs headlessly.
"""
from pathlib import Path
import trimesh

from picogk_mp.cfd import run_flow, run_thermal
from picogk_mp.cfd.postprocess import save_velocity_png, save_temperature_png
from picogk_mp.cfd.checks import DragCheck

DOCS = Path(__file__).resolve().parent.parent / "docs"
DOCS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Generate a test geometry (30 mm box, standing upright)
# ---------------------------------------------------------------------------
stl_path = DOCS / "cfd_test_box.stl"
mesh = trimesh.creation.box(extents=[30.0, 20.0, 60.0])
mesh.export(str(stl_path))
print(f"STL written: {stl_path}")

# ---------------------------------------------------------------------------
# 2. CFD-A: external flow analysis
# ---------------------------------------------------------------------------
print("\n--- CFD-A: External Flow ---")
flow = run_flow(
    stl_path=stl_path,
    velocity_m_s=0.5,
    flow_direction="x",
    resolution_mm=3.0,
    max_steps=2000,
)
print(f"  Domain  : {flow.domain.Ny} x {flow.domain.Nx} cells")
print(f"  Re      : {flow.Re:.0f}")
print(f"  Cd      : {flow.Cd:.3f}")
print(f"  Elapsed : {flow.elapsed_s:.1f} s")

velocity_png = save_velocity_png(flow, DOCS / "cfd_velocity.png")
print(f"  PNG     : {velocity_png}")

# ---------------------------------------------------------------------------
# 3. Drag check (informational)
# ---------------------------------------------------------------------------
drag_check = DragCheck(Cd_warn=3.0)
drag_result = drag_check.evaluate({"Cd": flow.Cd, "Re": flow.Re})
status = "OK" if drag_result.passed else "WARN"
print(f"  [{status}] {drag_result.detail}")

# ---------------------------------------------------------------------------
# 4. CFD-B: convective thermal analysis
# ---------------------------------------------------------------------------
print("\n--- CFD-B: Convective Thermal ---")
thermal = run_thermal(
    flow_result=flow,
    heat_flux_W_m2=500.0,
    T_inlet_C=20.0,
    max_steps=2000,
)
print(f"  h_conv       : {thermal.h_conv:.1f} W/m2K")
print(f"  T_surface_avg: {thermal.T_surface_avg:.2f} C")
print(f"  T_max        : {thermal.T_max:.2f} C")
print(f"  Elapsed      : {thermal.elapsed_s:.1f} s")

temperature_png = save_temperature_png(thermal, DOCS / "cfd_temperature.png")
print(f"  PNG          : {temperature_png}")

print("\nDone. Check docs/ for PNG outputs.")
