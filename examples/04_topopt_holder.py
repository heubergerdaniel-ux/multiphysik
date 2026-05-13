"""Topology-optimisation example (reference design: headphone holder).

Uses the generic BESO + disc-base-with-tip-load pipeline; the same code
applies to any standing cantilever part.

Workflow
--------
1. SimEngine asks for the operating load (load_mass_g).
2. Design load = load_mass_g * SAFETY_FACTOR (SF=2).
3. BESO iteratively removes material where strain energy is low.
4. Optimised mesh saved as STL + preview PNG.
5. Physics checks (tipping, cantilever bending) report safety factors.

Output
------
docs/headphone_holder_optimised.stl   -- optimised mesh
docs/headphone_holder_optimised.png   -- preview render
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

from picogk_mp.physics import (
    Param, SimEngine, TippingCheck, CantileverBendingCheck,
)
from picogk_mp.topopt  import TopoptPipeline, BoundaryConditions

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT    = Path(__file__).parent.parent
STL_IN  = ROOT / "tests" / "fixtures" / "headphone_holder_v2.stl"
STL_OUT = ROOT / "docs" / "headphone_holder_optimised.stl"
PNG_OUT = ROOT / "docs" / "headphone_holder_optimised.png"

# ------------------------------------------------------------------
# Geometry constants (must match 03_headphone_holder.py)
# ------------------------------------------------------------------
BASE_R_MM        = 48.0
LOAD_OFFSET_MM   = 82.0
MIN_SECTION_R_MM =  7.0
SAFETY_FACTOR    =  2.0        # design load = SF * operating load

# ------------------------------------------------------------------
# 1. Ask for operating parameters
# ------------------------------------------------------------------
query_engine = (
    SimEngine()
    .register(
        Param("load_mass_g",   "Lastmasse",        unit="g",     lo=50, hi=2000),
        Param("infill_pct",    "Infill-Anteil",    unit="%",     default=20, lo=5, hi=100),
        Param("density_g_cm3", "Filament-Dichte",  unit="g/cm3", default=1.24),
        Param("yield_mpa",     "Streckgrenze PLA", unit="MPa",   default=55.0),
    )
)
# Optional preset via env var (non-interactive mode)
if os.environ.get("LOAD_MASS_G"):
    query_engine.inject(load_mass_g=float(os.environ["LOAD_MASS_G"]))
query_engine.run(raise_on_failure=False)

load_mass_g = query_engine._params["load_mass_g"].resolved_value
design_mass = load_mass_g * SAFETY_FACTOR   # [g]  with SF=2
design_F_N  = design_mass * 1e-3 * 9.81     # [N]

print(f"  Operating load: {load_mass_g:.0f} g")
print(f"  Design load   : {design_mass:.0f} g  (SF={SAFETY_FACTOR})")
print(f"  Design force  : {design_F_N:.2f} N")

# ------------------------------------------------------------------
# 2. Topology optimisation
# ------------------------------------------------------------------
if not STL_IN.exists():
    print(f"\nSTL not found: {STL_IN}")
    print("Run 'uv run python examples/03_headphone_holder.py' first.")
    sys.exit(1)

pipeline = TopoptPipeline(
    STL_IN,
    topopt_h_mm=3.0,
    vol_frac=0.75,
    max_iter=40,
    er=0.02,
    r_filter=1.5,
    add_ratio=0.01,
)

# Boundary conditions with design load (SF already baked in)
bc = BoundaryConditions.disc_base_with_tip_load(
    *pipeline.grid_shape, pipeline.h, pipeline.offset,
    base_radius_mm=BASE_R_MM,
    load_point_mm=(-(LOAD_OFFSET_MM), 0.0, 244.0),
    load_mass_g=design_mass,
)

STL_OUT.parent.mkdir(parents=True, exist_ok=True)
pipeline.run(bc, out_stl=STL_OUT)

# ------------------------------------------------------------------
# 3. Physics checks on optimised volume
# ------------------------------------------------------------------
if STL_OUT.exists():
    import trimesh as _tm
    opt_mesh = _tm.load(str(STL_OUT), force="mesh")
    opt_vol  = float(opt_mesh.volume)
    print(f"  Optimised volume: {opt_vol:.0f} mm3")

    check_engine = (
        SimEngine(resolver={})
        .register(
            Param("load_mass_g",           "Lastmasse",          unit="g",     default=load_mass_g),
            Param("infill_pct",            "Infill-Anteil",      unit="%",     default=query_engine._params["infill_pct"].resolved_value),
            Param("density_g_cm3",         "Filament-Dichte",    unit="g/cm3", default=query_engine._params["density_g_cm3"].resolved_value),
            Param("yield_mpa",             "Streckgrenze PLA",   unit="MPa",   default=query_engine._params["yield_mpa"].resolved_value),
            Param("base_r_mm",             "Basisradius",        unit="mm",    default=BASE_R_MM),
            Param("load_offset_mm",        "Lastversatz",        unit="mm",    default=LOAD_OFFSET_MM),
            Param("min_section_radius_mm", "Min. Querschnittsradius", unit="mm", default=MIN_SECTION_R_MM),
            Param("volume_mm3",            "Druckvolumen",       unit="mm3",   default=opt_vol),
        )
        .add_check(TippingCheck())
        .add_check(CantileverBendingCheck())
    )
    check_engine.run(raise_on_failure=False)

# ------------------------------------------------------------------
# 4. Preview render
# ------------------------------------------------------------------
try:
    import vedo
    mesh_v = vedo.load(str(STL_OUT))
    mesh_v.color([100, 140, 120]).lighting("metallic")

    plt = vedo.Plotter(offscreen=True, size=(1280, 960),
                       bg=(35, 35, 46), bg2=(10, 10, 18))
    plt.add(mesh_v)
    plt.show()
    cam = plt.camera
    cam.SetPosition(65, -480, 295)
    cam.SetFocalPoint(-21, 0, 138)
    cam.SetViewUp(0, 0, 1)
    plt.renderer.ResetCameraClippingRange()
    plt.render()
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.screenshot(str(PNG_OUT))
    plt.close()
    print(f"  Preview -> {PNG_OUT}")
except Exception as exc:
    print(f"  Render skipped: {exc}")
