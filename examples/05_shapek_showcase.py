"""ShapeKernel showcase -- parametric orbital column.

No picogk.go context required.  Runs headlessly on any machine with the
normal uv environment.

What this demonstrates
----------------------
  LocalFrame       -- tilted coordinate system for the capital torus ring
  LineModulation   -- organic S-curve radius profile along the pipe shaft
  sdf_cone         -- inverted capital crown + flared base plinth
  sdf_torus        -- decorative rings at base and capital
  sdf_pipe         -- S-curve shaft with modulated radius
  CompoundShape    -- single union of all six geometry pieces
  Measure.from_stl -- volume, mass, CoG, moment of inertia tensor

Geometry summary (all dimensions in mm)
----------------------------------------
  z=0..10    cylindrical base disc, r=28
  z=0..32    flared base plinth (cone), base r=28, apex r=0 at z=32
  z=10       decorative base torus ring, major_r=24, minor_r=3
  z=10..200  organic pipe shaft, S-curve spine, r varies 11->6->11
  z=200..235 capital crown cone, r=22 at z=200 tapering to point at z=240
  z=200      capital torus ring, major_r=20, minor_r=3, tilted 12 deg
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from picogk_mp.shapek import (
    CompoundShape,
    ConeShape,
    CylinderShape,
    LineModulation,
    LocalFrame,
    Measure,
    PipeShape,
    SphereShape,
    TorusShape,
)

OUT_STL  = Path(__file__).parent.parent / "docs" / "shapek_column.stl"
OUT_PNG  = Path(__file__).parent.parent / "docs" / "shapek_column.png"
RESOLUTION_MM = 1.0   # 1 mm = good quality, ~5-10 s


# ---------------------------------------------------------------------------
# 1. Base disc + plinth
# ---------------------------------------------------------------------------

base_disc  = CylinderShape(center_xy=[0, 0], z_range=[0, 10], radius=28)

# Inverted cone: forms a flared plinth skirt below the base disc.
# apex at z=32 (tip), base at z=0 (wide), so it looks like a splayed foot.
plinth = ConeShape(apex=[0, 0, 32], base=[0, 0, 0], r_base=28)

# Decorative ring sitting on the base disc rim
base_ring = TorusShape(center=[0, 0, 10], major_r=24, minor_r=3)


# ---------------------------------------------------------------------------
# 2. Organic pipe shaft -- LineModulation + S-curve spine
# ---------------------------------------------------------------------------

spine = np.array([
    [  0,  0,  10],   # base (sits on top of disc)
    [  8,  0,  70],   # lean right (+x) as it rises
    [ -8,  0, 140],   # lean back (-x) through the middle waist
    [  0,  0, 200],   # return to centre at the capital
], dtype=float)

# Cubic spline radius profile: wide at base/top, narrows to an elegant waist
shaft_radius = LineModulation.from_control_points([
    (0.00, 12.0),   # wide collar at base
    (0.20,  8.5),   # quick taper after base join
    (0.50,  5.5),   # minimum waist at mid-height
    (0.80,  8.5),   # swells back toward capital
    (1.00, 12.0),   # wide collar at top
])

shaft = PipeShape(spine, shaft_radius)

# Junction spheres smooth the transition at base and capital
base_junction    = SphereShape([0, 0, 10],  12.0)
capital_junction = SphereShape([0, 0, 200], 12.0)


# ---------------------------------------------------------------------------
# 3. Capital crown -- tilted torus + cone
# ---------------------------------------------------------------------------

# LocalFrame tilted 12 degrees in XZ plane -- gives the capital ring
# a subtle lean, like a classical Ionic capital abacus.
tilt_rad   = np.radians(12)
cap_tangent = [np.sin(tilt_rad), 0.0, np.cos(tilt_rad)]

capital_frame = LocalFrame(
    origin=[0, 0, 200],
    tangent=cap_tangent,
    normal=[1, 0, 0],
)
capital_ring = TorusShape(
    center=[0, 0, 200],
    major_r=20,
    minor_r=3.5,
    frame=capital_frame,
)

# Crown cone: apex points upward, base ring at capital level
crown = ConeShape(apex=[0, 0, 240], base=[0, 0, 200], r_base=22)

# Top finishing disc
top_disc = CylinderShape(center_xy=[0, 0], z_range=[237, 245], radius=18)


# ---------------------------------------------------------------------------
# 4. Assemble into CompoundShape (SDF union)
# ---------------------------------------------------------------------------

column = CompoundShape(
    base_disc,
    plinth,
    base_ring,
    shaft,
    base_junction,
    capital_junction,
    capital_ring,
    crown,
    top_disc,
)


# ---------------------------------------------------------------------------
# 5. Generate STL via marching cubes (no picogk required)
# ---------------------------------------------------------------------------

print("=" * 60)
print("ShapeKernel Orbital Column -- showcase")
print("=" * 60)
print(f"\nPrimitives used:")
print(f"  CylinderShape  (base disc + top disc)")
print(f"  ConeShape      (plinth flare + crown cone)")
print(f"  TorusShape     (base ring + capital ring, tilted by LocalFrame)")
print(f"  PipeShape      (S-curve shaft with LineModulation radius)")
print(f"  SphereShape    (junction blending spheres)")
print(f"\nSpine waypoints: {len(spine)}")
print(f"Radius profile:  control-point spline, min={shaft_radius.at(0.5):.1f} mm,  "
      f"max={shaft_radius.at(0.0):.1f} mm")
print(f"Capital tilt:    {np.degrees(tilt_rad):.0f} deg (LocalFrame)")
print(f"\nGenerating mesh at {RESOLUTION_MM} mm resolution ...")

t0 = time.time()
OUT_STL.parent.mkdir(parents=True, exist_ok=True)
result = column.mesh_stl(resolution_mm=RESOLUTION_MM, out_stl=str(OUT_STL))
t_mesh = time.time() - t0

if result["status"] != "ok":
    print(f"ERROR: {result}")
    raise SystemExit(1)

print(f"\nMesh generated in {t_mesh:.1f} s")
print(f"  STL:        {OUT_STL}")
print(f"  Volume:     {result['volume_mm3']:,} mm3")
print(f"  Bounds min: {result['bounds_min']}")
print(f"  Bounds max: {result['bounds_max']}")


# ---------------------------------------------------------------------------
# 6. Measure: volume, mass, CoG, inertia
# ---------------------------------------------------------------------------

print("\nPhysical measurement (PLA, 20% infill):")
m = Measure.from_stl(OUT_STL, density_g_cm3=1.24, infill_pct=20.0)
print(f"  Volume:        {m.volume_mm3:,.0f} mm3")
print(f"  Surface area:  {m.surface_area_mm2:,.0f} mm2")
print(f"  Mass (20% PLA):{m.mass_g:.1f} g")
cog = m.center_of_gravity_mm
print(f"  Centre of gravity: ({cog[0]:.1f}, {cog[1]:.1f}, {cog[2]:.1f}) mm")
vals, _ = Measure.principal_axes(m)
print(f"  Principal moments of inertia [g*mm2]:")
print(f"    I1={vals[0]:.1f}  I2={vals[1]:.1f}  I3={vals[2]:.1f}")


# ---------------------------------------------------------------------------
# 7. Render to PNG (vedo offscreen)
# ---------------------------------------------------------------------------

print(f"\nRendering to {OUT_PNG} ...")
try:
    import vedo

    mv = vedo.load(str(OUT_STL))
    mv.color([120, 140, 170]).lighting("metallic")

    plt = vedo.Plotter(
        offscreen=True, size=(1280, 1280),
        bg=(30, 30, 40), bg2=(8, 8, 20),
    )
    plt.add(mv)
    plt.show()

    # Camera: 3/4 front-right view, slightly elevated, shows S-curve profile
    cam = plt.camera
    cam.SetPosition(220, -340, 200)
    cam.SetFocalPoint(0, 0, 120)
    cam.SetViewUp(0, 0, 1)
    plt.renderer.ResetCameraClippingRange()
    plt.render()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.screenshot(str(OUT_PNG))
    plt.close()
    print(f"  Saved: {OUT_PNG}  ({OUT_PNG.stat().st_size // 1024} KB)")
except ImportError:
    print("  (vedo not available; skipping render)")
except Exception as exc:
    print(f"  (render failed: {exc})")


print("\nDone.")
print("=" * 60)
print(f"  STL -> {OUT_STL.relative_to(OUT_STL.parent.parent.parent)}")
if OUT_PNG.exists():
    print(f"  PNG -> {OUT_PNG.relative_to(OUT_PNG.parent.parent.parent)}")
print("=" * 60)
