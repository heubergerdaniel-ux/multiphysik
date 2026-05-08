"""Modulation Showcase -- parametric pavilion.

Eight columns arranged in a ring, each driven by a DIFFERENT LineModulation
mode for radius.  Every engineering/math profile sits side by side so the
visual difference is immediately obvious.

No picogk.go context required.

Modulation modes demonstrated
------------------------------
  Col 0  constant            -- uniform 4.5 mm baseline
  Col 1  from_endpoints      -- linear taper: wide base, slender top (classic column)
  Col 2  from_endpoints inv  -- inverted taper: slender base, flares at capital
  Col 3  from_control_points -- symmetric organic waist (S-spline profile)
  Col 4  from_function sin^2 -- smooth lenticular belly (max at mid-height)
  Col 5  from_function |sin| -- triple-bead bamboo (periodic)
  Col 6  from_function exp   -- exponential stress decay (cantilever-optimal)
  Col 7  from_function para  -- parabolic bending-moment profile

Central spire
-------------
  Helical polyline spine (2.5 turns, r=7 mm) with abs-sin beaded radius:
  r(t) = 1.8 + 1.4 * |sin(7 pi t)|  -- 7 bulges over the full height

Connectors
----------
  CapsuleShape bridges between every pair of adjacent columns at 1/3 and 2/3
  height -- gives the pavilion its structural "skeleton" look.

Assembly
--------
  CompoundShape union of base plate, top plate, 2 torus rings, 8 modulated
  columns, 16 ring connectors, ~50 junction spheres, central helix spire.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from picogk_mp.shapek import (
    CapsuleShape,
    CompoundShape,
    CylinderShape,
    LineModulation,
    Measure,
    PipeShape,
    SphereShape,
    TorusShape,
)

OUT_STL       = Path(__file__).parent.parent / "docs" / "modulation_pavilion.stl"
OUT_PNG       = Path(__file__).parent.parent / "docs" / "modulation_pavilion.png"
RESOLUTION_MM = 1.0    # 1 mm -- full pavilion is ~130x130x150 mm, fast enough


# ---------------------------------------------------------------------------
# Pavilion parameters
# ---------------------------------------------------------------------------

N_COLS       = 8        # number of columns
R_RING       = 48.0     # column circle radius [mm]
H            = 140.0    # total height [mm]
H_BOT        = 5.0      # z where columns start (sits on base plate)
H_TOP        = H - 5.0  # z where columns end  (meets top plate)
N_SPINE      = 24       # spine waypoints per column  (more = smoother modulation)

R_PLATE      = R_RING + 14.0   # base / top plate radius
CONN_R       = 1.8             # connector capsule radius
JUNC_SCALE   = 1.15            # junction sphere = junc_scale * local column radius


# ---------------------------------------------------------------------------
# 1.  Define one LineModulation per column
# ---------------------------------------------------------------------------

COLUMN_MODS = [
    # (label, LineModulation)
    ("constant  r=4.5",
     LineModulation.constant(4.5)),

    ("linear taper  7->2  (classical column)",
     LineModulation.from_endpoints(7.0, 2.0)),

    ("inv. taper  2->7  (flared capital)",
     LineModulation.from_endpoints(2.0, 7.0)),

    ("spline waist  7-2.5-7  (organic S-profile)",
     LineModulation.from_control_points(
         [(0.0, 7.0), (0.2, 5.0), (0.5, 2.5), (0.8, 5.0), (1.0, 7.0)]
     )),

    ("sin^2 belly  (max at mid-height)",
     LineModulation.from_function(
         lambda t: 2.5 + 5.5 * np.sin(np.pi * t) ** 2
     )),

    ("|sin(3pi*t)|  triple-bead bamboo",
     LineModulation.from_function(
         lambda t: 2.0 + 5.5 * np.abs(np.sin(3 * np.pi * t))
     )),

    ("exp decay  2+5.5*e^-4t  (cantilever stress-optimal)",
     LineModulation.from_function(
         lambda t: 2.0 + 5.5 * np.exp(-4.0 * t)
     )),

    ("parabola  4t(1-t)  (bending-moment profile)",
     LineModulation.from_function(
         lambda t: 2.0 + 5.5 * 4.0 * t * (1.0 - t)
     )),
]

assert len(COLUMN_MODS) == N_COLS


# ---------------------------------------------------------------------------
# 2.  Build the eight columns + junction spheres
# ---------------------------------------------------------------------------

columns        = []
base_junctions = []
top_junctions  = []

for i, (label, mod) in enumerate(COLUMN_MODS):
    angle = i * 2.0 * np.pi / N_COLS
    cx    = R_RING * np.cos(angle)
    cy    = R_RING * np.sin(angle)

    # Straight vertical spine, H_BOT -> H_TOP
    zs    = np.linspace(H_BOT, H_TOP, N_SPINE)
    spine = np.column_stack([np.full(N_SPINE, cx),
                              np.full(N_SPINE, cy),
                              zs])
    columns.append(PipeShape(spine, mod))

    # Sphere at base + top sized to match local column radius
    r_bot = mod.at(0.0)
    r_top = mod.at(1.0)
    base_junctions.append(SphereShape([cx, cy, H_BOT], r_bot * JUNC_SCALE))
    top_junctions.append( SphereShape([cx, cy, H_TOP], r_top * JUNC_SCALE))


# ---------------------------------------------------------------------------
# 3.  Horizontal ring connectors at 1/3 and 2/3 height
# ---------------------------------------------------------------------------

connectors       = []
ring_junctions   = []

for frac in (1.0 / 3.0, 2.0 / 3.0):
    z_c = H_BOT + (H_TOP - H_BOT) * frac
    for i in range(N_COLS):
        j     = (i + 1) % N_COLS
        a_i   = i * 2.0 * np.pi / N_COLS
        a_j   = j * 2.0 * np.pi / N_COLS
        p1    = [R_RING * np.cos(a_i), R_RING * np.sin(a_i), z_c]
        p2    = [R_RING * np.cos(a_j), R_RING * np.sin(a_j), z_c]
        connectors.append(CapsuleShape(p1, p2, CONN_R, CONN_R))
        # Junction knot where connector meets column
        ring_junctions.append(SphereShape(p1, CONN_R * 2.5))


# ---------------------------------------------------------------------------
# 4.  Central helical spire
# ---------------------------------------------------------------------------

N_HELIX     = 120
HELIX_TURNS = 2.5
HELIX_R     = 7.0         # arm radius of the helix
helix_t     = np.linspace(0.0, 1.0, N_HELIX)
helix_spine = np.column_stack([
    HELIX_R * np.cos(2.0 * np.pi * HELIX_TURNS * helix_t),
    HELIX_R * np.sin(2.0 * np.pi * HELIX_TURNS * helix_t),
    H_BOT + (H_TOP - H_BOT) * helix_t,
])
# 7 beads over 2.5 turns: r = 1.8 + 1.4 * |sin(7 pi t)|
helix_mod = LineModulation.from_function(
    lambda t: 1.8 + 1.4 * np.abs(np.sin(7.0 * np.pi * t))
)
central_spire = PipeShape(helix_spine, helix_mod)


# ---------------------------------------------------------------------------
# 5.  Base plate, top plate, and decorative torus rings
# ---------------------------------------------------------------------------

base_plate = CylinderShape(center_xy=[0, 0], z_range=[0,     H_BOT],  radius=R_PLATE)
top_plate  = CylinderShape(center_xy=[0, 0], z_range=[H_TOP, H],      radius=R_PLATE)
base_ring  = TorusShape(   center=[0, 0, H_BOT], major_r=R_RING, minor_r=3.5)
top_ring   = TorusShape(   center=[0, 0, H_TOP], major_r=R_RING, minor_r=3.5)


# ---------------------------------------------------------------------------
# 6.  Assemble everything into one CompoundShape
# ---------------------------------------------------------------------------

pavilion = CompoundShape(
    base_plate, top_plate,
    base_ring,  top_ring,
    *columns,
    *base_junctions,
    *top_junctions,
    *connectors,
    *ring_junctions,
    central_spire,
)


# ---------------------------------------------------------------------------
# 7.  Generate STL
# ---------------------------------------------------------------------------

print("=" * 62)
print("Modulation Showcase -- Parametric Pavilion")
print("=" * 62)
print(f"\nColumn ring:  {N_COLS} columns at R = {R_RING:.0f} mm")
print(f"Height:       {H:.0f} mm")
print(f"Resolution:   {RESOLUTION_MM} mm\n")

print("Column modulation profiles:")
for i, (label, mod) in enumerate(COLUMN_MODS):
    r_min = min(mod.at(t) for t in np.linspace(0, 1, 40))
    r_max = max(mod.at(t) for t in np.linspace(0, 1, 40))
    print(f"  [{i}] {label:<42}  r in [{r_min:.1f}, {r_max:.1f}] mm")

print(f"\nHelix spire:  {HELIX_TURNS} turns, arm R = {HELIX_R:.0f} mm,"
      f" beaded r in [1.8, 3.2] mm")
print(f"\nTotal primitives in CompoundShape:")
print(f"  {N_COLS} PipeShapes x {N_SPINE-1} segments  = {N_COLS*(N_SPINE-1)} capsules")
print(f"  1  spire   x {N_HELIX-1} segments  = {N_HELIX-1} capsules")
print(f"  {len(connectors)} CapsuleShape connectors")
print(f"  {len(base_junctions)+len(top_junctions)+len(ring_junctions)} SphereShapes (junctions)")
print(f"  4 plates/rings")
total_prims = (N_COLS*(N_SPINE-1) + (N_HELIX-1) + len(connectors) +
               len(base_junctions)+len(top_junctions)+len(ring_junctions) + 4)
print(f"  TOTAL  ~{total_prims} primitive SDF calls per grid point\n")

print(f"Generating mesh ...")
t0 = time.time()
OUT_STL.parent.mkdir(parents=True, exist_ok=True)
result = pavilion.mesh_stl(resolution_mm=RESOLUTION_MM, out_stl=str(OUT_STL))
t_mesh = time.time() - t0

if result["status"] != "ok":
    print(f"ERROR: {result}")
    raise SystemExit(1)

print(f"\nMesh generated in {t_mesh:.1f} s")
print(f"  Volume:     {result['volume_mm3']:,} mm3")
print(f"  Bounds min: {result['bounds_min']}")
print(f"  Bounds max: {result['bounds_max']}")


# ---------------------------------------------------------------------------
# 8.  Measure
# ---------------------------------------------------------------------------

print("\nPhysical measurement (PLA, 100% infill):")
m = Measure.from_stl(OUT_STL, density_g_cm3=1.24, infill_pct=100.0)
print(f"  Volume:        {m.volume_mm3:,.0f} mm3")
print(f"  Surface area:  {m.surface_area_mm2:,.0f} mm2")
print(f"  Mass:          {m.mass_g:.1f} g")
cog = m.center_of_gravity_mm
print(f"  CoG:           ({cog[0]:.1f}, {cog[1]:.1f}, {cog[2]:.1f}) mm")
vals, _ = Measure.principal_axes(m)
print(f"  Principal inertia [g*mm2]:  "
      f"I1={vals[0]:.0f}  I2={vals[1]:.0f}  I3={vals[2]:.0f}")


# ---------------------------------------------------------------------------
# 9.  Render
# ---------------------------------------------------------------------------

print(f"\nRendering to {OUT_PNG} ...")
try:
    import vedo

    mv = vedo.load(str(OUT_STL))
    mv.color([170, 190, 220]).lighting("metallic")

    plt = vedo.Plotter(
        offscreen=True, size=(1280, 1280),
        bg=(18, 18, 26), bg2=(5, 5, 14),
    )
    plt.add(mv)
    plt.show()

    # Elevated 3/4 view -- shows circular layout and height of every column
    cam = plt.camera
    cam.SetPosition(180, -180, 180)
    cam.SetFocalPoint(0, 0, 60)
    cam.SetViewUp(0, 0, 1)
    plt.renderer.ResetCameraClippingRange()
    plt.render()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.screenshot(str(OUT_PNG))
    plt.close()
    print(f"  Saved: {OUT_PNG}  ({OUT_PNG.stat().st_size // 1024} KB)")
except ImportError:
    print("  (vedo not available)")
except Exception as exc:
    print(f"  (render failed: {exc})")


print("\nDone.")
print("=" * 62)
print(f"  STL -> {OUT_STL.relative_to(OUT_STL.parent.parent.parent)}")
if OUT_PNG.exists():
    print(f"  PNG -> {OUT_PNG.relative_to(OUT_PNG.parent.parent.parent)}")
print("=" * 62)
