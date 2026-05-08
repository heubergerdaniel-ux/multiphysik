"""ESP32 DevKitC housing tray -- DifferenceShape showcase.

No picogk.go context required.  Runs headlessly with the normal uv env.

What this demonstrates
----------------------
  DifferenceShape  -- hollow the outer shell (outer box - inner cavity)
  DifferenceShape  -- punch USB Micro-B port through the +X wall
  DifferenceShape  -- cable-exit slot through the -X wall
  DifferenceShape  -- four M3 mounting holes through the floor
  CompoundShape    -- assemble shell + standoffs + retaining nubs
  CylinderShape    -- PCB standoffs and screw-hole void cylinders
  SphereShape      -- snap-fit retaining nubs above standoffs
  Measure.from_stl -- volume, mass, CoG, inertia

Board form-factor (ESP32-DevKitC 38-pin)
-----------------------------------------
  PCB:          55 x 28 mm, 1.6 mm thick
  USB Micro-B:  on the +X short end, centred
  Headers:      2 x 19 pins, 2.54 mm pitch, on the long sides
  Max component height above PCB: 10 mm (antenna module)

Housing geometry (all dimensions mm)
--------------------------------------
  Wall thickness:  2 mm
  Floor thickness: 2 mm
  PCB standoffs:   2 mm high  (PCB bottom at z = 4 mm)
  Inner cavity:    57 x 30 x 14.6 mm (1 mm clearance each XY side)
  Outer shell:     61 x 34 x 16.6 mm
  USB cutout:      10 x 5 mm on +X face, at PCB connector height
  Cable slot:       8 x 6 mm on -X face, at floor level
  M3 screw holes:  r = 1.6 mm through floor at four corners
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from picogk_mp.shapek import (
    BoxShape,
    CompoundShape,
    CylinderShape,
    DifferenceShape,
    Measure,
    SphereShape,
)

OUT_STL       = Path(__file__).parent.parent / "docs" / "esp32_housing.stl"
OUT_PNG       = Path(__file__).parent.parent / "docs" / "esp32_housing.png"
RESOLUTION_MM = 0.5      # 0.5 mm = crisp detail on 2 mm walls, ~1-3 s


# ---------------------------------------------------------------------------
# Board + housing parameters
# ---------------------------------------------------------------------------

BOARD_L  = 55.0   # PCB length along X
BOARD_W  = 28.0   # PCB width  along Y
BOARD_T  =  1.6   # PCB thickness
COMP_H   = 10.0   # max component height above PCB

WALL     =  2.0   # wall / floor thickness
FLOOR    =  2.0   # floor thickness
STANDOFF =  2.0   # PCB standoff post height
CLEAR    =  1.0   # XY air-gap around PCB

# Inner cavity (PCB + clearance)
CAV_L = BOARD_L + 2 * CLEAR    # 57 mm
CAV_W = BOARD_W + 2 * CLEAR    # 30 mm
CAV_H = STANDOFF + BOARD_T + COMP_H + CLEAR   # 14.6 mm

# Outer shell
OUT_L = CAV_L + 2 * WALL       # 61 mm
OUT_W = CAV_W + 2 * WALL       # 34 mm
OUT_H = FLOOR + CAV_H          # 16.6 mm


# ---------------------------------------------------------------------------
# 1.  Outer box and inner cavity
# ---------------------------------------------------------------------------

outer_box = BoxShape(
    mn=[-OUT_L/2, -OUT_W/2,  0.0],
    mx=[ OUT_L/2,  OUT_W/2, OUT_H],
)

# Inner void extends 1 mm above OUT_H so the top stays open after subtraction
inner_cavity = BoxShape(
    mn=[-CAV_L/2, -CAV_W/2, FLOOR],
    mx=[ CAV_L/2,  CAV_W/2, OUT_H + 1.0],
)


# ---------------------------------------------------------------------------
# 2.  Feature cutouts (punched through walls / floor)
# ---------------------------------------------------------------------------

# USB Micro-B port on +X face: 10 mm wide, 5 mm tall.
# Must start inside the cavity (x < CAV_L/2) to punch through the full wall,
# and extend past the outer face (x > OUT_L/2) to guarantee a clean opening.
USB_Z_CEN = FLOOR + STANDOFF + BOARD_T / 2 + 2.0    # ~6.8 mm
usb_hole = BoxShape(
    mn=[ CAV_L/2 - 1.0, -5.0, USB_Z_CEN - 2.5],   # starts 1 mm inside cavity
    mx=[ OUT_L/2 + 1.0,  5.0, USB_Z_CEN + 2.5],   # extends 1 mm past outer face
)

# Cable-exit slot on -X face: same principle — span from inside cavity to outside wall.
cable_slot = BoxShape(
    mn=[-OUT_L/2 - 1.0,        -4.0, FLOOR - 0.1],   # 1 mm past outer face
    mx=[-CAV_L/2 + 1.0,         4.0, FLOOR + 6.0],   # 1 mm inside cavity
)

# M3 mounting holes through floor (z-direction cylinders)
# Placed at the four outer corners, clear of the PCB standoffs
SCREW_X = OUT_L / 2 - 3.5
SCREW_Y = OUT_W / 2 - 3.5
screw_holes = [
    CylinderShape(center_xy=[ SCREW_X,  SCREW_Y], z_range=[-0.1, FLOOR + 0.1], radius=1.6),
    CylinderShape(center_xy=[-SCREW_X,  SCREW_Y], z_range=[-0.1, FLOOR + 0.1], radius=1.6),
    CylinderShape(center_xy=[ SCREW_X, -SCREW_Y], z_range=[-0.1, FLOOR + 0.1], radius=1.6),
    CylinderShape(center_xy=[-SCREW_X, -SCREW_Y], z_range=[-0.1, FLOOR + 0.1], radius=1.6),
]


# ---------------------------------------------------------------------------
# 3.  Hollow tray body via DifferenceShape
# ---------------------------------------------------------------------------

# Union all voids, then subtract once from the solid outer box
all_voids = CompoundShape(inner_cavity, usb_hole, cable_slot, *screw_holes)
tray_body = DifferenceShape(outer_box, all_voids)


# ---------------------------------------------------------------------------
# 4.  Internal features (unioned back in)
# ---------------------------------------------------------------------------

# PCB standoff posts: inset enough to clear screw holes at corners
BOSS_X = BOARD_L / 2 - 4.0    # 4 mm inset from PCB edge
BOSS_Y = BOARD_W / 2 - 4.0

standoffs = [
    # z_range starts at 0 (embedded through the full floor) so the standoff
    # post is solidly fused with the floor material all the way down.
    CylinderShape(center_xy=[ BOSS_X,  BOSS_Y], z_range=[0, FLOOR + STANDOFF], radius=2.0),
    CylinderShape(center_xy=[-BOSS_X,  BOSS_Y], z_range=[0, FLOOR + STANDOFF], radius=2.0),
    CylinderShape(center_xy=[ BOSS_X, -BOSS_Y], z_range=[0, FLOOR + STANDOFF], radius=2.0),
    CylinderShape(center_xy=[-BOSS_X, -BOSS_Y], z_range=[0, FLOOR + STANDOFF], radius=2.0),
]

# Snap-fit nubs: small spheres just above PCB height to grip the board
NUB_Z = FLOOR + STANDOFF + BOARD_T + 0.4
nubs = [
    SphereShape([ BOSS_X,  BOSS_Y, NUB_Z], 1.2),
    SphereShape([-BOSS_X,  BOSS_Y, NUB_Z], 1.2),
    SphereShape([ BOSS_X, -BOSS_Y, NUB_Z], 1.2),
    SphereShape([-BOSS_X, -BOSS_Y, NUB_Z], 1.2),
]


# ---------------------------------------------------------------------------
# 5.  Final assembly
# ---------------------------------------------------------------------------

housing = CompoundShape(tray_body, *standoffs, *nubs)


# ---------------------------------------------------------------------------
# 6.  Generate STL
# ---------------------------------------------------------------------------

print("=" * 60)
print("ESP32 DevKitC Housing Tray -- shapek DifferenceShape demo")
print("=" * 60)
print(f"\nBoard footprint:  {BOARD_L:.0f} x {BOARD_W:.0f} mm")
print(f"Housing envelope: {OUT_L:.0f} x {OUT_W:.0f} x {OUT_H:.1f} mm")
print(f"Wall / floor:     {WALL:.0f} / {FLOOR:.0f} mm")
print(f"PCB standoffs:    {STANDOFF:.0f} mm  (PCB bottom at z = {FLOOR+STANDOFF:.0f} mm)")
print(f"\nShapes used:")
print(f"  DifferenceShape  (outer_box - all_voids)")
print(f"  CompoundShape    (tray + 4 standoffs + 4 nubs)")
print(f"  BoxShape         (outer shell, cavity, USB hole, cable slot)")
print(f"  CylinderShape    (standoffs, M3 screw holes)")
print(f"  SphereShape      (snap-fit retaining nubs)")
print(f"\nGenerating mesh at {RESOLUTION_MM} mm resolution ...")

t0 = time.time()
OUT_STL.parent.mkdir(parents=True, exist_ok=True)
result = housing.mesh_stl(resolution_mm=RESOLUTION_MM, out_stl=str(OUT_STL))
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
# 7.  Physical measurement (solid PLA walls -- infill 100%)
# ---------------------------------------------------------------------------

print("\nPhysical measurement (solid PLA, density 1.24 g/cm3):")
m = Measure.from_stl(OUT_STL, density_g_cm3=1.24, infill_pct=100.0)
print(f"  Volume:        {m.volume_mm3:,.0f} mm3")
print(f"  Surface area:  {m.surface_area_mm2:,.0f} mm2")
print(f"  Mass (solid):  {m.mass_g:.1f} g")
cog = m.center_of_gravity_mm
print(f"  Centre of gravity: ({cog[0]:.1f}, {cog[1]:.1f}, {cog[2]:.1f}) mm")
vals, _ = Measure.principal_axes(m)
print(f"  Principal moments of inertia [g*mm2]:")
print(f"    I1={vals[0]:.0f}  I2={vals[1]:.0f}  I3={vals[2]:.0f}")


# ---------------------------------------------------------------------------
# 8.  Render to PNG (vedo offscreen)
# ---------------------------------------------------------------------------

print(f"\nRendering to {OUT_PNG} ...")
try:
    import vedo

    mv = vedo.load(str(OUT_STL))
    mv.color([60, 120, 200]).lighting("plastic")

    plt = vedo.Plotter(
        offscreen=True, size=(1280, 960),
        bg=(20, 20, 28), bg2=(5, 5, 14),
    )
    plt.add(mv)
    plt.show()

    # Top-angled 3/4 view: shows open cavity, all 4 standoffs, USB and cable slots
    cam = plt.camera
    cam.SetPosition(85, -110, 110)
    cam.SetFocalPoint(0, 0, 5)
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
