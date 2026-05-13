"""Reference design example: headphone holder ('Arc Pro' v2).

This is ONE example of using the generic multiphysik pipeline (disc-base
geometry + cantilever arm + physics checks).  The same workflow applies
to stands, hooks, brackets, fixtures, etc. -- only the numbers change.

Geometry
--------
  - Base disc r=48 mm, h=14 mm
  - Tapered stem 9 mm -> 7 mm over 220 mm
  - 4-segment S-curve arm with a 30 mm plateau (rest zone)
  - Wide end cap r=9 mm to distribute contact force
"""
import sys
import time
from pathlib import Path

import picogk
from picogk import Lattice, Mesh, VedoViewer, Voxels

from picogk_mp.csg import cylinder_voxels, union
from picogk_mp.physics import (
    Param, SimEngine, TippingCheck, CantileverBendingCheck,
)

OUT     = Path(__file__).parent.parent / "tests"  / "fixtures" / "headphone_holder_v2.stl"
PREVIEW = Path(__file__).parent.parent / "docs"   / "headphone_holder_v2_preview.png"
VOXEL   = 0.5   # mm

# ------------------------------------------------------------------
# Geometry constants (must match _build() below)
# ------------------------------------------------------------------
BASE_R_MM        = 48.0   # NOTE: below stable limit for 20% infill -- kept as
                          # reference; engine will flag and print min. radius
LOAD_OFFSET_MM   = 82.0
MIN_SECTION_R_MM =  7.0   # tapered stem minimum radius at top

# ------------------------------------------------------------------
# Physics engine -- load_mass_g is UNKNOWN -> user will be asked
# ------------------------------------------------------------------
ENGINE = (
    SimEngine()
    .register(
        Param("load_mass_g",           "Lastmasse (Kopfhoerer)", unit="g",     lo=50, hi=2000),
        Param("infill_pct",            "Infill-Anteil",          unit="%",     default=20, lo=5, hi=100),
        Param("density_g_cm3",         "Filament-Dichte",        unit="g/cm3", default=1.24),
        Param("yield_mpa",             "Streckgrenze",           unit="MPa",   default=55.0),
        # geometry -- injected after build
        Param("base_r_mm",             "Basisradius",            unit="mm"),
        Param("load_offset_mm",        "Lastversatz",            unit="mm"),
        Param("min_section_radius_mm", "Min. Querschnittsradius",unit="mm"),
        Param("volume_mm3",            "Druckvolumen",           unit="mm3"),
    )
    .add_check(TippingCheck())
    .add_check(CantileverBendingCheck())
)


def _build(viewer: VedoViewer) -> None:
    # ------------------------------------------------------------------
    # 1. BASE DISC -- solid cylinder, r=48 mm, h=14 mm
    # ------------------------------------------------------------------
    base = cylinder_voxels(center=[0, 0, 7], radius=48, height=14, sections=128)

    # ------------------------------------------------------------------
    # 2. STEM + ARM  -- pure Lattice (native C, sub-second)
    # ------------------------------------------------------------------
    lat = Lattice()
    lat.add_sphere([0, 0, 14], 11.0)
    lat.add_beam([0, 0, 14], [0, 0, 234], 9.0, 7.0)
    lat.add_sphere([0, 0, 234], 10.5)
    lat.add_beam([0,   0, 234], [-28,  0, 246], 9.5, 8.5)
    lat.add_beam([-28, 0, 246], [-60,  0, 250], 8.5, 8.5)   # plateau
    lat.add_beam([-60, 0, 250], [-82,  0, 244], 8.5, 7.5)
    lat.add_sphere([-82, 0, 244], 9.0)

    frame = Voxels.from_lattice(lat)

    # ------------------------------------------------------------------
    # 3. UNION  base disc + lattice frame
    # ------------------------------------------------------------------
    holder = union(base, frame)

    # ------------------------------------------------------------------
    # 4. STL EXPORT
    # ------------------------------------------------------------------
    mesh = Mesh.from_voxels(holder)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mesh.SaveToStlFile(str(OUT))
    vol, _ = holder.calculate_properties()
    print(f"headphone_holder_v2.stl: {mesh.triangle_count()} tri | "
          f"{vol:.0f} mm3 | {OUT.stat().st_size // 1024} KB")

    # ------------------------------------------------------------------
    # 4b. PHYSICS CHECKS -- inject geometry, ask user for unknowns
    # ------------------------------------------------------------------
    ENGINE.inject(
        base_r_mm=BASE_R_MM,
        load_offset_mm=LOAD_OFFSET_MM,
        min_section_radius_mm=MIN_SECTION_R_MM,
        volume_mm3=vol,
    )
    ENGINE.run(raise_on_failure=False)   # warn but don't abort viewer

    # ------------------------------------------------------------------
    # 5. VIEWER
    # ------------------------------------------------------------------
    viewer.add_mesh(mesh, color=[0.22, 0.22, 0.27])
    viewer.SetGroupMaterial(0, [0.22, 0.22, 0.27], fMetallic=0.06, fRoughness=0.62)
    viewer.SetViewAngles(fOrbit=-38, fElevation=20)
    viewer.SetZoom(1.5)
    viewer.request_render()

    time.sleep(3)
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    viewer.RequestScreenShot(str(PREVIEW))
    time.sleep(1)
    print(f"preview -> {PREVIEW}")


if __name__ == "__main__":
    interactive = "--interactive" in sys.argv
    viewer = VedoViewer(title="Kopfhoererhalter Arc Pro v2", offscreen=not interactive)
    eotc   = not interactive
    picogk.go(VOXEL, lambda: _build(viewer), viewer=viewer, end_on_task_completion=eotc)
