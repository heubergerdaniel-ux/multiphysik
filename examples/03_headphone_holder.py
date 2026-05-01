"""Headphone holder -- 'Arc Pro' v2 (research-informed redesign).

Design rationale vs v1 (pure-lattice spider base):
  - Height 170mm -> 255mm: clears ear cups of full-size headphones above desk
  - Spider base -> solid cylinder disc r=48mm h=14mm: stable, premium look
  - Stem tapers 9mm->7mm (slender/elegant vs uniform 10mm cylinder)
  - Hook arm: 4-segment smooth S-curve with 30mm plateau zone where
    headband actually rests (v1 had no flat rest zone, load on single point)
  - Wider end cap r=9mm to distribute contact force
  - Junction sphere enlarged to produce smooth visual transition

Market reference dimensions absorbed:
  - Typical premium stand height: 250-300 mm (e.g. 8-12 inch)
  - Typical base footprint: 90-120 mm diameter
  - Arm reach: 80-100 mm; arm must clear ear pads ~80 mm diameter each
  - Stable rule of thumb: base_radius > height / 5
"""
import sys
import time
from pathlib import Path

import picogk
from picogk import Lattice, Mesh, VedoViewer, Voxels

from picogk_mp.csg import cylinder_voxels, union
from picogk_mp.physics import Param, SimEngine, TippingCheck, StemBendingCheck

OUT     = Path(__file__).parent.parent / "tests"  / "fixtures" / "headphone_holder_v2.stl"
PREVIEW = Path(__file__).parent.parent / "docs"   / "headphone_holder_v2_preview.png"
VOXEL   = 0.5   # mm

# ------------------------------------------------------------------
# Geometry constants (must match _build() below)
# ------------------------------------------------------------------
BASE_R_MM      = 48.0   # NOTE: below stable limit for 20% infill -- kept as
                        # reference; engine will flag and print min. radius
ARM_REACH_MM   = 82.0
STEM_R_MIN_MM  =  7.0   # tapered stem minimum radius at top

# ------------------------------------------------------------------
# Physics engine -- head_mass_g is UNKNOWN -> user will be asked
# ------------------------------------------------------------------
ENGINE = (
    SimEngine()
    .register(
        Param("head_mass_g",   "Kopfhoerermasse",    unit="g",     lo=50, hi=2000),
        Param("infill_pct",    "Infill-Anteil",      unit="%",     default=20, lo=5, hi=100),
        Param("density_g_cm3", "Filament-Dichte",    unit="g/cm3", default=1.24),
        Param("yield_mpa",     "Streckgrenze",       unit="MPa",   default=55.0),
        # geometry -- injected after build
        Param("base_r_mm",     "Basisradius",        unit="mm"),
        Param("arm_reach_mm",  "Armreichweite",      unit="mm"),
        Param("stem_r_min_mm", "Stem-Mindestradius", unit="mm"),
        Param("volume_mm3",    "Druckvolumen",       unit="mm3"),
    )
    .add_check(TippingCheck())
    .add_check(StemBendingCheck())
)


def _build(viewer: VedoViewer) -> None:
    # ------------------------------------------------------------------
    # 1. BASE DISC -- solid cylinder, r=48 mm, h=14 mm
    #    Provides mass/stability. trimesh->mesh->voxels: instant, no offset.
    #    Sits at z=0..14 (centre at z=7).
    # ------------------------------------------------------------------
    base = cylinder_voxels(center=[0, 0, 7], radius=48, height=14, sections=128)

    # ------------------------------------------------------------------
    # 2. STEM + ARM  -- pure Lattice (native C, sub-second)
    # ------------------------------------------------------------------
    lat = Lattice()

    # Anchor knob: merges base disc top with stem foot smoothly
    lat.add_sphere([0, 0, 14], 11.0)

    # Stem: tapered 9mm -> 7mm over 220 mm (elegant, slender)
    lat.add_beam([0, 0, 14], [0, 0, 234], 9.0, 7.0)

    # Junction at stem top: larger sphere for smooth visual transition
    lat.add_sphere([0, 0, 234], 10.5)

    # Hook arm: 4-segment S-curve
    #   Seg 1: sweep out + rise         (0,0,234) -> (-28, 0, 246)
    #   Seg 2: plateau (rest zone)      (-28,0,246) -> (-60, 0, 250)
    #   Seg 3: gentle hook-down         (-60,0,250) -> (-82, 0, 244)
    lat.add_beam([0,   0, 234], [-28,  0, 246], 9.5, 8.5)
    lat.add_beam([-28, 0, 246], [-60,  0, 250], 8.5, 8.5)   # plateau
    lat.add_beam([-60, 0, 250], [-82,  0, 244], 8.5, 7.5)

    # End cap: wide sphere distributes contact force, prevents slipping
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
        arm_reach_mm=ARM_REACH_MM,
        stem_r_min_mm=STEM_R_MIN_MM,
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
