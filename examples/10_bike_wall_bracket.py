"""Example 10 -- Mountainbike Wall Bracket (Physics-First + Math Surfaces)

Use-case
--------
A PLA wall bracket holds a mountainbike (8 kg) by its handlebar.
The bracket is screwed to the wall at its back face with 4x M6 wood screws.
The storage room reaches up to 35 degrees C in summer.

Geometry philosophy
-------------------
Structural body: mathematically described surfaces via ShapeKernel
  - BoxShape          -- back plate (x = 0..22 mm)
  - PipeShape         -- main arm with Euler-Bernoulli moment taper
                         r(t) = r_min*(1-t)^(1/3)  (t=0 at wall, t=1 at hook)
  - PipeShape         -- diagonal brace, linearly tapered
  - SphereShape       -- smooth junction sphere at brace-arm meeting point
  - PipeShape         -- hook post, constant radius
  - SphereShape       -- rounded hook tip

Interfaces: CylinderXShape cut from body via DifferenceShape
  - 4x M6 SCREW_THROUGH with counterbore, axis="x"

Physics-First workflow
----------------------
1. PhysicsBrief with InterfaceFeatures
2. Requirements derive minimum geometry analytically
3. Body built as CompoundShape; cuts applied via DifferenceShape
4. mesh_stl() -> STL
5. SimEngine verifies Bending, Buckling, ScrewBearing
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from picogk_mp.physics.brief import (
    ComponentType, Constraint, ConstraintType, DesignIntent,
    FailureCriteria, LoadCase, LoadCombination, LoadType,
    Material, MaterialPreset, PhysicsBrief,
)
from picogk_mp.physics.interface import (
    InterfaceFeature, InterfaceType, interface_to_shapek_cuts,
)
from picogk_mp.physics.brief_mapper import (
    brief_to_requirements, brief_to_sim_engine, brief_to_topopt_kwargs,
    suggest_geometry,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOAD_N      = 8.0 * 9.81   # 78.48 N
REACH_MM    = 450.0
HOOK_Z_MM   = 180.0

WALL_PLATE_W  = 160
WALL_PLATE_H  = 280
WALL_PLATE_D  = 22
HOOK_R        = 14
HOOK_DROP     = 60

# ---------------------------------------------------------------------------
# 1. PhysicsBrief with 4x M6 screw interfaces
# ---------------------------------------------------------------------------

M6_IFACES = [
    InterfaceFeature(InterfaceType.SCREW_THROUGH, [10.0, +55.0, 240.0],
                     6.0, float(WALL_PLATE_D), axis="x",
                     clearance_mm=0.5, counterbore_d_mm=11.0, counterbore_depth_mm=6.5,
                     thread_spec="M6", n_count=1, description="Oben rechts"),
    InterfaceFeature(InterfaceType.SCREW_THROUGH, [10.0, -55.0, 240.0],
                     6.0, float(WALL_PLATE_D), axis="x",
                     clearance_mm=0.5, counterbore_d_mm=11.0, counterbore_depth_mm=6.5,
                     thread_spec="M6", n_count=1, description="Oben links"),
    InterfaceFeature(InterfaceType.SCREW_THROUGH, [10.0, +55.0, 40.0],
                     6.0, float(WALL_PLATE_D), axis="x",
                     clearance_mm=0.5, counterbore_d_mm=11.0, counterbore_depth_mm=6.5,
                     thread_spec="M6", n_count=1, description="Unten rechts"),
    InterfaceFeature(InterfaceType.SCREW_THROUGH, [10.0, -55.0, 40.0],
                     6.0, float(WALL_PLATE_D), axis="x",
                     clearance_mm=0.5, counterbore_d_mm=11.0, counterbore_depth_mm=6.5,
                     thread_spec="M6", n_count=1, description="Unten links"),
]

brief = PhysicsBrief(
    source_prompt=(
        "Wandhalterung fuer Mountainbike (8 kg) an Lenker, "
        "4x M6 Holzschrauben, max. Umgebungstemperatur 35 Grad C"
    ),
    material=Material(preset=MaterialPreset.PLA, infill_pct=80),
    load_cases=[
        LoadCase(LoadType.FORCE, LOAD_N, direction=[0.0, 0.0, -1.0],
                 application_point=[REACH_MM, 0.0, HOOK_Z_MM],
                 sf_static=2.0, sf_dynamic=2.0,
                 description="Gewicht Mountainbike am Lenker"),
    ],
    constraints=[
        Constraint(ConstraintType.FIXED_FACE, face="x0",
                   description="Wandmontage (Holzduebelschrauben)"),
    ],
    load_combination=LoadCombination.AND,
    failure=FailureCriteria(
        sf_bending=3.0, sf_buckling=3.0, sf_tension=2.0,
        min_wall_thickness_mm=1.6, max_overhang_deg=45.0,
        max_temperature_c=35.0,
    ),
    intent=DesignIntent(
        component_type=ComponentType.BRACKET,
        keywords=["maximale Steifigkeit"],
        notes="Wandhalterung fuer Innenraum, Hakenform am Ausleger",
    ),
    interfaces=M6_IFACES,
)

assert brief.is_valid(), brief.validate()

print()
print("=" * 60)
print("  PHYSICS BRIEF")
print("=" * 60)
print(f"  ID            : {brief.brief_id}")
print(f"  Betriebslast  : {LOAD_N:.1f} N  ({LOAD_N/9.81:.1f} kg)")
print(f"  Lastarm       : {REACH_MM:.0f} mm")
lc0 = brief.load_cases[0]
print(f"  Bemessungslast: {lc0.design_magnitude:.1f} N  (x sf={lc0.sf_static})")
print(f"  Material      : {brief.material.preset.value}, "
      f"Infill {brief.material.infill_pct:.0f}%, "
      f"rho_eff={brief.material.effective_density_g_cm3():.3f} g/cm3")
print(f"  Schnittstellen: {len(brief.interfaces)}x M6 SCREW_THROUGH")

# ---------------------------------------------------------------------------
# 2. Derive minimum geometry
# ---------------------------------------------------------------------------

reqs = brief_to_requirements(brief)

print()
print("=" * 60)
print("  MINDESTGEOMETRIE (derive)")
print("=" * 60)

derived: dict = {}
for req in reqs:
    d = req.derive(brief)
    print(f"  [{req.name}]")
    for k, v in (d or {}).items():
        print(f"    {k} = {v}")
        if k not in derived or (isinstance(v, (int, float)) and v > derived.get(k, 0)):
            derived[k] = v

r_arm_mm     = derived.get("section_r_min_mm", 20.0)
r_arm_design = math.ceil(r_arm_mm / 5) * 5
r_tip_mm     = max(r_arm_design * 0.4, 8.0)   # tip radius >= 8 mm for hook clearance

print()
print(f"  -> Auslegerradius min. {r_arm_mm:.1f} mm  =>  Design: {r_arm_design} mm")
print(f"     Spitzenradius        : {r_tip_mm:.0f} mm  (Momentenverjuengung)")

# ---------------------------------------------------------------------------
# 3. Build geometry -- ShapeKernel bodies + DifferenceShape for interfaces
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  GEOMETRIEAUFBAU (ShapeKernel)")
print("=" * 60)

from picogk_mp.shapek.base_shape import (
    BoxShape, PipeShape, SphereShape,
    CompoundShape, DifferenceShape,
)
from picogk_mp.shapek.modulation import LineModulation

# --- Back plate (BoxShape -- exact axis-aligned box) ---
plate = BoxShape(
    mn=[0,              -WALL_PLATE_W / 2, 0],
    mx=[WALL_PLATE_D,   WALL_PLATE_W / 2, WALL_PLATE_H],
)

# --- Main arm: Euler-Bernoulli moment taper ---
# M(t) = F * (1-t) * L  =>  optimal r(t) prop. M^(1/3)
# r(0) = r_arm_design at wall (full moment), r(1) = r_tip_mm at hook (moment -> 0)
def _arm_radius(t: float) -> float:
    alpha = (1.0 - t) ** (1.0 / 3.0)
    return r_arm_design * alpha + r_tip_mm * (1.0 - alpha)

arm = PipeShape(
    spine=np.array([
        [WALL_PLATE_D, 0.0, HOOK_Z_MM],
        [REACH_MM,     0.0, HOOK_Z_MM],
    ]),
    radius_mod=LineModulation(func=_arm_radius),
)

# --- Diagonal brace: linear taper from thick (at plate) to thin (at arm) ---
brace_r_wall = r_arm_design * 0.55
brace_r_tip  = r_arm_design * 0.35
brace = PipeShape(
    spine=np.array([
        [WALL_PLATE_D,      0.0, 30.0],
        [REACH_MM * 0.55,   0.0, HOOK_Z_MM],
    ]),
    radius_mod=LineModulation.from_endpoints(brace_r_wall, brace_r_tip),
)

# --- Junction sphere at brace-arm meeting point ---
junction = SphereShape(
    center=[REACH_MM * 0.55, 0.0, HOOK_Z_MM],
    radius=r_arm_design * 0.6,
)

# --- Hook post: constant radius ---
hook = PipeShape(
    spine=np.array([
        [REACH_MM, 0.0, HOOK_Z_MM],
        [REACH_MM, 0.0, HOOK_Z_MM - HOOK_DROP],
    ]),
    radius_mod=LineModulation(value=float(HOOK_R)),
)

# --- Hook tip: rounded end ---
hook_tip = SphereShape(
    center=[REACH_MM, 0.0, HOOK_Z_MM - HOOK_DROP],
    radius=float(HOOK_R),
)

# --- Union of structural body ---
body = CompoundShape(plate, arm, brace, junction, hook, hook_tip)

print(f"  Koerper: BoxShape + 3x PipeShape + 2x SphereShape")
print(f"    BoxShape  : Rueckplatte {WALL_PLATE_D}x{WALL_PLATE_W}x{WALL_PLATE_H} mm")
print(f"    PipeShape : Arm  r={r_arm_design}..{r_tip_mm:.0f} mm (Momentenverjuengung r(t)=(1-t)^1/3)")
print(f"    PipeShape : Diagonalstrebe r={brace_r_wall:.0f}..{brace_r_tip:.0f} mm (linear)")
print(f"    PipeShape : Haken r={HOOK_R} mm (konstant)")

# --- Boolean subtraction: interface cut shapes ---
# CylinderXShape (x-axis aligned) via interface_to_shapek_cuts()
cut_shapes = []
for iface in brief.interfaces:
    cut_shapes.extend(interface_to_shapek_cuts(iface))

print(f"  Interface-Cuts: {len(cut_shapes)} CylinderXShape "
      f"({len(brief.interfaces)}x Bohrung + Senkung)")

final_shape = body
for cut in cut_shapes:
    final_shape = DifferenceShape(final_shape, cut)

print(f"  DifferenceShape: {len(cut_shapes)} Subtraktionen")

# ---------------------------------------------------------------------------
# 4. mesh_stl
# ---------------------------------------------------------------------------

out_dir = Path(__file__).resolve().parent.parent / "docs"
out_dir.mkdir(exist_ok=True)
out_stl = out_dir / "bike_bracket.stl"

result = final_shape.mesh_stl(resolution_mm=1.5, out_stl=str(out_stl))

if result["status"] != "ok":
    print(f"  STL-Fehler: {result}")
    sys.exit(1)

volume_mm3 = result["volume_mm3"]
rho_eff    = brief.material.effective_density_g_cm3()
mass_g     = volume_mm3 * rho_eff / 1000.0

print()
print(f"  STL gespeichert : {out_stl}")
print(f"  Volumen         : {volume_mm3:,.0f} mm3")
print(f"  Druckmasse      : {mass_g:.1f} g  "
      f"(Infill {brief.material.infill_pct:.0f}%, rho_eff={rho_eff:.3f} g/cm3)")

# ---------------------------------------------------------------------------
# 5. Verify
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  PHYSIKVERIFIKATION (verify)")
print("=" * 60)

engine = brief_to_sim_engine(brief)
engine.inject(
    load_reach_mm      = REACH_MM,
    section_r_mm       = float(r_arm_design),
    buckling_length_mm = REACH_MM,
    base_r_mm          = WALL_PLATE_W / 2.0,
    volume_mm3         = float(volume_mm3),
)

results = engine.run(raise_on_failure=False)
for r in results:
    status = "OK  " if r.passed else "FAIL"
    print(f"  [{status}] {r}")

failed = [r for r in results if not r.passed]
if not failed:
    print()
    print("  Alle Anforderungen erfuellt -- STL druckbereit.")
else:
    print()
    print(f"  {len(failed)} Anforderung(en) nicht erfuellt -- Geometrie anpassen!")

print()
