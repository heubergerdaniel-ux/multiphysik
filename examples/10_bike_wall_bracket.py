"""Example 10 -- Mountainbike Wall Bracket (Physics-First + Interfaces)

Use-case
--------
A PLA wall bracket holds a mountainbike (8 kg) by its handlebar.
The bracket is screwed to the wall at its back face with 4x M6 wood screws.
The storage room reaches up to 35 degrees C in summer.

Physics-First workflow
----------------------
1. Translate the design intent into a PhysicsBrief (+ InterfaceFeatures)
2. Requirements derive minimum geometry analytically (before any shape exists)
3. Build a geometry that satisfies the derived constraints:
   - Structural body: SDF primitives (capsule arm with r >= r_min from BendingRequirement)
   - Interfaces: cylinder cut-primitives (4x M6 screw holes)
4. Verify the finished geometry with SimEngine (Bending, Buckling, ScrewBearing)

Geometry philosophy
-------------------
Structural bodies: SDF-based (capsule = constant-radius approximation of tapered arm).
  For a full moment-taper arm use PipeShape + LineModulation from picogk_mp.shapek
  (see brief_to_body_shapes in brief_mapper.py for the analytical formulation).
Interfaces: cylinder cut-primitives -- rotationally symmetric, dimensionally exact.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from picogk_mp.physics.brief import (
    ComponentType, Constraint, ConstraintType, DesignIntent,
    FailureCriteria, LoadCase, LoadCombination, LoadType,
    Material, MaterialPreset, PhysicsBrief,
)
from picogk_mp.physics.interface import (
    InterfaceFeature, InterfaceType,
)
from picogk_mp.physics.brief_mapper import (
    brief_to_requirements, brief_to_sim_engine, brief_to_topopt_kwargs,
    brief_to_interface_primitives, suggest_geometry,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOAD_N      = 8.0 * 9.81   # 78.48 N -- bike weight
REACH_MM    = 450.0         # horizontal arm length [mm]
HOOK_Z_MM   = 180.0         # hook height above bracket base [mm]

WALL_PLATE_W = 160          # wall plate width [mm]
WALL_PLATE_H = 280          # wall plate height [mm]
WALL_PLATE_D = 22           # wall plate depth / screw engagement depth [mm]
HOOK_R       = 14           # hook post radius [mm]
HOOK_DROP    = 60           # hook drop below arm centreline [mm]

# ---------------------------------------------------------------------------
# 1. Physics Brief with 4x M6 screw interfaces
# ---------------------------------------------------------------------------

# Four M6 screws at the corners of the wall plate back face (x = 0, axis = x).
# Clearance hole: d + 0.5 mm for M6 in PLA (looser than metal, slight thermal expansion).
# Counterbore: Torx 30 head (11 mm diameter, 6.5 mm depth).
M6_IFACES = [
    InterfaceFeature(
        feature_type=InterfaceType.SCREW_THROUGH,
        position=[10.0, +55.0, 240.0],
        diameter_mm=6.0,
        depth_mm=float(WALL_PLATE_D),
        axis="x",
        clearance_mm=0.5,
        counterbore_d_mm=11.0,
        counterbore_depth_mm=6.5,
        thread_spec="M6",
        n_count=1,
        description="Obere rechte Wandschraube",
    ),
    InterfaceFeature(
        feature_type=InterfaceType.SCREW_THROUGH,
        position=[10.0, -55.0, 240.0],
        diameter_mm=6.0,
        depth_mm=float(WALL_PLATE_D),
        axis="x",
        clearance_mm=0.5,
        counterbore_d_mm=11.0,
        counterbore_depth_mm=6.5,
        thread_spec="M6",
        n_count=1,
        description="Obere linke Wandschraube",
    ),
    InterfaceFeature(
        feature_type=InterfaceType.SCREW_THROUGH,
        position=[10.0, +55.0, 40.0],
        diameter_mm=6.0,
        depth_mm=float(WALL_PLATE_D),
        axis="x",
        clearance_mm=0.5,
        counterbore_d_mm=11.0,
        counterbore_depth_mm=6.5,
        thread_spec="M6",
        n_count=1,
        description="Untere rechte Wandschraube",
    ),
    InterfaceFeature(
        feature_type=InterfaceType.SCREW_THROUGH,
        position=[10.0, -55.0, 40.0],
        diameter_mm=6.0,
        depth_mm=float(WALL_PLATE_D),
        axis="x",
        clearance_mm=0.5,
        counterbore_d_mm=11.0,
        counterbore_depth_mm=6.5,
        thread_spec="M6",
        n_count=1,
        description="Untere linke Wandschraube",
    ),
]

brief = PhysicsBrief(
    source_prompt=(
        "Wandhalterung fuer Mountainbike (8 kg) an Lenker, "
        "4x M6 Holzschrauben, max. Umgebungstemperatur 35 Grad C"
    ),
    material=Material(
        preset=MaterialPreset.PLA,
        infill_pct=80,
    ),
    load_cases=[
        LoadCase(
            load_type=LoadType.FORCE,
            magnitude=LOAD_N,
            direction=[0.0, 0.0, -1.0],
            application_point=[REACH_MM, 0.0, HOOK_Z_MM],
            sf_static=2.0,
            sf_dynamic=2.0,
            description="Gewicht Mountainbike am Lenker",
        ),
    ],
    constraints=[
        Constraint(
            constraint_type=ConstraintType.FIXED_FACE,
            face="x0",
            description="Wandmontage (Holzduebelschrauben)",
        ),
    ],
    load_combination=LoadCombination.AND,
    failure=FailureCriteria(
        sf_bending=3.0,
        sf_buckling=3.0,
        sf_tension=2.0,
        min_wall_thickness_mm=1.6,
        max_overhang_deg=45.0,
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
print(f"  Lastarm       : {REACH_MM:.0f} mm  (Wand -> Lenker)")
lc0 = brief.load_cases[0]
print(f"  Bemessungslast: {lc0.design_magnitude:.1f} N  (x sf_static={lc0.sf_static})")
print(f"  Material      : {brief.material.preset.value}, "
      f"Infill {brief.material.infill_pct:.0f}%, "
      f"rho_eff={brief.material.effective_density_g_cm3():.3f} g/cm3")
print(f"  Schnittstellen: {len(brief.interfaces)}x M6 SCREW_THROUGH (Wandmontage)")
print(f"  Temp.hinweis  : max. {brief.failure.max_temperature_c} C  "
      f"(PLA Tg ~60 C -- Sicherheitsmarge {60 - brief.failure.max_temperature_c:.0f} K)")

# ---------------------------------------------------------------------------
# 2. Derive minimum geometry
# ---------------------------------------------------------------------------

reqs = brief_to_requirements(brief)

print()
print("=" * 60)
print("  ANFORDERUNGEN & MINDESTGEOMETRIE (derive)")
print("=" * 60)

derived: dict = {}
for req in reqs:
    d = req.derive(brief)
    print(f"  [{req.name}]")
    if d:
        for k, v in d.items():
            print(f"    {k} = {v}")
            if k not in derived or (isinstance(v, (int, float)) and v > derived.get(k, 0)):
                derived[k] = v
    else:
        print("    (kein analytisches Derivat moeglich)")

print()
print("  Massgebliche Mindestgeometrie (Maximum aller Anforderungen):")
for k, v in derived.items():
    print(f"    {k} = {v}")

# ---------------------------------------------------------------------------
# 3. Geometry suggestion
# ---------------------------------------------------------------------------

geo = suggest_geometry(brief)
topopt_kw = brief_to_topopt_kwargs(brief)

print()
print("=" * 60)
print("  GEOMETRIEEMPFEHLUNG")
print("=" * 60)
print(f"  Klasse       : {geo['geometry_class']}")
print(f"  Querschnitt  : {geo['cross_section']}")
print(f"  Begruendung  : {geo['rationale']}")
print(f"  vol_frac     : {topopt_kw['vol_frac']:.0%}  (Topologieoptimierung)")
print(f"  Wandstaerke  : >= {geo['min_wall_mm']} mm")

r_arm_mm = derived.get("section_r_min_mm", 20.0)
print()
print(f"  -> Auslegerradius mindestens {r_arm_mm:.1f} mm  (aus Anforderungsableitung)")
r_arm_design = math.ceil(r_arm_mm / 5) * 5
print(f"     Gewaehlter Konstruktionsradius: {r_arm_design:.0f} mm "
      f"(auf naechste 5 mm aufgerundet)")

# ---------------------------------------------------------------------------
# 4. Build geometry (structural body + interface cut-primitives)
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  GEOMETRIEAUFBAU")
print("=" * 60)

# Structural body -- SDF primitive approach (approximation of tapered arm)
# For exact moment-taper use brief_to_body_shapes(brief, derived) -> PipeShape
solid_primitives = [
    # Back plate -- contact surface to wall
    {
        "type": "box",
        "min": [0, -WALL_PLATE_W / 2, 0],
        "max": [WALL_PLATE_D, WALL_PLATE_W / 2, WALL_PLATE_H],
    },
    # Main horizontal arm (r >= r_arm_design)
    {
        "type": "capsule",
        "from": [WALL_PLATE_D, 0, HOOK_Z_MM],
        "to":   [REACH_MM,     0, HOOK_Z_MM],
        "radius_from": r_arm_design,
        "radius_to":   r_arm_design,
    },
    # Diagonal brace (wall plate lower edge -> arm midpoint)
    {
        "type": "capsule",
        "from": [WALL_PLATE_D,      0, 30],
        "to":   [REACH_MM * 0.55,   0, HOOK_Z_MM],
        "radius_from": r_arm_design * 0.55,
        "radius_to":   r_arm_design * 0.55,
    },
    # Smooth sphere at arm-brace junction
    {
        "type": "sphere",
        "center": [REACH_MM * 0.55, 0, HOOK_Z_MM],
        "radius": r_arm_design * 0.6,
    },
    # Hook: vertical post at tip
    {
        "type": "capsule",
        "from": [REACH_MM, 0, HOOK_Z_MM],
        "to":   [REACH_MM, 0, HOOK_Z_MM - HOOK_DROP],
        "radius_from": HOOK_R,
        "radius_to":   HOOK_R,
    },
    # Hook tip
    {
        "type": "sphere",
        "center": [REACH_MM, 0, HOOK_Z_MM - HOOK_DROP],
        "radius": HOOK_R,
    },
]

# Interface cut-primitives (4x M6 screw holes + counterbores)
cut_primitives = brief_to_interface_primitives(brief)

all_primitives = solid_primitives + cut_primitives

print(f"  Vollkoerper    : {len(solid_primitives)} Primitive")
print(f"  Interface-Cuts : {len(cut_primitives)} Primitive "
      f"({len(M6_IFACES)}x M6 Bohrung + Senkung je 2 Cuts)")
print(f"  Gesamt         : {len(all_primitives)} Primitive")
print(f"  Auslegerradius : {r_arm_design} mm  (>= {r_arm_mm:.1f} mm Mindest-SF)")

try:
    from picogk_mp.generators.shape import generate_shape_stl

    out_dir = Path(__file__).resolve().parent.parent / "docs"
    out_dir.mkdir(exist_ok=True)
    out_stl = out_dir / "bike_bracket.stl"

    result = generate_shape_stl(
        primitives=all_primitives,
        resolution_mm=1.5,
        out_stl=str(out_stl),
    )

    if result["status"] != "ok":
        print(f"  STL-Fehler: {result}")
    else:
        volume_mm3 = result["volume_mm3"]
        rho_eff    = brief.material.effective_density_g_cm3()
        mass_g     = volume_mm3 * rho_eff / 1000.0

        print()
        print(f"  STL gespeichert : {out_stl}")
        print(f"  Volumen         : {volume_mm3:,.0f} mm3")
        print(f"  Druckmasse      : {mass_g:.1f} g  "
              f"(Infill {brief.material.infill_pct:.0f}%, "
              f"rho_eff={rho_eff:.3f} g/cm3)")

        # ---------------------------------------------------------------------------
        # 5. Verify (Bending + Buckling + ScrewBearing)
        # ---------------------------------------------------------------------------

        print()
        print("=" * 60)
        print("  PHYSIKVERIFIKATION (verify)")
        print("=" * 60)

        # brief_to_sim_engine automatically pre-populates n_screws / screw_d_mm /
        # plate_t_mm from brief.interfaces -- only geometry params need inject().
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

except ImportError as exc:
    print()
    print(f"  (picogk_mp.generators nicht verfuegbar: {exc})")
    print(f"  Benoetiger Auslegerradius : {r_arm_design} mm")
    print(f"  Interface-Cuts vorbereitet: {len(cut_primitives)} Primitive")

print()
