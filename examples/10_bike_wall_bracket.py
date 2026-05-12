"""Example 10 -- Mountainbike Wall Bracket (Physics-First)

Use-case
--------
A PLA wall bracket holds a mountainbike (8 kg) by its handlebar.
The bracket is screwed to the wall at its back face.
The storage room reaches up to 35 degrees C in summer.

Physics-First workflow
----------------------
1. Translate the design intent into a PhysicsBrief
2. Requirements derive minimum geometry analytically (before any shape exists)
3. Build a geometry that satisfies the derived constraints
4. Verify the finished geometry with SimEngine
"""
import math
import sys
from pathlib import Path

# Make sure the package is importable when running from the examples/ folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from picogk_mp.physics.brief import (
    ComponentType, Constraint, ConstraintType, DesignIntent,
    FailureCriteria, LoadCase, LoadCombination, LoadType,
    Material, MaterialPreset, PhysicsBrief,
)
from picogk_mp.physics.brief_mapper import (
    brief_to_requirements, brief_to_sim_engine, brief_to_topopt_kwargs,
    suggest_geometry,
)

# ---------------------------------------------------------------------------
# 1. Physics Brief
# ---------------------------------------------------------------------------

# Bike: 8 kg @ handlebar.
# Handlebar is ~450 mm from the wall (horizontal reach).
# Hook height: 180 mm above mounting plate bottom.
LOAD_N      = 8.0 * 9.81   # 78.48 N
REACH_MM    = 450.0         # horizontal arm length
HOOK_Z_MM   = 180.0         # height of hook above bracket base

brief = PhysicsBrief(
    source_prompt=(
        "Wandhalterung fuer Mountainbike (8 kg) an Lenker, "
        "Wandmontage mit Schrauben, max. Umgebungstemperatur 35 Grad C"
    ),
    material=Material(
        preset=MaterialPreset.PLA,
        infill_pct=80,      # structural part -- high infill for reliability
    ),
    load_cases=[
        LoadCase(
            load_type=LoadType.FORCE,
            magnitude=LOAD_N,
            direction=[0.0, 0.0, -1.0],           # gravity: straight down
            application_point=[REACH_MM, 0.0, HOOK_Z_MM],
            sf_static=2.0,                         # design factor
            sf_dynamic=2.0,                        # shock when hanging bike
            description="Gewicht Mountainbike am Lenker",
        ),
    ],
    constraints=[
        Constraint(
            constraint_type=ConstraintType.FIXED_FACE,
            face="x0",      # back face (x=0) screwed to wall
            description="Wandmontage (Holzduebelschrauben)",
        ),
    ],
    load_combination=LoadCombination.AND,
    failure=FailureCriteria(
        sf_bending=3.0,
        sf_buckling=3.0,
        sf_tension=2.0,
        min_wall_thickness_mm=1.6,   # slightly thicker for structural part
        max_overhang_deg=45.0,
        # 35 deg C storage -- PLA glass transition ~60 deg C, margin = 25 K.
        # Not a hard failure at 35 C, but log it for material awareness.
        max_temperature_c=35.0,
    ),
    intent=DesignIntent(
        component_type=ComponentType.BRACKET,
        keywords=["maximale Steifigkeit"],   # bike bracket -- stiffness over weight
        notes="Wandhalterung fuer Innenraum, Hakenform am Ausleger",
    ),
)

assert brief.is_valid(), brief.validate()
print()
print("=" * 60)
print("  PHYSICS BRIEF")
print("=" * 60)
print(f"  ID            : {brief.brief_id}")
print(f"  Betriebslast  : {LOAD_N:.1f} N  ({LOAD_N/9.81:.1f} kg)")
print(f"  Lastarm       : {REACH_MM:.0f} mm  (Wand -> Lenker)")
print(f"  Bemessungslast: {brief.load_cases[0].design_magnitude:.1f} N  (x sf_static={brief.load_cases[0].sf_static})")
print(f"  Material      : {brief.material.preset.value}, "
      f"Infill {brief.material.infill_pct:.0f}%, "
      f"rho_eff={brief.material.effective_density_g_cm3():.3f} g/cm3")
print(f"  Temp.hinweis  : max. {brief.failure.max_temperature_c} C  "
      f"(PLA Tg ~60 C -- Sicherheitsmarge {60 - brief.failure.max_temperature_c:.0f} K)")

# ---------------------------------------------------------------------------
# 2. Derive minimum geometry (Physics -> Geometrie, vor der Form)
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
            # conservative: take the maximum across all requirements
            if k not in derived or v > derived[k]:
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

# Derived r_min to use for geometry:
r_arm_mm = derived.get("section_r_min_mm", 20.0)
print()
print(f"  -> Auslegerradius mindestens {r_arm_mm:.1f} mm  (aus Anforderungsableitung)")
print(f"     Gewaehlter Konstruktionsradius: {math.ceil(r_arm_mm / 5) * 5:.0f} mm "
      f"(auf naechste 5 mm aufgerundet)")

r_arm_design = math.ceil(r_arm_mm / 5) * 5   # round up to nearest 5 mm

# ---------------------------------------------------------------------------
# 4. Build geometry
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  GEOMETRIEAUFBAU")
print("=" * 60)

WALL_PLATE_W = 160      # wall plate width [mm]
WALL_PLATE_H = 280      # wall plate height [mm]
WALL_PLATE_D = 22       # wall plate depth (thickness at wall) [mm]
HOOK_R       = 14       # hook post radius [mm]
HOOK_DROP    = 60       # hook drops 60 mm below arm centreline [mm]

primitives = [
    # Back plate -- contact surface to wall
    {
        "type": "box",
        "min": [0, -WALL_PLATE_W / 2, 0],
        "max": [WALL_PLATE_D, WALL_PLATE_W / 2, WALL_PLATE_H],
    },
    # Main horizontal arm (satisfies r >= r_arm_design)
    {
        "type": "capsule",
        "from": [WALL_PLATE_D, 0, HOOK_Z_MM],
        "to":   [REACH_MM,     0, HOOK_Z_MM],
        "radius_from": r_arm_design,
        "radius_to":   r_arm_design,
    },
    # Diagonal brace (wall plate lower edge -> arm midpoint) -- reduces bending
    {
        "type": "capsule",
        "from": [WALL_PLATE_D,       0, 30],
        "to":   [REACH_MM * 0.55,    0, HOOK_Z_MM],
        "radius_from": r_arm_design * 0.55,
        "radius_to":   r_arm_design * 0.55,
    },
    # Smooth sphere at arm-brace junction
    {
        "type": "sphere",
        "center": [REACH_MM * 0.55, 0, HOOK_Z_MM],
        "radius": r_arm_design * 0.6,
    },
    # Hook: vertical post at tip, drops down
    {
        "type": "capsule",
        "from": [REACH_MM, 0, HOOK_Z_MM],
        "to":   [REACH_MM, 0, HOOK_Z_MM - HOOK_DROP],
        "radius_from": HOOK_R,
        "radius_to":   HOOK_R,
    },
    # Hook tip: rounded end so the handlebar slides on cleanly
    {
        "type": "sphere",
        "center": [REACH_MM, 0, HOOK_Z_MM - HOOK_DROP],
        "radius": HOOK_R,
    },
]

print(f"  {len(primitives)} Primitive:  Rueckplatte, Ausleger, Diagonalstrebe,")
print(f"                  Verbindungskugel, Haken, Hakenkuppe")
print(f"  Auslegerradius: {r_arm_design} mm  (>= {r_arm_mm:.1f} mm Mindest-SF)")

try:
    from picogk_mp.generators.shape import generate_shape_stl

    out_dir = Path(__file__).resolve().parent.parent / "docs"
    out_dir.mkdir(exist_ok=True)
    out_stl = out_dir / "bike_bracket.stl"

    result = generate_shape_stl(
        primitives=primitives,
        resolution_mm=1.5,
        out_stl=str(out_stl),
    )
    if result["status"] != "ok":
        print(f"  STL-Fehler: {result}")
    else:
        volume_mm3 = result["volume_mm3"]
        rho_eff    = brief.material.effective_density_g_cm3()   # g/cm3
        mass_g     = volume_mm3 * rho_eff / 1000.0

        print()
        print(f"  STL gespeichert: {out_stl}")
        print(f"  Volumen        : {volume_mm3:,.0f} mm3")
        print(f"  Druckmasse     : {mass_g:.1f} g  "
              f"(Infill {brief.material.infill_pct:.0f}%, "
              f"rho_eff={rho_eff:.3f} g/cm3)")

        # Estimate section_r from arm (conservative: use design value)
        section_r_actual = r_arm_design

        print()
        print("=" * 60)
        print("  PHYSIKVERIFIKATION (verify)")
        print("=" * 60)

        engine = brief_to_sim_engine(brief)
        engine.inject(
            load_reach_mm      = REACH_MM,
            section_r_mm       = section_r_actual,
            buckling_length_mm = REACH_MM,        # cantilever: knicklange = armlaenge
            base_r_mm          = WALL_PLATE_W / 2,
            volume_mm3         = volume_mm3,
        )
        results = engine.run(raise_on_failure=False)

        for r in results:
            status = "OK" if r.passed else "FAIL"
            print(f"  [{status}] {r}")

        failed = [r for r in results if not r.passed]
        if not failed:
            print()
            print("  Alle Anforderungen erfuellt -- STL druckbereit.")
        else:
            print()
            print(f"  {len(failed)} Anforderung(en) nicht erfuellt -- Geometrie anpassen!")

except ImportError:
    print()
    print("  (picogk_mp.generators nicht verfuegbar -- nur Physics-Analyse ausgegeben)")
    print(f"  Benoetiger Auslegerradius: {r_arm_design} mm")
    print(f"  Primitives fuer generate_shape bereit (siehe Skript).")

print()
