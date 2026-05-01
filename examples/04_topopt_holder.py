"""Topologieoptimierung des Kopfhoererhalters Arc Pro v2.

Workflow
--------
1. SimEngine fragt fehlende Parameter ab (Kopfhoerermasse).
2. Design-Last = Kopfhoerermasse * g * SAFETY_FACTOR (SF=2).
3. BESO entfernt iterativ Material wo die Dehnungsenergie gering ist.
4. Optimiertes Netz wird als STL + Vorschau-PNG gespeichert.
5. Physics-Checks auf Originaldimension pruefen Kipp-SF.

Ausgabe
-------
docs/headphone_holder_optimised.stl   -- optimiertes Netz
docs/headphone_holder_optimised.png   -- Vorschau-Render
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

from picogk_mp.physics import Param, SimEngine, TippingCheck, StemBendingCheck
from picogk_mp.topopt  import TopoptPipeline, BoundaryConditions

# ------------------------------------------------------------------
# Pfade
# ------------------------------------------------------------------
ROOT    = Path(__file__).parent.parent
STL_IN  = ROOT / "tests" / "fixtures" / "headphone_holder_v2.stl"
STL_OUT = ROOT / "docs" / "headphone_holder_optimised.stl"
PNG_OUT = ROOT / "docs" / "headphone_holder_optimised.png"

# ------------------------------------------------------------------
# Geometrie-Konstanten (aus 03_headphone_holder.py)
# ------------------------------------------------------------------
BASE_R_MM     = 48.0
ARM_REACH_MM  = 82.0
STEM_R_MIN_MM =  7.0
SAFETY_FACTOR =  2.0        # Design-Last = SF * Betriebslast

# ------------------------------------------------------------------
# 1. Nur Betriebsparameter abfragen (Geometrie ist erst nach Topopt bekannt)
# ------------------------------------------------------------------
query_engine = (
    SimEngine()
    .register(
        Param("head_mass_g",   "Kopfhoerermasse",  unit="g",  lo=50, hi=2000),
        Param("infill_pct",    "Infill-Anteil",    unit="%",  default=20, lo=5, hi=100),
        Param("density_g_cm3", "Filament-Dichte",  unit="g/cm3", default=1.24),
        Param("yield_mpa",     "Streckgrenze PLA", unit="MPa",   default=55.0),
    )
)
# Optionale Vorbelegung via Umgebungsvariable (non-interaktiver Modus)
if os.environ.get("HEAD_MASS_G"):
    query_engine.inject(head_mass_g=float(os.environ["HEAD_MASS_G"]))
# run() fragt nur head_mass_g ab wenn kein Default/Value gesetzt -- Rest hat Defaults
query_engine.run(raise_on_failure=False)

head_mass_g = query_engine._params["head_mass_g"].resolved_value
design_mass = head_mass_g * SAFETY_FACTOR   # [g]  mit SF=2
design_F_N  = design_mass * 1e-3 * 9.81    # [N]

print(f"  Betriebslast : {head_mass_g:.0f} g")
print(f"  Design-Last  : {design_mass:.0f} g  (SF={SAFETY_FACTOR})")
print(f"  Design-Kraft : {design_F_N:.2f} N")

# ------------------------------------------------------------------
# 2. Topologieoptimierung
# ------------------------------------------------------------------
if not STL_IN.exists():
    print(f"\nFehler: STL nicht gefunden: {STL_IN}")
    print("Bitte zuerst 'uv run python examples/03_headphone_holder.py' ausfuehren.")
    sys.exit(1)

pipeline = TopoptPipeline(
    STL_IN,
    topopt_h_mm=3.0,       # 3 mm Auflosung fuer Topopt (ca. 10 s/Iteration)
    vol_frac=0.75,         # Ziel: 75% Ausgangsmaterial behalten.
                           # 50% entfernte Teile der Basis -> Kipp-SF < 1.5.
                           # 75% haelt Basisplatte intakt und Konnektivitaet stabil.
    max_iter=40,
    er=0.02,               # 2% Materialabbau pro Iteration (~94 Elemente/Schritt)
    r_filter=1.5,
    add_ratio=0.01,
)

# Randbedingungen mit Design-Last (SF=2 bereits eingerechnet)
bc = BoundaryConditions.headphone_holder(
    *pipeline.grid_shape, pipeline.h, pipeline.offset,
    base_radius_mm=BASE_R_MM,
    arm_tip_mm=(-(ARM_REACH_MM), 0.0, 244.0),
    head_mass_g=design_mass,           # <-- SF=2 bereits eingerechnet
)

STL_OUT.parent.mkdir(parents=True, exist_ok=True)
pipeline.run(bc, out_stl=STL_OUT)

# ------------------------------------------------------------------
# 3. Physics-Checks auf optimiertem Volumen
# ------------------------------------------------------------------
if STL_OUT.exists():
    import trimesh as _tm
    opt_mesh = _tm.load(str(STL_OUT), force="mesh")
    opt_vol  = float(opt_mesh.volume)
    print(f"  Optimiertes Volumen: {opt_vol:.0f} mm3")

    # Neuen Check-Engine mit allen bekannten Werten aufbauen
    check_engine = (
        SimEngine(resolver={})        # kein interaktiver Input mehr
        .register(
            Param("head_mass_g",   "Kopfhoerermasse",  unit="g",     default=head_mass_g),
            Param("infill_pct",    "Infill-Anteil",    unit="%",     default=query_engine._params["infill_pct"].resolved_value),
            Param("density_g_cm3", "Filament-Dichte",  unit="g/cm3", default=query_engine._params["density_g_cm3"].resolved_value),
            Param("yield_mpa",     "Streckgrenze PLA", unit="MPa",   default=query_engine._params["yield_mpa"].resolved_value),
            Param("base_r_mm",     "Basisradius",      unit="mm",    default=BASE_R_MM),
            Param("arm_reach_mm",  "Armreichweite",    unit="mm",    default=ARM_REACH_MM),
            Param("stem_r_min_mm", "Stem-Mindestradius", unit="mm",  default=STEM_R_MIN_MM),
            Param("volume_mm3",    "Druckvolumen",     unit="mm3",   default=opt_vol),
        )
        .add_check(TippingCheck())
        .add_check(StemBendingCheck())
    )
    check_engine.run(raise_on_failure=False)

# ------------------------------------------------------------------
# 4. Vorschau-Render (gleiche Kameraeinstellung wie v2)
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
    print(f"  Vorschau -> {PNG_OUT}")
except Exception as exc:
    print(f"  Render uebersprungen: {exc}")
