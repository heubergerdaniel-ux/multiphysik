"""Example 09 -- Pin-Fin Waermetauscher: vollstaendiger Pipeline-Test.

Geometrie: 4x3 Stiftrippen-Array (pin fins) als Waermetauscher-Kern.
           Luftstrom in x-Richtung, Stifte sind z-ausgerichtete Zylinder.

Pipeline:
  1. Geometrie-Erstellung via trimesh (keine picogk-Abhaengigkeit)
  2. CFD-A: Umstroemung der Stifte -> Cd, Re, Geschwindigkeitsfeld
  3. CFD-B: Konvektionskuehlung -> h_conv, T_max, Temperaturfeld
  4. DragCheck (informatorisch)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
DOCS = Path(__file__).resolve().parent.parent / "docs"
DOCS.mkdir(exist_ok=True)

STL_PATH = DOCS / "heat_exchanger_pins.stl"
VEL_PNG  = DOCS / "hx_velocity.png"
TEMP_PNG = DOCS / "hx_temperature.png"

# ---------------------------------------------------------------------------
# 1. Geometrie: 4x3 Pin-Fin-Array
#    Stifte: r=5mm, Hoehe=30mm
#    Raster: dx=15mm in x, dy=15mm in y
#    Kanal:  Stifte bei x=15,30,45,60mm  und  y=-15,0,+15mm
# ---------------------------------------------------------------------------
print("=== Waermetauscher Pin-Fin-Array ===\n")
print("Erstelle Geometrie ...")

pins: list[trimesh.Trimesh] = []
PIN_R   = 5.0    # mm
PIN_H   = 30.0   # mm
X_POS   = [15.0, 30.0, 45.0, 60.0]
Y_POS   = [-15.0, 0.0, 15.0]

for x in X_POS:
    for y in Y_POS:
        cyl = trimesh.creation.cylinder(
            radius=PIN_R,
            height=PIN_H,
            sections=20,
        )
        cyl.apply_translation([x, y, PIN_H / 2])
        pins.append(cyl)

# Kein explizites Kanalgehaeuse -- das LBM-Gitter liefert die Kanalwaende
# (Neumann oben/unten im Solver), Stifte allein erzeugen den Stroemungswiderstand
all_parts = pins
mesh = trimesh.util.concatenate(all_parts)
mesh.export(str(STL_PATH))

print(f"  STL:   {STL_PATH.name}")
print(f"  Stifte: {len(pins)} ({len(X_POS)} x {len(Y_POS)})")
print(f"  Stiftradius: {PIN_R} mm,  Hoehe: {PIN_H} mm")
print(f"  Bounding-Box: {np.array(mesh.bounds[1] - mesh.bounds[0]).round(1)} mm\n")

# ---------------------------------------------------------------------------
# 2. CFD-A: Umstroemung
# ---------------------------------------------------------------------------
print("--- CFD-A: Externe Umstroemung ---")
t0 = time.time()

from picogk_mp.cfd import run_flow
from picogk_mp.cfd.postprocess import save_velocity_png

flow = run_flow(
    stl_path=STL_PATH,
    velocity_m_s=2.0,         # 2 m/s typisch fuer Luefterkuehlung
    flow_direction="x",
    resolution_mm=2.0,        # 2 mm: ~5 Zellen pro Stiftradius
    max_steps=5000,
)

vel_png = save_velocity_png(flow, VEL_PNG)

print(f"  Gitter    : {flow.domain.Ny} x {flow.domain.Nx} Zellen")
print(f"  Re        : {flow.Re:.0f}")
print(f"  Cd        : {flow.Cd:.3f}")
print(f"  Laufzeit  : {flow.elapsed_s:.1f} s")
print(f"  PNG       : {vel_png.name}\n")

# ---------------------------------------------------------------------------
# 3. DragCheck (informatorisch)
# ---------------------------------------------------------------------------
from picogk_mp.cfd.checks import DragCheck
drag_result = DragCheck(Cd_warn=5.0).evaluate({"Cd": flow.Cd, "Re": flow.Re})
status = "OK  " if drag_result.passed else "WARN"
print(f"  [{status}] {drag_result.detail}\n")

# ---------------------------------------------------------------------------
# 4. CFD-B: Konvektionskuehlung
# ---------------------------------------------------------------------------
print("--- CFD-B: Konvektionskuehlung ---")

from picogk_mp.cfd import run_thermal
from picogk_mp.cfd.postprocess import save_temperature_png

# Reale Waermetauscher-Parameter:
# Waermestrom pro Flaeche: ~2000 W/m2 (mittelmaessig belasteter Kuehlkoerper)
# Lufteinlass: 25 C
thermal = run_thermal(
    flow_result=flow,
    heat_flux_W_m2=2000.0,
    T_inlet_C=25.0,
    max_steps=5000,
)

temp_png = save_temperature_png(thermal, TEMP_PNG)

delta_T = thermal.T_surface_avg - 25.0
print(f"  h_conv         : {thermal.h_conv:.1f} W/m2K")
print(f"  T_Einlass      : 25.0 C")
print(f"  T_Oberfl.(avg) : {thermal.T_surface_avg:.2f} C")
print(f"  T_max          : {thermal.T_max:.2f} C")
print(f"  Delta-T        : {delta_T:.2f} K")
print(f"  Laufzeit       : {thermal.elapsed_s:.1f} s")
print(f"  PNG            : {temp_png.name}\n")

# ---------------------------------------------------------------------------
# 5. Bewertung
# ---------------------------------------------------------------------------
print("=== Auswertung ===")
print(f"  Widerstandsbeiwert Cd = {flow.Cd:.3f}  bei Re = {flow.Re:.0f}")
print(f"  Konvektionskoeff.  h  = {thermal.h_conv:.1f} W/m2K")
print(f"  Gesamtlaufzeit        = {time.time() - t0:.1f} s")
print()

# Guetezahl: hoehes h bei niedrigem Cd ist gut
gk = thermal.h_conv / max(flow.Cd, 0.01)
print(f"  Guetezahl h/Cd        = {gk:.1f}  (hoeher = effizienter Waermetauscher)")
print()
print(f"  Bilder gespeichert in: {DOCS.name}/")
