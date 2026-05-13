# Design Constraints — PicoGK Multiphysik Pipeline

Diese Datei wird am Anfang jeder Session gelesen.
Alle Entwürfe MÜSSEN diese Bedingungen erfüllen, bevor ein STL exportiert wird.

---

## Mechanische Randbedingungen

### Material & Druck
- Material: PLA, Dichte 1.24 g/cm³
- Standard-Infill: 20 % → effektive Dichte 0.248 g/cm³ (= 0.000248 g/mm³)
- Max. Überhang ohne Support: 45°
- Mindest-Wandstärke für Druckbarkeit: 1.2 mm

### Stabilitätskriterien
| Kriterium | Mindest-Sicherheitsfaktor |
|---|---|
| Kippen (Tipping) unter Betriebslast | ≥ 1.5 |
| Biegung am schwächsten Querschnitt | ≥ 3.0 |
| Bruch/Versagen unter Spitzenlast | ≥ 5.0 |

### Standardlasten (falls nicht anders angegeben)
- Punktlast `M_load` [g] am horizontalen Versatz `L_offset` [mm] vom Basis-Achse
- Dynamischer Lastfaktor (Anstoßen, Ablegen): 2.0 × statische Last

---

## Stabilitätsprüfung Kippen — Pflichtformel

Vor jedem STL-Export analytisch prüfen:

```
lever_load  = L_offset - base_r          [mm]  -- Kipphebelarm Last
lever_part  = base_r - |CG_x|            [mm]  -- Gegenhebelarm Bauteil
tip_moment  = M_load    * lever_load     [g·mm]
restore     = M_part    * lever_part     [g·mm]
SF_tip      = restore / tip_moment       [-]   -- muss ≥ 1.5 sein
```

**Bei SF < 1.5: Design ablehnen und Parameter anpassen (base_r, L_offset, Wandstärke).**

Mindest-Basisradius-Formel (direkt lösbar):

```
base_r_min = (M_load * safety * L_offset) / (M_part + M_load * safety)
```

---

## Prozess-Checkliste vor STL-Export

- [ ] Kipp-Sicherheitsfaktor ≥ 1.5 analytisch verifiziert (Wert ausgeben)
- [ ] Schwächster Querschnitt identifiziert und Biegespannung berechnet
- [ ] Keine Überhänge > 45° (visuell oder per Slicer-Vorschau)
- [ ] Mindest-Wandstärke ≥ 1.2 mm überall eingehalten
- [ ] Volumen und Masse ausgegeben (als Plausibilitätscheck)

---

## Voxel & Performance

- Standard-Voxelgröße: 0.5 mm (Kompromiss Qualität/Geschwindigkeit auf T14)
- `Voxels.offset()` vermeiden bei großen Feldern (hängt sich auf)
- TPMS `fSignedDistance` in Python: ~3.5 µs/Aufruf → bei Bedarf auf C-Extension migrieren

---

## Stack (zuletzt geprüft 2026-05-01)

- Python 3.12.13 (uv), pycogk 0.3.0, vedo 2026.6.1, trimesh 4.12.1
- Windows 11 Pro, PowerShell, .NET SDK 8.0.420
- Import: `import picogk` (NICHT `pycogk`)
- Bool-Ops in-place: immer `a.duplicate().bool_add(b)` verwenden

---

## Anhang — Referenzdesign: Kopfhörerhalter

Ein worked example der Pipeline mit konkreten Zahlen. Andere Designs
können andere Zahlen verwenden — die Formeln und SF-Schwellen oben gelten generisch.

**Standardlast:** `M_load = 400 g` am Arm-Endpunkt.

| Parameter | Minimum | Referenz | Maximum |
|---|---|---|---|
| Gesamthöhe | 220 mm | 255 mm | 320 mm |
| Basis-Radius `base_r` | 65 mm | 95 mm | 130 mm |
| Basis-Höhe | 10 mm | 16 mm | 25 mm |
| Lastversatz `L_offset` (horizontal) | 60 mm | 80 mm | 100 mm |
| Min. Querschnittsradius (Stem-Fuß) | 10 mm | 12 mm | 18 mm |

Code-Referenz: [examples/03_headphone_holder.py](examples/03_headphone_holder.py),
[examples/04_topopt_holder.py](examples/04_topopt_holder.py).
