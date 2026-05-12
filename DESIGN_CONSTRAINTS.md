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
- Betriebslast: vom User/Prompt angeben (keine Default-Annahme)
- Dynamischer Lastfaktor (Stoß, Ablegen): 2.0 × statische Last

---

## Stabilitätsprüfung Kippen — Pflichtformel

Vor jedem STL-Export analytisch prüfen (gilt für alle stehenden Teile mit Basisfläche):

```
lever_load  = load_reach - base_r         [mm]  -- Kipphebelarm Last
lever_stand = base_r - |CG_x|            [mm]  -- Gegenhebelarm Körper
tip_moment  = load_mass * lever_load     [g·mm]
restore     = part_mass * lever_stand    [g·mm]
SF_tip      = restore / tip_moment       [-]   -- muss ≥ 1.5 sein
```

**Bei SF < 1.5: Design ablehnen und Parameter anpassen (base_r, load_reach, Wandstärke).**

Mindest-Basisradius-Formel (direkt lösbar):

```
base_r_min = (load_mass * safety * load_reach) / (part_mass + load_mass * safety)
```

---

## Prozess-Checkliste vor STL-Export

- [ ] Kipp-SF ≥ 1.5 analytisch verifiziert (nur bei disc_base Teilen mit Lastversatz)
- [ ] Schwächster Querschnitt identifiziert und Biegespannung ≥ SF 3.0 berechnet
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
