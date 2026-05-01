# Phase 2 -- TPMS Lattice Generation

**Datum:** 2026-05-01

## Gelieferte Artefakte

| Datei | Beschreibung |
|---|---|
| `src/picogk_mp/tpms.py` | Gyroid + SchwartzP als IBoundedImplicit |
| `examples/02_tpms_lattice.py` | 5-Shape-Demo: Gyroid, Schwartz-P, dick, Infill, Infill+Shell |
| `tests/test_tpms.py` | 7 pytest-Tests (alle gruen) |

## IBoundedImplicit Interface

Beide TPMS-Klassen erben von `picogk.IBoundedImplicit` und implementieren:
- `fSignedDistance(vecPt) -> float` -- negativ = Material, positiv = Luft
- `oBounds -> (min, max)` -- Bounding-Box fuer den Voxelizer

Sheet-TPMS-Formel (SDF aus Level-Set-Funktion):

    fSignedDistance = |f(x,y,z)| - isovalue

Negativ (= Material) wo |f| < isovalue. Groesseres isovalue -> dickere Waende -> hoehere Volumen-Fraktion.

## Isovalue-Kalibrierung

| Surface | f-Wertebereich | iso | Fuellung |
|---|---|---|---|
| Gyroid | +-1.73 | 0.3 | ~25% |
| Gyroid | +-1.73 | 0.5 | ~46% |
| Gyroid | +-1.73 | 0.9 | ~81% |
| Schwartz-P | +-3.0 | 0.3 | ~22% |
| Schwartz-P | +-3.0 | 0.5 | ~42% |
| Schwartz-P | +-3.0 | 1.0 | ~81% |

Faustregel: iso/f_max ~ gleicher Fuellung. Fuer ~40%: Gyroid iso=0.5, Schwartz-P iso=0.5.

## Demo-Ergebnisse (30x30x30 mm, cell=8mm, voxel=0.5mm)

| Shape | Datei | Groesse | Fuellung | Dreiecke | Zeit |
|---|---|---|---|---|---|
| Gyroid iso=0.5 | phase2_gyroid.stl | 18.3 MB | 46.1% | 366 940 | 60s |
| Schwartz-P iso=0.5 | phase2_schwartz_p.stl | 13.3 MB | 39.4% | 266 496 | 40s |
| Gyroid iso=0.9 | phase2_gyroid_thick.stl | 16.1 MB | 81.1% | 322 092 | 51s |
| Gyroid + Sphere clip | phase2_gyroid_sphere_infill.stl | 5.7 MB | 14.1% | 114 308 | 19s |
| Hollow sphere + Gyroid | phase2_sphere_with_gyroid_infill.stl | 6.5 MB | 19.9% | 130 916 | 21s |

## Kritische Erkenntnisse

### 1. Performance: fSignedDistance ist Pure-Python-Bottleneck

pycogk ruft `fSignedDistance()` per Voxel-Evaluation aus einem nativen C-Loop auf.
Bei 0.5mm Voxeln und 30mm Box: 60^3 = 216 000 Python-Aufrufe.
Gyroid (6 Trig-Calls/Punkt): 60 Sekunden. Schwartz-P (3 Trig-Calls): 40 Sekunden.

**Faustregel:** ~3.5 us pro `fSignedDistance`-Aufruf in Python.

Optimierungsoptionen fuer spaeter:
- Groessere Voxel (1.0mm) fuer Vorschau: 8x schneller (8^3/1^3 = 8-fach weniger Voxel)
- `Voxels.from_scalar_field(ScalarField)` pruefen ob C-seitig schneller
- Numpy-Grid vorberechnen + direktes Voxel-Setzen (wenn API verfuegbar)

### 2. Bool-Ops nach from_bounded_implicit identisch zu Phase 1

`intersection(gyroid_vox, sphere_vox)` funktioniert nahtlos.
Wrapper `duplicate()` bleibt wichtig -- auch TPMS-Voxels werden in-place mutiert.

### 3. Grosse STL-Fixtures (bis 18 MB)

Backlog-Eintrag "Git-LFS ab Phase 2" tritt ein.
Vorlaeufig in git commitet (private Backup-Repo). LFS-Migration wenn Repo >100 MB.

## Test-Ergebnis

```
7 passed in 9.19s
```

## Bereit fuer Phase 3

Ja. scikit-fem fuer FEM auf Voxel-Mesh.
Naechster Schritt: Voxel-Mesh -> FEM-Mesh konvertieren, Waerme-Simulation.
