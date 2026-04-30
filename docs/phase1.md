# Phase 1 -- CSG Boolean-Operationen

**Datum:** 2026-04-30

## Gelieferte Artefakte

| Datei | Beschreibung |
|---|---|
| `src/picogk_mp/csg.py` | Nicht-destruktive CSG-Wrapper + box_voxels-Utility |
| `examples/01_csg_boolean.py` | 5-Operations-Demo: Differenz, Union, Schnitt, Smooth-Union, Sphere-Box |
| `tests/test_csg.py` | 8 pytest-Tests (alle gruen) |

## Kritische Erkenntnisse

### 1. Bool-Ops sind in-place (mutieren self)

`Voxels.bool_add(other)` mutiert `self` UND gibt `self` zurueck.
Konsequenz: Alle Wrapper in `csg.py` rufen `a.duplicate()` vor der Operation auf.
Ohne duplicate() verbraucht jede Op das Input-Objekt -- subtiler Bug.

### 2. bool_add_smooth nicht verfuegbar

`Voxels_BoolAddSmooth` fehlt in der pycogk-0.3.0-Runtime (nicht im nativen DLL-Symbol).
Fallback: plain `bool_add`. In `csg.py` transparent abgefangen via `NotImplementedError`.

### 3. Voxelisierungs-Oberflaechenexpansion bei Boxen

Jede flache Flaeche wird um ~voxel_size/2 aufgeblasen.
Formel: `vol_vox(L) approx (L + voxel_size)^3` fuer ein Wuerfel der Seitenlaenge L.
Bei L=10mm, voxel=0.5mm: 15.8% Uebervolumen. Sphären sind viel besser (0.3%).
Der Box-Test prueft das Expansions-Modell, nicht den Nominal-Wert.

### 4. Windows-Konsole cp1252

Unicode-Pfeile (U+2192 etc.) crashen `print()` auf Windows-cp1252-Terminals.
Fix: ASCII-Ersatz (`->` statt `-->`).

## Ergebnisse Demo

| Shape | Volume (mm3) | Analytisch | Delta |
|---|---|---|---|
| Hollow sphere (r15 - r13) | 4954.2 | 4938.7 | 0.3% |
| Union (2x sphere r10, d=16mm) | 8117.1 | ~8144 | 0.3% |
| Intersection (lens) | 233.1 | 234.6 | 0.6% |
| Open hemisphere (sphere r12 - box) | 3499.0 | ~3619 | 3.3% |

## Test-Ergebnis

```
8 passed in 2.97s
```

## Backlog-Update

- box_voxels via trimesh-STL-Umweg funktioniert (kein Implicit noetig)
- Smooth Union: Runtime-Limitation, in Phase 5 neu pruefen (pycogk-Update?)
- TPMS-Lattices: Voxels.from_bounded_implicit(IBoundedImplicit) vorhanden -- Phase 2 ready

## Bereit fuer Phase 2

Ja. TPMS-Gyroid via `IBoundedImplicit` + `Voxels.from_bounded_implicit()`.
