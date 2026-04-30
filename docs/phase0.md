# Phase 0 — Bootstrap Report

**Datum:** 2026-04-30  
**Hardware:** ThinkPad T14, AMD Ryzen Pro 5, Windows 11 Pro 10.0.26200

## Installierte Versionen

| Komponente | Version |
|---|---|
| Windows | 11 Pro 10.0.26200 |
| winget | 1.28.240 |
| git | 2.53.0.windows.2 |
| .NET SDK | 8.0.420 |
| uv | 0.10.9 |
| Python (uv-managed) | 3.12.13 |
| pycogk (PyPI-Paket) | 0.3.0 |
| trimesh | 4.12.1 |
| vedo | 2026.6.1 |
| vtk | 9.6.1 |

## API-Anpassungen ggü. Plan

Der ursprüngliche Smoke-Test im Plan war ein Annahmegerüst.
Tatsächliche pycogk v0.3.0 API:

| Plan-Annahme | Realität |
|---|---|
| `import picogk` (Modul) | korrekt — `pycogk` ist der PyPI-Name, `picogk` der Import |
| `Library(0.5)` als Context-Manager | NEIN — kein CM. Einstiegspunkt: `picogk.go(voxel_size_mm, task_fn)` |
| `Sphere(center=..., radius=...)` | NEIN — kein Sphere-Standalone. `Voxels.sphere(center=[x,y,z], radius=r)` |
| `Mesh.from_voxels(sdf)` | korrekt |
| `mesh.export_stl(path)` | NEIN — heißt `mesh.SaveToStlFile(path)` |
| `Vector3(x,y,z)` | NEIN — `Vector3Like = Sequence[float]`, einfach `[x,y,z]` übergeben |

Korrigiertes Pattern:

```python
def _build():
    vox = Voxels.sphere(center=[0.0, 0.0, 0.0], radius=10.0)
    mesh = Mesh.from_voxels(vox)
    mesh.SaveToStlFile(str(OUT))

picogk.go(0.5, _build, end_on_task_completion=True)
```

## Smoke-Test Ergebnis

```
wrote tests/fixtures/phase0_sphere.stl (747,084 bytes, 14940 triangles)
watertight: True
volume mm³: 4176.8  (erwartet: 4188.8, Delta 0.3% — Voxelisierungs-Artefakt, normal)
```

## End-to-End Verifikation

| Check | Ergebnis |
|---|---|
| `dotnet --list-sdks` | `8.0.420` ok |
| `uv run python --version` | `Python 3.12.13` ok |
| `import picogk` | ok |
| Smoke-Test exit 0, STL > 0 | ok (747 KB) |
| STL watertight | True ok |
| Volumen ~ 4189 mm³ | 4176.8 ok |

## Lessons Learned

1. **pycogk vs. picogk:** PyPI-Paket heißt `pycogk`, Import-Modul heißt `picogk`. Nicht verwechseln.
2. **Kein Context-Manager:** Empfohlener Einstiegspunkt ist `picogk.go()`.
3. **Kein `Sphere`-Typ:** Sphere-Voxelisierung ist Methode `Voxels.sphere()`.
4. **Viewer:** vedo/vtk installiert (77 MB), headless reicht für Phase 0.
5. **Kompatibilität:** pycogk v0.3.0 + PicoGK Runtime — kein Inkompatibilitätsproblem festgestellt.

## Bereit für Phase 1

Ja. Boolean-Operationen (BoolAdd, BoolSubtract, BoolIntersect) sind in der Voxels-API vorhanden.
Naechster Schritt: einfache CSG-Operationen.
