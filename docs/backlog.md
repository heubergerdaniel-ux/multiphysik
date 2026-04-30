# Backlog (nicht in Phase 0)

## Phase 1
- Boolean-Operationen-Wrapper (BoolAdd, BoolSubtract, BoolIntersect)
- CSG-Demo: Sphere minus Box = offene Schale

## Phase 2
- TPMS-Lattices (Gyroid, Schwartz-P) via Voxels.render_implicit

## Phase 3
- FEM-Integration: scikit-fem auf Voxel-Mesh
- Waerme-Simulation einfacher Geometrien

## Phase 4
- Kopplung Geometrie <-> FEM-Output

## Phase 5
- TopOpt: SIMP auf Voxeln (eigene Impl.), topy als Referenz
- topy Windows-Kompatibilitaet pruefen

## Phase 6
- CFD: OpenFOAM/Docker oder dolfinx (WSL2 oder Docker Desktop)
- Entscheidung: WSL2 vs. Docker Desktop

## Phase 7
- Multiphysik-Kopplung (Struktur + Waerme + CFD)

## Infrastruktur-Backlog
- Git-LFS fuer Lattice-STLs > 50 MB (ab Phase 2 pruefen)
- C#-Pfad parallel, falls pycogk-Limitierungen auftreten
- pycogk-Viewer-Test (vedo/vtk auf Windows, interaktiv)
