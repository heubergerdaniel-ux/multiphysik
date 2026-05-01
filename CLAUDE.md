# Claude Instructions — multiphysik

## Pflichtlektüre zu Sessionbeginn

Lies immer zuerst:
1. `DESIGN_CONSTRAINTS.md` — mechanische Randbedingungen, Sicherheitsfaktoren, Prozess-Checkliste
2. `.claude/projects/C--Users-user-multiphysik/memory/project_phase0.md` — API-Fakten, Stack, Phasenstatus

## Arbeitsregeln

- Vor jedem STL-Export: Kipp-SF analytisch berechnen und ausgeben. Abbruch bei SF < 1.5.
- Keine Unicode-Zeichen in `print()` (Windows cp1252 crasht).
- Bool-Ops immer mit `a.duplicate()` (in-place Mutation, siehe Constraints).
- `Voxels.offset()` bei großen Geometrien NICHT verwenden.
- Neue Phasen/Features: erst `DESIGN_CONSTRAINTS.md` gegen den Plan prüfen.
