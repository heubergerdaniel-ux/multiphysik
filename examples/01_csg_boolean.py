"""Phase 1: CSG operations -> STL export + volume report."""
from pathlib import Path

import picogk
from picogk import Mesh, Voxels

from picogk_mp.csg import box_voxels, difference, intersection, smooth_union, union

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"


def _export(vox: Voxels, name: str) -> None:
    path = FIXTURES / name
    mesh = Mesh.from_voxels(vox)
    mesh.SaveToStlFile(str(path))
    vol, _ = vox.calculate_properties()
    print(f"  {name}: {path.stat().st_size:>9,} bytes | volume {vol:>8.1f} mm³ | {mesh.triangle_count()} tri")


def _build() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    print("Phase 1 — CSG boolean operations\n")

    # 1. Hollow sphere — outer(r=15) MINUS inner(r=13)  → 2 mm wall shell
    outer = Voxels.sphere([0, 0, 0], 15.0)
    inner = Voxels.sphere([0, 0, 0], 13.0)
    print("1. Difference  (hollow sphere, 2 mm wall)")
    _export(difference(outer, inner), "phase1_hollow_sphere.stl")

    # 2. Union — two overlapping spheres (centers ±8 mm, r=10)
    a = Voxels.sphere([-8, 0, 0], 10.0)
    b = Voxels.sphere([ 8, 0, 0], 10.0)
    print("2. Union       (two overlapping spheres)")
    _export(union(a, b), "phase1_union.stl")

    # 3. Intersection — same pair → lens / vesica
    print("3. Intersection (lens / vesica piscis)")
    _export(intersection(a, b), "phase1_intersect.stl")

    # 4. Smooth union — falls back to plain union in pycogk 0.3.0 (runtime missing)
    print("4. Smooth union (4 mm blend — falls back to plain union in this runtime)")
    _export(smooth_union(a, b, blend_mm=4.0), "phase1_smooth_union.stl")

    # 5. Sphere–box difference — sphere r=12 minus box 14×14×14, offset +z
    sphere = Voxels.sphere([0, 0, 0], 12.0)
    box = box_voxels([0, 0, 12], [24, 24, 24])
    print("5. Difference  (sphere minus box -> open hemisphere)")
    _export(difference(sphere, box), "phase1_open_hemisphere.stl")

    print("\nAll done.")


if __name__ == "__main__":
    picogk.go(0.5, _build, end_on_task_completion=True)
