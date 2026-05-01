"""Phase 2: TPMS lattice generation -> STL export + volume report."""
import time
from pathlib import Path

import picogk
from picogk import Mesh, Voxels

from picogk_mp.csg import difference, intersection
from picogk_mp.tpms import Gyroid, SchwartzP

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"
BOX = (30.0, 30.0, 30.0)   # mm, bounding cube for all shapes
CELL = 8.0                  # mm, TPMS unit-cell size


def _export(vox: Voxels, name: str) -> None:
    path = FIXTURES / name
    mesh = Mesh.from_voxels(vox)
    mesh.SaveToStlFile(str(path))
    vol, _ = vox.calculate_properties()
    box_vol = BOX[0] * BOX[1] * BOX[2]
    print(f"  {name}: {path.stat().st_size:>10,} bytes | {vol:>7.1f} mm3 "
          f"({100*vol/box_vol:.1f}% fill) | {mesh.triangle_count()} tri")


def _build() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    print("Phase 2 -- TPMS lattice\n")

    # 1. Gyroid sheet, isovalue=0.5
    t0 = time.time()
    print("1. Gyroid (cell=8mm, iso=0.5)")
    vox_gyroid = Voxels.from_bounded_implicit(
        Gyroid(cell_size_mm=CELL, isovalue=0.5, bounds_mm=BOX)
    )
    _export(vox_gyroid, "phase2_gyroid.stl")
    print(f"   voxelized in {time.time()-t0:.2f}s")

    # 2. Schwartz-P sheet, isovalue=0.5  (P ranges +-3; iso=0.5 ~ 42% fill, matches gyroid)
    t0 = time.time()
    print("2. Schwartz-P (cell=8mm, iso=0.5)")
    vox_schwartz = Voxels.from_bounded_implicit(
        SchwartzP(cell_size_mm=CELL, isovalue=0.5, bounds_mm=BOX)
    )
    _export(vox_schwartz, "phase2_schwartz_p.stl")
    # Note: Schwartz-P function ranges +-3 (vs Gyroid +-1.73), so comparable fill
    # requires iso ≈ 0.5 for P vs 0.5 for Gyroid
    print(f"   voxelized in {time.time()-t0:.2f}s")

    # 3. Gyroid thicker walls (iso=0.9) for comparison
    t0 = time.time()
    print("3. Gyroid thick (cell=8mm, iso=0.9)")
    vox_thick = Voxels.from_bounded_implicit(
        Gyroid(cell_size_mm=CELL, isovalue=0.9, bounds_mm=BOX)
    )
    _export(vox_thick, "phase2_gyroid_thick.stl")
    print(f"   voxelized in {time.time()-t0:.2f}s")

    # 4. Gyroid infill clipped to sphere -- printable demo part
    t0 = time.time()
    print("4. Gyroid infill clipped to sphere (r=14mm)")
    gyroid_for_clip = Voxels.from_bounded_implicit(
        Gyroid(cell_size_mm=CELL, isovalue=0.5, bounds_mm=BOX)
    )
    sphere_shell = Voxels.sphere([0, 0, 0], 14.0)
    vox_infill = intersection(gyroid_for_clip, sphere_shell)
    _export(vox_infill, "phase2_gyroid_sphere_infill.stl")
    print(f"   voxelized in {time.time()-t0:.2f}s")

    # 5. Hollow sphere with gyroid infill inside (outer shell + internal lattice)
    t0 = time.time()
    print("5. Hollow sphere + gyroid infill (r=14 shell 1mm + internal gyroid)")
    outer = Voxels.sphere([0, 0, 0], 14.0)
    inner_void = Voxels.sphere([0, 0, 0], 13.0)
    shell = difference(outer, inner_void)
    gyroid_inner = Voxels.from_bounded_implicit(
        Gyroid(cell_size_mm=CELL, isovalue=0.5, bounds_mm=BOX)
    )
    inner_gyroid_clipped = intersection(gyroid_inner, Voxels.sphere([0, 0, 0], 13.0))
    vox_part = shell.bool_add(inner_gyroid_clipped)
    _export(vox_part, "phase2_sphere_with_gyroid_infill.stl")
    print(f"   voxelized in {time.time()-t0:.2f}s")

    print("\nAll done.")


if __name__ == "__main__":
    picogk.go(0.5, _build, end_on_task_completion=True)
