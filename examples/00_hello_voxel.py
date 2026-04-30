"""Phase 0 smoke test: voxel sphere -> STL export (headless, no viewer)."""
from pathlib import Path

import picogk
from picogk import Voxels, Mesh

OUT = Path(__file__).parent.parent / "tests" / "fixtures" / "phase0_sphere.stl"


def _build() -> None:
    vox = Voxels.sphere(center=[0.0, 0.0, 0.0], radius=10.0)
    mesh = Mesh.from_voxels(vox)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mesh.SaveToStlFile(str(OUT))
    size = OUT.stat().st_size
    print(f"wrote {OUT} ({size:,} bytes, {mesh.triangle_count()} triangles)")


if __name__ == "__main__":
    picogk.go(0.5, _build, end_on_task_completion=True)
