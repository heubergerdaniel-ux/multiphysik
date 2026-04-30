"""Phase 1: CSG boolean operation tests."""
import threading

import pytest
import trimesh

import picogk
from picogk import Mesh, Voxels

from picogk_mp.csg import box_voxels, difference, intersection, smooth_union, union

# picogk.go() must run on the same thread as Library.init, so we use a fixture
# that wraps each test in picogk.go().


def _run(fn):
    """Execute fn inside picogk.go at 0.5 mm voxel size."""
    errors = []

    def task():
        try:
            fn()
        except Exception as exc:
            errors.append(exc)

    picogk.go(0.5, task, end_on_task_completion=True)
    if errors:
        raise errors[0]


# -- helpers ------------------------------------------------------------------

def _volume(vox: Voxels) -> float:
    vol, _ = vox.calculate_properties()
    return vol


def _watertight(vox: Voxels) -> bool:
    m = trimesh.load(
        trimesh.util.wrap_as_stream(b""),  # type: ignore[arg-type]
        file_type="stl",
    )
    # build trimesh from picogk mesh vertices/faces
    pmesh = Mesh.from_voxels(vox)
    verts = [pmesh.get_vertex(i) for i in range(pmesh.vertex_count())]
    tris = [pmesh.get_triangle(i) for i in range(pmesh.triangle_count())]
    tm = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
    return bool(tm.is_watertight)


# -- tests --------------------------------------------------------------------

class TestDifference:
    def test_hollow_sphere_volume(self):
        """Outer r=15 minus inner r=13 -> shell volume near 4939 mm^3."""
        result = {}

        def task():
            outer = Voxels.sphere([0, 0, 0], 15.0)
            inner = Voxels.sphere([0, 0, 0], 13.0)
            shell = difference(outer, inner)
            result["vol"] = _volume(shell)

        _run(task)
        # analytical: 4/3*pi*(15^3-13^3) = 4938.7 mm^3; allow 1% voxel error
        assert abs(result["vol"] - 4938.7) / 4938.7 < 0.01

    def test_inputs_preserved(self):
        """difference() must not mutate its inputs."""
        result = {}

        def task():
            a = Voxels.sphere([0, 0, 0], 10.0)
            b = Voxels.sphere([0, 0, 0], 8.0)
            vol_a_before = _volume(a)
            difference(a, b)
            result["unchanged"] = abs(_volume(a) - vol_a_before) < 1.0

        _run(task)
        assert result["unchanged"]


class TestUnion:
    def test_union_volume_gt_single(self):
        """Union of two overlapping spheres > one sphere."""
        result = {}

        def task():
            a = Voxels.sphere([-8, 0, 0], 10.0)
            b = Voxels.sphere([ 8, 0, 0], 10.0)
            vol_single = _volume(Voxels.sphere([0, 0, 0], 10.0))
            vol_union = _volume(union(a, b))
            result["ok"] = vol_union > vol_single

        _run(task)
        assert result["ok"]

    def test_inputs_preserved(self):
        """union() must not mutate its inputs."""
        result = {}

        def task():
            a = Voxels.sphere([-8, 0, 0], 10.0)
            b = Voxels.sphere([ 8, 0, 0], 10.0)
            vol_a_before = _volume(a)
            union(a, b)
            result["unchanged"] = abs(_volume(a) - vol_a_before) < 1.0

        _run(task)
        assert result["unchanged"]


class TestIntersection:
    def test_lens_volume(self):
        """Intersection of two r=10 spheres, d=16mm -> ~234 mm^3."""
        result = {}

        def task():
            a = Voxels.sphere([-8, 0, 0], 10.0)
            b = Voxels.sphere([ 8, 0, 0], 10.0)
            result["vol"] = _volume(intersection(a, b))

        _run(task)
        # analytical: 2 * pi*h^2*(3r-h)/3 with h=2, r=10 -> 234.6 mm^3
        assert abs(result["vol"] - 234.6) / 234.6 < 0.05

    def test_inputs_preserved(self):
        """intersection() must not mutate its inputs."""
        result = {}

        def task():
            a = Voxels.sphere([-8, 0, 0], 10.0)
            b = Voxels.sphere([ 8, 0, 0], 10.0)
            vol_a_before = _volume(a)
            intersection(a, b)
            result["unchanged"] = abs(_volume(a) - vol_a_before) < 1.0

        _run(task)
        assert result["unchanged"]


class TestBoxVoxels:
    def test_box_volume(self):
        """10x10x10 box at 0.5 mm voxels.

        Voxelization expands each face by ~voxel_size/2, so the measured volume
        is approximately (10 + 0.5)^3 = 1157.6 mm^3, not 1000.  We assert the
        surface-expansion model is accurate within 1%.
        """
        result = {}

        def task():
            vox = box_voxels([0, 0, 0], [10, 10, 10])
            result["vol"] = _volume(vox)

        _run(task)
        voxel_size = 0.5
        expected_voxelized = (10.0 + voxel_size) ** 3  # ~1157.6
        assert abs(result["vol"] - expected_voxelized) / expected_voxelized < 0.01

    def test_sphere_minus_box(self):
        """Open hemisphere: sphere r=12 minus large box -> roughly half sphere."""
        result = {}

        def task():
            sphere = Voxels.sphere([0, 0, 0], 12.0)
            box = box_voxels([0, 0, 12], [24, 24, 24])
            hemi = difference(sphere, box)
            result["vol"] = _volume(hemi)

        _run(task)
        # half sphere: 2/3*pi*12^3 = 3619 mm^3; allow 5% (box cut not perfect)
        assert abs(result["vol"] - 3619.0) / 3619.0 < 0.05
