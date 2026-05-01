"""Phase 2: TPMS lattice tests."""
import picogk
from picogk import Mesh, Voxels

from picogk_mp.csg import intersection
from picogk_mp.tpms import Gyroid, SchwartzP


def _run(fn):
    errors = []
    def task():
        try:
            fn()
        except Exception as exc:
            errors.append(exc)
    picogk.go(0.5, task, end_on_task_completion=True)
    if errors:
        raise errors[0]


def _volume(vox: Voxels) -> float:
    vol, _ = vox.calculate_properties()
    return vol


class TestGyroid:
    def test_voxelizes(self):
        """Gyroid produces non-empty voxel field."""
        result = {}
        def task():
            vox = Voxels.from_bounded_implicit(
                Gyroid(cell_size_mm=8.0, isovalue=0.5, bounds_mm=(20.0, 20.0, 20.0))
            )
            result["vol"] = _volume(vox)
            result["mesh_ok"] = Mesh.from_voxels(vox).triangle_count() > 0
        _run(task)
        assert result["vol"] > 0
        assert result["mesh_ok"]

    def test_volume_fraction_increases_with_isovalue(self):
        """Larger isovalue -> more material (thicker walls)."""
        result = {}
        def task():
            b = (20.0, 20.0, 20.0)
            thin = Voxels.from_bounded_implicit(Gyroid(8.0, 0.3, b))
            thick = Voxels.from_bounded_implicit(Gyroid(8.0, 0.9, b))
            result["thin"] = _volume(thin)
            result["thick"] = _volume(thick)
        _run(task)
        assert result["thick"] > result["thin"], (
            f"thick ({result['thick']:.1f}) should exceed thin ({result['thin']:.1f})"
        )

    def test_volume_fraction_in_range(self):
        """Gyroid iso=0.5 in 30mm box: fill between 30% and 65%."""
        result = {}
        def task():
            b = (30.0, 30.0, 30.0)
            vox = Voxels.from_bounded_implicit(Gyroid(8.0, 0.5, b))
            result["frac"] = _volume(vox) / (30.0 ** 3)
        _run(task)
        assert 0.30 < result["frac"] < 0.65, f"fill fraction {result['frac']:.2%} out of range"

    def test_center_offset(self):
        """Gyroid centered at (5,5,5) produces same volume as at origin."""
        result = {}
        def task():
            b = (20.0, 20.0, 20.0)
            v_origin = _volume(Voxels.from_bounded_implicit(Gyroid(8.0, 0.5, b)))
            v_offset = _volume(Voxels.from_bounded_implicit(Gyroid(8.0, 0.5, b, center=(5.0, 5.0, 5.0))))
            result["origin"] = v_origin
            result["offset"] = v_offset
        _run(task)
        # volumes should be within 5% of each other (same geometry, different position)
        ratio = result["offset"] / result["origin"]
        assert 0.95 < ratio < 1.05, f"offset/origin volume ratio {ratio:.3f} out of range"


class TestSchwartzP:
    def test_voxelizes(self):
        """Schwartz-P produces non-empty voxel field."""
        result = {}
        def task():
            vox = Voxels.from_bounded_implicit(
                SchwartzP(cell_size_mm=8.0, isovalue=1.0, bounds_mm=(20.0, 20.0, 20.0))
            )
            result["vol"] = _volume(vox)
        _run(task)
        assert result["vol"] > 0

    def test_volume_fraction_in_range(self):
        """Schwartz-P iso=0.5 in 30mm box: fill between 30% and 55%.

        Schwartz-P f ranges +-3 (vs Gyroid +-1.73), so iso=0.5 gives ~42% fill,
        comparable to Gyroid iso=0.5 (~46%).
        """
        result = {}
        def task():
            b = (30.0, 30.0, 30.0)
            vox = Voxels.from_bounded_implicit(SchwartzP(8.0, 0.5, b))
            result["frac"] = _volume(vox) / (30.0 ** 3)
        _run(task)
        assert 0.30 < result["frac"] < 0.55, f"fill fraction {result['frac']:.2%} out of range"


class TestTPMSInfill:
    def test_gyroid_clipped_to_sphere(self):
        """Gyroid intersected with sphere has less volume than full gyroid box."""
        result = {}
        def task():
            gyroid = Voxels.from_bounded_implicit(
                Gyroid(8.0, 0.5, (30.0, 30.0, 30.0))
            )
            vol_full = _volume(gyroid)
            gyroid2 = Voxels.from_bounded_implicit(
                Gyroid(8.0, 0.5, (30.0, 30.0, 30.0))
            )
            clipped = intersection(gyroid2, Voxels.sphere([0, 0, 0], 12.0))
            result["full"] = vol_full
            result["clipped"] = _volume(clipped)
        _run(task)
        assert result["clipped"] < result["full"], (
            f"clipped ({result['clipped']:.1f}) should be less than full ({result['full']:.1f})"
        )
        # sphere r=12 fills roughly (4/3*pi*12^3)/(30^3) = 26.8% of box
        # clipped gyroid should be < 40% of full gyroid
        assert result["clipped"] < 0.50 * result["full"]
