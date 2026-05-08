"""Unit tests for the shapek module.

Phase A tests: pure numpy/scipy -- no picogk.go context required.
Covers: LocalFrame, LineModulation, ControlPointSpline, SurfaceModulation,
        all SDF primitives, BaseShape.mesh_stl(), Measure.from_stl().
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"


def pts(*rows) -> np.ndarray:
    """Shorthand: pts([x,y,z], ...) -> (N,3) float64 array."""
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# LocalFrame
# ---------------------------------------------------------------------------

class TestLocalFrame:
    def test_world_frame_identity(self):
        from picogk_mp.shapek.frame import LocalFrame
        f  = LocalFrame.world()
        p  = pts([1, 2, 3], [4, 5, 6])
        assert np.allclose(f.to_world(f.to_local(p)), p)

    def test_roundtrip_arbitrary_frame(self):
        from picogk_mp.shapek.frame import LocalFrame
        f = LocalFrame(origin=[10, -5, 3], tangent=[1, 1, 0], normal=[0, 0, 1])
        p = pts([0, 0, 0], [100, 200, 300])
        assert np.allclose(f.to_world(f.to_local(p)), p, atol=1e-10)

    def test_along_segment_tangent(self):
        from picogk_mp.shapek.frame import LocalFrame
        f = LocalFrame.along_segment([0, 0, 0], [0, 0, 10])
        assert np.allclose(f.T, [0, 0, 1])

    def test_axes_orthonormal(self):
        from picogk_mp.shapek.frame import LocalFrame
        f = LocalFrame(origin=[0, 0, 0], tangent=[1, 2, 3], normal=[4, 5, 6])
        assert abs(np.dot(f.T, f.N)) < 1e-12
        assert abs(np.dot(f.T, f.B)) < 1e-12
        assert abs(np.dot(f.N, f.B)) < 1e-12
        assert abs(np.linalg.norm(f.T) - 1) < 1e-12
        assert abs(np.linalg.norm(f.N) - 1) < 1e-12
        assert abs(np.linalg.norm(f.B) - 1) < 1e-12

    def test_binormal_right_hand(self):
        from picogk_mp.shapek.frame import LocalFrame
        f = LocalFrame.world()
        assert np.allclose(f.B, np.cross(f.T, f.N))


# ---------------------------------------------------------------------------
# ControlPointSpline
# ---------------------------------------------------------------------------

class TestControlPointSpline:
    def test_at_control_points_exact(self):
        from picogk_mp.shapek.modulation import ControlPointSpline
        sp = ControlPointSpline([(0.0, 0.0), (0.5, 10.0), (1.0, 0.0)])
        assert abs(sp.evaluate(0.0)  - 0.0)  < 1e-9
        assert abs(sp.evaluate(0.5)  - 10.0) < 1e-9
        assert abs(sp.evaluate(1.0)  - 0.0)  < 1e-9

    def test_evaluate_array_shape(self):
        from picogk_mp.shapek.modulation import ControlPointSpline
        sp  = ControlPointSpline([(0, 0), (1, 5)])
        out = sp.evaluate_array(np.linspace(0, 1, 50))
        assert out.shape == (50,)

    def test_fewer_than_two_points_raises(self):
        from picogk_mp.shapek.modulation import ControlPointSpline
        with pytest.raises(ValueError):
            ControlPointSpline([(0.5, 1.0)])


# ---------------------------------------------------------------------------
# LineModulation
# ---------------------------------------------------------------------------

class TestLineModulation:
    def test_constant(self):
        from picogk_mp.shapek.modulation import LineModulation
        m = LineModulation.constant(7.0)
        assert m.at(0.0) == pytest.approx(7.0)
        assert m.at(0.5) == pytest.approx(7.0)
        assert m.at(1.0) == pytest.approx(7.0)

    def test_from_endpoints_midpoint(self):
        from picogk_mp.shapek.modulation import LineModulation
        m = LineModulation.from_endpoints(3.0, 7.0)
        assert m.at(0.0) == pytest.approx(3.0, abs=1e-6)
        assert m.at(0.5) == pytest.approx(5.0, abs=1e-6)
        assert m.at(1.0) == pytest.approx(7.0, abs=1e-6)

    def test_from_function(self):
        from picogk_mp.shapek.modulation import LineModulation
        m = LineModulation.from_function(lambda t: t * 10)
        assert m.at(0.3) == pytest.approx(3.0)

    def test_at_array_shape(self):
        from picogk_mp.shapek.modulation import LineModulation
        m = LineModulation.constant(5.0)
        out = m.at_array(np.linspace(0, 1, 100))
        assert out.shape == (100,)
        assert np.all(out == 5.0)

    def test_from_control_points(self):
        from picogk_mp.shapek.modulation import LineModulation
        m = LineModulation.from_control_points([(0, 0), (0.5, 10), (1, 0)])
        assert m.at(0.5) == pytest.approx(10.0, abs=1e-6)

    def test_exactly_one_required(self):
        from picogk_mp.shapek.modulation import LineModulation
        with pytest.raises(ValueError):
            LineModulation()
        with pytest.raises(ValueError):
            LineModulation(value=1.0, func=lambda t: t)


# ---------------------------------------------------------------------------
# SurfaceModulation
# ---------------------------------------------------------------------------

class TestSurfaceModulation:
    def test_constant(self):
        from picogk_mp.shapek.modulation import SurfaceModulation
        m = SurfaceModulation.constant(3.5)
        assert m.at(0.2, 0.8) == pytest.approx(3.5)

    def test_function(self):
        from picogk_mp.shapek.modulation import SurfaceModulation
        m = SurfaceModulation(func=lambda u, v: u + v)
        assert m.at(0.3, 0.7) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SDF primitives
# ---------------------------------------------------------------------------

class TestSdfSphere:
    def test_on_surface(self):
        from picogk_mp.shapek.primitives import sdf_sphere
        v = sdf_sphere(pts([10, 0, 0]), [0, 0, 0], 10.0)
        assert v[0] == pytest.approx(0.0, abs=1e-9)

    def test_inside_negative(self):
        from picogk_mp.shapek.primitives import sdf_sphere
        v = sdf_sphere(pts([0, 0, 0]), [0, 0, 0], 10.0)
        assert v[0] == pytest.approx(-10.0)

    def test_outside_positive(self):
        from picogk_mp.shapek.primitives import sdf_sphere
        v = sdf_sphere(pts([20, 0, 0]), [0, 0, 0], 10.0)
        assert v[0] == pytest.approx(10.0)


class TestSdfBox:
    def test_interior_negative(self):
        from picogk_mp.shapek.primitives import sdf_box
        v = sdf_box(pts([0, 0, 0]), [-5, -5, -5], [5, 5, 5])
        assert v[0] < 0

    def test_on_face(self):
        from picogk_mp.shapek.primitives import sdf_box
        v = sdf_box(pts([5, 0, 0]), [-5, -5, -5], [5, 5, 5])
        assert v[0] == pytest.approx(0.0, abs=1e-9)

    def test_outside(self):
        from picogk_mp.shapek.primitives import sdf_box
        v = sdf_box(pts([10, 0, 0]), [-5, -5, -5], [5, 5, 5])
        assert v[0] == pytest.approx(5.0)


class TestSdfCapsule:
    def test_midpoint_surface(self):
        from picogk_mp.shapek.primitives import sdf_capsule
        # Uniform capsule along z, radius 5
        mid = [5, 0, 5]   # midpoint on surface
        v = sdf_capsule(pts(mid), [0, 0, 0], [0, 0, 10], 5.0, 5.0)
        assert v[0] == pytest.approx(0.0, abs=1e-9)

    def test_center_inside(self):
        from picogk_mp.shapek.primitives import sdf_capsule
        v = sdf_capsule(pts([0, 0, 5]), [0, 0, 0], [0, 0, 10], 5.0, 5.0)
        assert v[0] < 0


class TestSdfCylinder:
    def test_on_lateral_surface(self):
        from picogk_mp.shapek.primitives import sdf_cylinder
        v = sdf_cylinder(pts([5, 0, 5]), [0, 0], [0, 10], 5.0)
        assert v[0] == pytest.approx(0.0, abs=1e-9)

    def test_on_top_face(self):
        from picogk_mp.shapek.primitives import sdf_cylinder
        v = sdf_cylinder(pts([0, 0, 10]), [0, 0], [0, 10], 5.0)
        assert v[0] == pytest.approx(0.0, abs=1e-9)

    def test_center_inside(self):
        from picogk_mp.shapek.primitives import sdf_cylinder
        v = sdf_cylinder(pts([0, 0, 5]), [0, 0], [0, 10], 5.0)
        assert v[0] < 0


class TestSdfCone:
    def test_base_center_on_surface(self):
        """Base center is on the flat base disc of the cone."""
        from picogk_mp.shapek.primitives import sdf_cone
        v = sdf_cone(pts([0, 0, 10]), [0, 0, 0], [0, 0, 10], 5.0)
        assert v[0] == pytest.approx(0.0, abs=0.01)

    def test_base_edge_on_surface(self):
        """Base circle edge should be on the surface."""
        from picogk_mp.shapek.primitives import sdf_cone
        v = sdf_cone(pts([5, 0, 10]), [0, 0, 0], [0, 0, 10], 5.0)
        assert v[0] == pytest.approx(0.0, abs=0.01)

    def test_lateral_midpoint_on_surface(self):
        """Midpoint of lateral surface: r = r_base/2, z = h/2."""
        from picogk_mp.shapek.primitives import sdf_cone
        v = sdf_cone(pts([2.5, 0, 5]), [0, 0, 0], [0, 0, 10], 5.0)
        assert v[0] == pytest.approx(0.0, abs=0.01)

    def test_apex_on_surface(self):
        from picogk_mp.shapek.primitives import sdf_cone
        v = sdf_cone(pts([0, 0, 0]), [0, 0, 0], [0, 0, 10], 5.0)
        assert v[0] == pytest.approx(0.0, abs=0.01)

    def test_interior_negative(self):
        from picogk_mp.shapek.primitives import sdf_cone
        v = sdf_cone(pts([1, 0, 5]), [0, 0, 0], [0, 0, 10], 5.0)
        assert v[0] < 0

    def test_exterior_positive(self):
        from picogk_mp.shapek.primitives import sdf_cone
        v = sdf_cone(pts([0, 0, 20]), [0, 0, 0], [0, 0, 10], 5.0)
        assert v[0] > 0

    def test_below_apex_positive(self):
        from picogk_mp.shapek.primitives import sdf_cone
        v = sdf_cone(pts([0, 0, -5]), [0, 0, 0], [0, 0, 10], 5.0)
        assert v[0] == pytest.approx(5.0, abs=0.01)


class TestSdfTorus:
    def test_tube_surface_point(self):
        """Point at major_r + minor_r on XY-plane should be on surface."""
        from picogk_mp.shapek.primitives import sdf_torus
        # torus in XY plane (Z axis), major=20, minor=5
        # Point [25, 0, 0]: on outer equator of tube
        v = sdf_torus(pts([25, 0, 0]), [0, 0, 0], 20, 5)
        assert v[0] == pytest.approx(0.0, abs=1e-9)

    def test_inner_equator(self):
        """Inner equator: major_r - minor_r from centre."""
        from picogk_mp.shapek.primitives import sdf_torus
        v = sdf_torus(pts([15, 0, 0]), [0, 0, 0], 20, 5)
        assert v[0] == pytest.approx(0.0, abs=1e-9)

    def test_center_outside(self):
        from picogk_mp.shapek.primitives import sdf_torus
        v = sdf_torus(pts([0, 0, 0]), [0, 0, 0], 20, 5)
        assert v[0] > 0

    def test_inside_tube_negative(self):
        from picogk_mp.shapek.primitives import sdf_torus
        # Point at ring centre (major_r, 0, 0): inside tube
        v = sdf_torus(pts([20, 0, 0]), [0, 0, 0], 20, 5)
        assert v[0] == pytest.approx(-5.0, abs=1e-9)


class TestSdfRevolve:
    def test_revolve_disc(self):
        """Revolve a segment at r=10 into a thin disc -> torus-like."""
        from picogk_mp.shapek.primitives import sdf_revolve

        def disc_profile(pts2d):
            # Thin disc at r=10, half-thickness 2
            r, z = pts2d[:, 0], pts2d[:, 1]
            return np.maximum(np.abs(r - 10.0) - 0.5, np.abs(z) - 2.0)

        # Point at [10.5, 0, 0] should be on surface (r=10.5, z=0: disc at r=10 +/-0.5)
        test_pts = pts([10.5, 0, 0])
        v = sdf_revolve(test_pts, disc_profile)
        assert v[0] == pytest.approx(0.0, abs=0.01)


class TestSdfPipe:
    def test_constant_radius_midpoint(self):
        """Point on surface at midpoint of straight pipe."""
        from picogk_mp.shapek.modulation import LineModulation
        from picogk_mp.shapek.primitives import sdf_pipe

        spine = np.array([[0, 0, 0], [0, 0, 20]], dtype=float)
        rmod  = LineModulation.constant(5.0)
        # Point [5, 0, 10]: exactly on surface of the pipe
        v = sdf_pipe(pts([5, 0, 10]), spine, rmod)
        assert v[0] == pytest.approx(0.0, abs=1e-9)

    def test_tapered_pipe(self):
        """Tapered pipe: radius varies from 4 at t=0 to 8 at t=1."""
        from picogk_mp.shapek.modulation import LineModulation
        from picogk_mp.shapek.primitives import sdf_pipe

        spine = np.array([[0, 0, 0], [0, 0, 10]], dtype=float)
        rmod  = LineModulation.from_endpoints(4.0, 8.0)
        # At t=1 (z=10): radius=8, point [8,0,10] should be on surface
        v = sdf_pipe(pts([8, 0, 10]), spine, rmod)
        assert v[0] == pytest.approx(0.0, abs=0.01)

    def test_returns_inf_for_empty_spine(self):
        from picogk_mp.shapek.modulation import LineModulation
        from picogk_mp.shapek.primitives import sdf_pipe

        with pytest.raises(ValueError):
            sdf_pipe(pts([0, 0, 0]), np.array([[0, 0, 0]]), LineModulation.constant(5.0))


# ---------------------------------------------------------------------------
# BaseShape / CompoundShape (mesh_stl only -- no picogk.go)
# ---------------------------------------------------------------------------

class TestBaseShapeStl:
    def test_sphere_shape_mesh_stl(self, tmp_path):
        from picogk_mp.shapek.base_shape import SphereShape
        s   = SphereShape([0, 0, 0], 10.0)
        out = tmp_path / "sphere.stl"
        r   = s.mesh_stl(resolution_mm=2.0, out_stl=str(out))
        assert r["status"] == "ok"
        assert r["volume_mm3"] > 0
        assert out.exists()

    def test_cone_shape_mesh_stl(self, tmp_path):
        from picogk_mp.shapek.base_shape import ConeShape
        c   = ConeShape([0, 0, 0], [0, 0, 30], 10.0)
        out = tmp_path / "cone.stl"
        r   = c.mesh_stl(resolution_mm=2.0, out_stl=str(out))
        assert r["status"] == "ok"
        # pi * r^2 * h / 3 = pi * 100 * 30 / 3 ~ 3142 mm3 at 2mm resolution
        assert r["volume_mm3"] == pytest.approx(3142, rel=0.15)

    def test_torus_shape_mesh_stl(self, tmp_path):
        from picogk_mp.shapek.base_shape import TorusShape
        t   = TorusShape([0, 0, 0], 20.0, 5.0)
        out = tmp_path / "torus.stl"
        r   = t.mesh_stl(resolution_mm=2.0, out_stl=str(out))
        assert r["status"] == "ok"
        assert r["volume_mm3"] > 0

    def test_compound_shape_mesh_stl(self, tmp_path):
        from picogk_mp.shapek.base_shape import CompoundShape, SphereShape, ConeShape
        cs  = CompoundShape(
            SphereShape([0, 0, 0], 8),
            ConeShape([0, 0, 0], [0, 0, 25], 8),
        )
        out = tmp_path / "compound.stl"
        r   = cs.mesh_stl(resolution_mm=2.0, out_stl=str(out))
        assert r["status"] == "ok"
        assert r["volume_mm3"] > 0

    def test_pipe_shape_mesh_stl(self, tmp_path):
        from picogk_mp.shapek.base_shape import PipeShape
        from picogk_mp.shapek.modulation import LineModulation
        spine = np.array([[0, 0, 0], [30, 0, 0], [30, 30, 0]], dtype=float)
        ps    = PipeShape(spine, LineModulation.constant(4.0))
        out   = tmp_path / "pipe.stl"
        r     = ps.mesh_stl(resolution_mm=2.0, out_stl=str(out))
        assert r["status"] == "ok"
        assert r["volume_mm3"] > 0

    def test_build_compound_from_spec_cone(self, tmp_path):
        from picogk_mp.shapek.base_shape import build_compound_from_spec
        spec = [{"type": "cone", "apex": [0, 0, 0], "base": [0, 0, 30], "r_base": 10}]
        cs   = build_compound_from_spec(spec)
        out  = tmp_path / "spec_cone.stl"
        r    = cs.mesh_stl(resolution_mm=2.0, out_stl=str(out))
        assert r["status"] == "ok"
        assert r["volume_mm3"] > 0

    def test_build_compound_from_spec_pipe(self, tmp_path):
        from picogk_mp.shapek.base_shape import build_compound_from_spec
        spec = [{"type": "pipe", "spine": [[0,0,0],[0,0,40]], "radius": 5}]
        cs   = build_compound_from_spec(spec)
        out  = tmp_path / "spec_pipe.stl"
        r    = cs.mesh_stl(resolution_mm=2.0, out_stl=str(out))
        assert r["status"] == "ok"

    def test_build_compound_from_spec_pipe_radii(self, tmp_path):
        from picogk_mp.shapek.base_shape import build_compound_from_spec
        spec = [{"type": "pipe", "spine": [[0,0,0],[0,0,40]], "radii": [4, 8]}]
        cs   = build_compound_from_spec(spec)
        out  = tmp_path / "spec_pipe_radii.stl"
        r    = cs.mesh_stl(resolution_mm=2.0, out_stl=str(out))
        assert r["status"] == "ok"


# ---------------------------------------------------------------------------
# Measure.from_stl
# ---------------------------------------------------------------------------

class TestMeasureFromStl:
    def test_phase0_sphere_volume_positive(self):
        from picogk_mp.shapek.measure import Measure
        stl = FIXTURES / "phase0_sphere.stl"
        if not stl.exists():
            pytest.skip("phase0_sphere.stl fixture not found")
        m = Measure.from_stl(stl)
        assert m.volume_mm3 > 0
        assert m.surface_area_mm2 > 0

    def test_phase0_sphere_cog_finite(self):
        from picogk_mp.shapek.measure import Measure
        stl = FIXTURES / "phase0_sphere.stl"
        if not stl.exists():
            pytest.skip("phase0_sphere.stl fixture not found")
        m = Measure.from_stl(stl)
        assert np.all(np.isfinite(m.center_of_gravity_mm))

    def test_mass_with_density(self):
        from picogk_mp.shapek.measure import Measure
        stl = FIXTURES / "phase0_sphere.stl"
        if not stl.exists():
            pytest.skip("phase0_sphere.stl fixture not found")
        m = Measure.from_stl(stl, density_g_cm3=1.24, infill_pct=100.0)
        assert m.mass_g is not None
        assert m.mass_g > 0
        # mass = volume_mm3 * 1.24e-3 g/mm3
        expected = m.volume_mm3 * 1.24e-3
        assert m.mass_g == pytest.approx(expected, rel=0.01)

    def test_inertia_tensor_shape(self):
        from picogk_mp.shapek.measure import Measure
        stl = FIXTURES / "phase0_sphere.stl"
        if not stl.exists():
            pytest.skip("phase0_sphere.stl fixture not found")
        m = Measure.from_stl(stl, density_g_cm3=1.24, infill_pct=100.0)
        assert m.inertia_tensor_g_mm2.shape == (3, 3)

    def test_principal_axes_orthogonal(self):
        from picogk_mp.shapek.measure import Measure
        stl = FIXTURES / "phase0_sphere.stl"
        if not stl.exists():
            pytest.skip("phase0_sphere.stl fixture not found")
        m   = Measure.from_stl(stl, density_g_cm3=1.24, infill_pct=100.0)
        vals, vecs = Measure.principal_axes(m)
        # Eigenvectors should form an orthonormal set
        assert np.allclose(vecs @ vecs.T, np.eye(3), atol=1e-10)
        # Eigenvalues should be positive
        assert np.all(vals > 0)

    def test_holder_fixture_volume(self):
        from picogk_mp.shapek.measure import Measure
        stl = FIXTURES / "headphone_holder.stl"
        if not stl.exists():
            pytest.skip("headphone_holder.stl fixture not found")
        m = Measure.from_stl(stl)
        # Holder is a non-trivial solid, volume should be significant
        assert m.volume_mm3 > 1000


# ---------------------------------------------------------------------------
# _DISPATCH extension (shapek.__init__ must extend generators/shape._DISPATCH)
# ---------------------------------------------------------------------------

class TestDispatchExtension:
    def test_cone_added_to_dispatch(self):
        import picogk_mp.shapek  # noqa: F401 -- triggers _DISPATCH extension
        from picogk_mp.generators.shape import _DISPATCH
        assert "cone" in _DISPATCH

    def test_torus_added_to_dispatch(self):
        import picogk_mp.shapek  # noqa: F401
        from picogk_mp.generators.shape import _DISPATCH
        assert "torus" in _DISPATCH

    def test_generate_shape_with_cone(self, tmp_path):
        import picogk_mp.shapek  # noqa: F401
        from picogk_mp.generators.shape import generate_shape_stl
        out = tmp_path / "cone.stl"
        r = generate_shape_stl(
            [{"type": "cone", "apex": [0, 0, 0], "base": [0, 0, 30], "r_base": 10}],
            resolution_mm=2.0,
            out_stl=str(out),
        )
        assert r["status"] == "ok"
        assert r["volume_mm3"] > 0

    def test_generate_shape_with_torus(self, tmp_path):
        import picogk_mp.shapek  # noqa: F401
        from picogk_mp.generators.shape import generate_shape_stl
        out = tmp_path / "torus.stl"
        r = generate_shape_stl(
            [{"type": "torus", "center": [0, 0, 0], "major_r": 20, "minor_r": 5}],
            resolution_mm=2.0,
            out_stl=str(out),
        )
        assert r["status"] == "ok"
        assert r["volume_mm3"] > 0
