"""Tests for the CFD sub-package (D2Q9 LBM flow + D2Q5 thermal).

All tests run without picogk and without a GPU.  The test geometry is
generated from trimesh primitives so no fixture STL is required.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sphere_stl(radius_mm: float, tmp_path: Path) -> Path:
    """Export a trimesh sphere to a temporary STL."""
    mesh = trimesh.primitives.Sphere(radius=radius_mm)
    p = tmp_path / f"sphere_r{int(radius_mm)}.stl"
    mesh.export(str(p))
    return p


def _box_stl(size_mm: float, tmp_path: Path) -> Path:
    mesh = trimesh.creation.box(extents=[size_mm] * 3)
    p = tmp_path / f"box_{int(size_mm)}.stl"
    mesh.export(str(p))
    return p


# ---------------------------------------------------------------------------
# lbm_core unit tests
# ---------------------------------------------------------------------------

class TestLbmCore:
    def test_feq_mass_conservation(self):
        """Sum of equilibrium distributions equals density."""
        from picogk_mp.cfd.lbm_core import _feq9
        rho = np.ones((5, 7))
        ux  = np.zeros((5, 7))
        uy  = np.zeros((5, 7))
        feq = _feq9(rho, ux, uy)
        assert feq.shape == (9, 5, 7)
        np.testing.assert_allclose(feq.sum(axis=0), rho, atol=1e-12)

    def test_feq_momentum(self):
        """Sum of c_i * f_eq_i equals rho*u."""
        from picogk_mp.cfd.lbm_core import _feq9, _CX9, _CY9
        rho = np.full((4, 6), 1.0)
        ux  = np.full((4, 6), 0.05)
        uy  = np.full((4, 6), 0.02)
        feq = _feq9(rho, ux, uy)
        mom_x = (_CX9[:, None, None] * feq).sum(axis=0)
        mom_y = (_CY9[:, None, None] * feq).sum(axis=0)
        np.testing.assert_allclose(mom_x, rho * ux, atol=1e-12)
        np.testing.assert_allclose(mom_y, rho * uy, atol=1e-12)

    def test_streaming_no_solid(self):
        """Streaming preserves total mass on a periodic domain."""
        from picogk_mp.cfd.lbm_core import _feq9, _stream9
        rho = np.ones((10, 12))
        ux  = np.full((10, 12), 0.05)
        uy  = np.zeros((10, 12))
        f = _feq9(rho, ux, uy)
        mass_before = f.sum()
        f2 = _stream9(f)
        np.testing.assert_allclose(f2.sum(), mass_before, rtol=1e-10)

    def test_drag_zero_for_no_solid(self):
        """Drag is zero when there is no solid obstacle."""
        from picogk_mp.cfd.lbm_core import _feq9, compute_drag_x
        rho = np.ones((8, 10))
        ux  = np.full((8, 10), 0.1)
        uy  = np.zeros((8, 10))
        fstar = _feq9(rho, ux, uy)
        mask  = np.zeros((8, 10), dtype=bool)
        assert compute_drag_x(mask, fstar) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Domain builder
# ---------------------------------------------------------------------------

class TestDomain:
    def test_domain_shape(self, tmp_path):
        """Padded mask is larger than raw voxelised geometry."""
        from picogk_mp.cfd.domain import build_domain
        stl = _box_stl(30.0, tmp_path)
        domain = build_domain(stl, resolution_mm=5.0, velocity_m_s=0.5)
        # Padded domain must be larger than a raw 6x6 box
        assert domain.Nx > 6
        assert domain.Ny > 6
        assert domain.mask.dtype == bool

    def test_char_length(self, tmp_path):
        """char_length_m is close to the box dimension in flow axis."""
        from picogk_mp.cfd.domain import build_domain
        stl = _box_stl(30.0, tmp_path)
        domain = build_domain(stl, resolution_mm=5.0, velocity_m_s=0.5)
        # 30 mm box at 5 mm resolution -> ~6 cells * 5e-3 m = 0.030 m
        assert 0.020 <= domain.char_length_m <= 0.050

    def test_reynolds_number(self, tmp_path):
        """Re = U * L / nu is computed correctly."""
        from picogk_mp.cfd.domain import build_domain
        stl = _box_stl(30.0, tmp_path)
        domain = build_domain(stl, resolution_mm=5.0, velocity_m_s=1.0,
                               nu_air_m2s=1e-5)
        # Re = 1.0 * ~0.03 / 1e-5 ≈ 3000  (rough -- depends on voxel count)
        assert domain.Re > 100


# ---------------------------------------------------------------------------
# CFD-A: run_flow integration test
# ---------------------------------------------------------------------------

class TestRunFlow:
    @pytest.mark.parametrize("direction", ["x", "y"])
    def test_flow_runs(self, tmp_path, direction):
        """run_flow completes without error for both x and y flow directions."""
        from picogk_mp.cfd import run_flow
        stl = _sphere_stl(15.0, tmp_path)
        result = run_flow(
            stl, velocity_m_s=0.5, flow_direction=direction,
            resolution_mm=5.0, max_steps=200,
        )
        assert result.Cd is not None
        assert result.Re > 0
        assert result.ux_lb.shape == result.domain.mask.shape

    def test_cd_physical_range(self, tmp_path):
        """Cd for a sphere cross-section should be in a physically plausible range."""
        from picogk_mp.cfd import run_flow
        stl = _sphere_stl(20.0, tmp_path)
        result = run_flow(
            stl, velocity_m_s=0.5, flow_direction="x",
            resolution_mm=4.0, max_steps=500,
        )
        # For 2D flow around a compact shape, Cd is typically 0.1..5.0
        assert 0.05 <= result.Cd <= 10.0, f"Cd={result.Cd:.3f} outside expected range"

    def test_velocity_field_non_zero(self, tmp_path):
        """Velocity in the fluid region must be non-zero after simulation."""
        from picogk_mp.cfd import run_flow
        stl = _box_stl(20.0, tmp_path)
        result = run_flow(
            stl, velocity_m_s=0.5, flow_direction="x",
            resolution_mm=5.0, max_steps=300,
        )
        fluid_ux = result.ux_lb[~result.domain.mask]
        assert float(np.abs(fluid_ux).max()) > 1e-6


# ---------------------------------------------------------------------------
# CFD-B: run_thermal integration test
# ---------------------------------------------------------------------------

class TestRunThermal:
    def test_thermal_runs(self, tmp_path):
        """run_thermal completes and returns a temperature field."""
        from picogk_mp.cfd import run_flow, run_thermal
        stl = _box_stl(20.0, tmp_path)
        flow = run_flow(
            stl, velocity_m_s=0.5, flow_direction="x",
            resolution_mm=5.0, max_steps=300,
        )
        thermal = run_thermal(
            flow, heat_flux_W_m2=500.0, T_inlet_C=20.0, max_steps=300,
        )
        assert thermal.T_field.shape == flow.domain.mask.shape
        assert thermal.T_max >= 20.0

    def test_h_conv_positive(self, tmp_path):
        """h_conv must be positive (heat is removed by convection)."""
        from picogk_mp.cfd import run_flow, run_thermal
        stl = _sphere_stl(15.0, tmp_path)
        flow = run_flow(
            stl, velocity_m_s=1.0, flow_direction="x",
            resolution_mm=5.0, max_steps=300,
        )
        thermal = run_thermal(
            flow, heat_flux_W_m2=1000.0, T_inlet_C=20.0, max_steps=300,
        )
        assert thermal.h_conv > 0.0, f"h_conv={thermal.h_conv:.3f} must be positive"

    def test_t_max_above_inlet(self, tmp_path):
        """Surface temperature must exceed inlet temperature."""
        from picogk_mp.cfd import run_flow, run_thermal
        stl = _box_stl(20.0, tmp_path)
        flow = run_flow(
            stl, velocity_m_s=0.5, flow_direction="x",
            resolution_mm=5.0, max_steps=200,
        )
        thermal = run_thermal(
            flow, heat_flux_W_m2=2000.0, T_inlet_C=20.0, max_steps=300,
        )
        assert thermal.T_max > 20.0


# ---------------------------------------------------------------------------
# DragCheck
# ---------------------------------------------------------------------------

class TestDragCheck:
    def test_below_threshold_passes(self):
        from picogk_mp.cfd.checks import DragCheck
        r = DragCheck(Cd_warn=2.0).evaluate({"Cd": 1.2, "Re": 500})
        assert r.passed

    def test_above_threshold_fails(self):
        from picogk_mp.cfd.checks import DragCheck
        r = DragCheck(Cd_warn=2.0).evaluate({"Cd": 3.5, "Re": 500})
        assert not r.passed

    def test_sf_ratio(self):
        from picogk_mp.cfd.checks import DragCheck
        r = DragCheck(Cd_warn=2.0).evaluate({"Cd": 1.0, "Re": 100})
        assert r.sf == pytest.approx(2.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Postprocess (PNG generation)
# ---------------------------------------------------------------------------

class TestPostprocess:
    def test_velocity_png_created(self, tmp_path):
        from picogk_mp.cfd import run_flow
        from picogk_mp.cfd.postprocess import save_velocity_png
        stl = _box_stl(20.0, tmp_path)
        result = run_flow(
            stl, velocity_m_s=0.5, flow_direction="x",
            resolution_mm=5.0, max_steps=100,
        )
        out = save_velocity_png(result, tmp_path / "velocity.png")
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_temperature_png_created(self, tmp_path):
        from picogk_mp.cfd import run_flow, run_thermal
        from picogk_mp.cfd.postprocess import save_temperature_png
        stl = _box_stl(20.0, tmp_path)
        flow = run_flow(
            stl, velocity_m_s=0.5, flow_direction="x",
            resolution_mm=5.0, max_steps=100,
        )
        thermal = run_thermal(flow, max_steps=100)
        out = save_temperature_png(thermal, tmp_path / "temperature.png")
        assert out.exists()
        assert out.stat().st_size > 1000
