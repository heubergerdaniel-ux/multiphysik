"""Tests for the multiphysics engine (params, checks, engine)."""
import math
import pytest

from picogk_mp.physics import Param, SimEngine, TippingCheck, CantileverBendingCheck
from picogk_mp.physics.engine import PhysicsFailure


# ======================================================================
# Param tests
# ======================================================================

class TestParam:
    def test_unresolved_by_default(self):
        p = Param("mass", "Masse", unit="g")
        assert not p.is_resolved

    def test_set_resolves(self):
        p = Param("mass", "Masse", unit="g")
        p.set(350)
        assert p.is_resolved
        assert p.resolved_value == 350.0

    def test_default_used_when_no_value(self):
        p = Param("infill", "Infill", unit="%", default=20)
        assert not p.is_resolved          # no explicit value
        assert p.resolved_value == 20.0   # but default works

    def test_range_validation_rejects_low(self):
        p = Param("mass", "Masse", unit="g", lo=50)
        with pytest.raises(ValueError):
            p.set(10)

    def test_range_validation_rejects_high(self):
        p = Param("mass", "Masse", unit="g", hi=2000)
        with pytest.raises(ValueError):
            p.set(9999)

    def test_range_validation_accepts_boundary(self):
        p = Param("mass", "Masse", unit="g", lo=50, hi=1000)
        p.set(50)
        assert p.resolved_value == 50.0
        p.set(1000)
        assert p.resolved_value == 1000.0

    def test_resolved_value_raises_if_neither(self):
        p = Param("x", "X", unit="mm")
        with pytest.raises(ValueError):
            _ = p.resolved_value

    def test_reset_clears_value(self):
        p = Param("mass", "Masse", unit="g")
        p.set(400)
        p.reset()
        assert not p.is_resolved

    def test_prompt_text_contains_label_and_unit(self):
        p = Param("mass", "Lastmasse", unit="g", lo=50, hi=1000, default=400)
        txt = p.prompt_text()
        assert "Lastmasse" in txt
        assert "[g]" in txt
        assert "50" in txt
        assert "1000" in txt
        assert "400" in txt


# ======================================================================
# TippingCheck tests
# ======================================================================

# load offset (82 mm) inside base (95 mm) -> instant pass, no SF calculation
STABLE_CTX = {
    "load_mass_g":    350,
    "base_r_mm":      95,
    "load_offset_mm": 82,
    "volume_mm3":     300_000,
    "infill_pct":     100,
    "density_g_cm3":  1.24,
}

# load offset (82 mm) outside base (70 mm) but big enough to pass SF >= 1.5
STABLE_SF_CTX = {
    "load_mass_g":    350,
    "base_r_mm":      70,
    "load_offset_mm": 82,
    "volume_mm3":     500_000,
    "infill_pct":     100,
    "density_g_cm3":  1.24,
}

UNSTABLE_CTX = {
    **STABLE_CTX,
    "base_r_mm":   48,      # too small
    "infill_pct":  20,      # lightweight print
}

class TestTippingCheck:
    def test_passes_with_large_base(self):
        r = TippingCheck().evaluate(STABLE_CTX)
        assert r.passed
        assert r.sf >= 1.5

    def test_fails_with_small_base_and_low_infill(self):
        r = TippingCheck().evaluate(UNSTABLE_CTX)
        assert not r.passed
        assert r.sf < 1.5

    def test_no_tip_risk_when_load_inside_base(self):
        ctx = {**STABLE_CTX, "load_offset_mm": 40, "base_r_mm": 95}
        r = TippingCheck().evaluate(ctx)
        assert r.passed
        assert r.sf == 999.0

    def test_detail_contains_sf(self):
        # Use STABLE_SF_CTX where load extends beyond base -> SF is calculated
        r = TippingCheck().evaluate(STABLE_SF_CTX)
        assert "SF=" in r.detail

    def test_fail_detail_suggests_min_radius(self):
        r = TippingCheck().evaluate(UNSTABLE_CTX)
        assert "min base radius" in r.detail

    def test_missing_key_raises(self):
        ctx = {k: v for k, v in STABLE_CTX.items() if k != "base_r_mm"}
        with pytest.raises(KeyError):
            TippingCheck().evaluate(ctx)


# ======================================================================
# CantileverBendingCheck tests
# ======================================================================

BEND_CTX = {
    "load_mass_g":           400,
    "load_offset_mm":        82,
    "min_section_radius_mm": 5.0,   # 10 mm diameter min cross-section
    "yield_mpa":             60.0,  # PLA typical yield ~50-65 MPa
}

class TestCantileverBendingCheck:
    def test_passes_for_thick_section(self):
        ctx = {**BEND_CTX, "min_section_radius_mm": 8.0}
        r = CantileverBendingCheck().evaluate(ctx)
        assert r.passed, f"Expected pass, SF={r.sf:.2f}"

    def test_fails_for_thin_section(self):
        ctx = {**BEND_CTX, "min_section_radius_mm": 1.5}
        r = CantileverBendingCheck().evaluate(ctx)
        assert not r.passed

    def test_sf_scales_with_radius(self):
        r_thin  = CantileverBendingCheck().evaluate({**BEND_CTX, "min_section_radius_mm": 4.0})
        r_thick = CantileverBendingCheck().evaluate({**BEND_CTX, "min_section_radius_mm": 8.0})
        assert r_thick.sf > r_thin.sf

    def test_detail_contains_moment_and_stress(self):
        r = CantileverBendingCheck().evaluate(BEND_CTX)
        assert "N*mm" in r.detail
        assert "MPa" in r.detail


# ======================================================================
# SimEngine tests
# ======================================================================

def _make_engine(resolver: dict) -> SimEngine:
    """Helper: engine with all params for a disc-base cantilever scenario."""
    engine = SimEngine(resolver=resolver)
    engine.register(
        Param("load_mass_g",           "Lastmasse",       unit="g",      lo=50,   hi=2000),
        Param("infill_pct",            "Infill",          unit="%",      default=20, lo=5, hi=100),
        Param("density_g_cm3",         "Materialdichte",  unit="g/cm3",  default=1.24),
        Param("yield_mpa",             "Streckgrenze",    unit="MPa",    default=55.0),
        # geometry -- injected later
        Param("base_r_mm",             "Basisradius",     unit="mm"),
        Param("load_offset_mm",        "Lastversatz",     unit="mm"),
        Param("volume_mm3",            "Volumen",         unit="mm3"),
        Param("min_section_radius_mm", "Min-Querschnittsradius", unit="mm"),
    )
    engine.add_check(TippingCheck())
    engine.add_check(CantileverBendingCheck())
    return engine


class TestSimEngine:
    def test_passes_stable_design(self):
        engine = _make_engine({"load_mass_g": 350})
        engine.inject(
            base_r_mm=95, load_offset_mm=82,
            volume_mm3=400_000, min_section_radius_mm=7.0,
        )
        results = engine.run(raise_on_failure=False)
        assert all(r.passed for r in results), [str(r) for r in results]

    def test_raises_on_unstable_design(self):
        engine = _make_engine({"load_mass_g": 350})
        engine.inject(
            base_r_mm=48, load_offset_mm=82,     # original too-small base
            volume_mm3=172_355, min_section_radius_mm=7.0,
        )
        with pytest.raises(PhysicsFailure):
            engine.run(raise_on_failure=True)

    def test_resolver_dict_shorthand(self):
        # All params registered BEFORE inject so inject() finds them
        engine = SimEngine(resolver={"load_mass_g": 400})
        engine.register(
            Param("load_mass_g",    "Masse",      unit="g",     lo=50, hi=2000),
            Param("infill_pct",     "Infill",     unit="%",     default=20),
            Param("density_g_cm3",  "Dichte",     unit="g/cm3", default=1.24),
            Param("base_r_mm",      "Basis",      unit="mm"),
            Param("load_offset_mm", "Versatz",    unit="mm"),
            Param("volume_mm3",     "Volumen",    unit="mm3"),
        )
        engine.add_check(TippingCheck())
        engine.inject(base_r_mm=95, load_offset_mm=82, volume_mm3=400_000)
        results = engine.run(raise_on_failure=False)
        assert results[0].passed

    def test_inject_ignores_unknown_keys(self):
        engine = SimEngine(resolver={})
        engine.register(Param("base_r_mm", "Basis", unit="mm"))
        # extra keys that aren't registered -- should not raise
        engine.inject(base_r_mm=95, volume_mm3=300_000, nonexistent=42)
        assert engine._params["base_r_mm"].resolved_value == 95.0

    def test_summary_lists_params_and_checks(self):
        engine = _make_engine({"load_mass_g": 400})
        engine.inject(base_r_mm=95, load_offset_mm=82, volume_mm3=300_000, min_section_radius_mm=7.0)
        s = engine.summary()
        assert "SimEngine" in s
        assert "load_mass_g" in s
        assert "TippingStability" in s

    def test_no_checks_returns_empty_list(self):
        engine = SimEngine(resolver={})
        results = engine.run(raise_on_failure=False)
        assert results == []

    def test_raise_on_failure_false_returns_results(self):
        engine = _make_engine({"load_mass_g": 350})
        engine.inject(
            base_r_mm=48, load_offset_mm=82,
            volume_mm3=172_355, min_section_radius_mm=7.0,
        )
        results = engine.run(raise_on_failure=False)
        failed = [r for r in results if not r.passed]
        assert len(failed) >= 1
