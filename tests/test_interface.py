"""Tests fuer InterfaceFeature, Cut-Primitives, neue Requirements und Mapper-Erweiterungen."""
import math
import pytest

from picogk_mp.physics.interface import (
    InterfaceFeature,
    InterfaceType,
    interface_to_cut_primitives,
    press_fit_params,
    screw_bearing_params,
)
from picogk_mp.physics.brief import (
    Constraint, ConstraintType, FailureCriteria,
    LoadCase, LoadType, Material, MaterialPreset, PhysicsBrief,
)
from picogk_mp.physics.requirement import (
    PressFitRetentionRequirement,
    ScrewBearingRequirement,
)
from picogk_mp.physics.brief_mapper import (
    brief_to_interface_primitives,
    brief_to_requirements,
)


# ======================================================================
# Hilfsfunktionen
# ======================================================================

def _screw_feature(axis="x", n_count=1, with_counterbore=False) -> InterfaceFeature:
    cb_d   = 11.0 if with_counterbore else None
    cb_dep = 6.5  if with_counterbore else None
    return InterfaceFeature(
        feature_type=InterfaceType.SCREW_THROUGH,
        position=[10.0, 0.0, 0.0],
        diameter_mm=6.0,
        depth_mm=22.0,
        axis=axis,
        clearance_mm=0.3,
        n_count=n_count,
        counterbore_d_mm=cb_d,
        counterbore_depth_mm=cb_dep,
        thread_spec="M6",
    )


def _press_fit_feature() -> InterfaceFeature:
    return InterfaceFeature(
        feature_type=InterfaceType.PRESS_FIT,
        position=[0.0, 0.0, 50.0],
        diameter_mm=20.0,
        depth_mm=30.0,
        axis="z",
        interference_mm=0.05,
        hub_outer_d_mm=40.0,
        mu_friction=0.30,
    )


def _bracket_brief_with_ifaces(interfaces=None) -> PhysicsBrief:
    return PhysicsBrief(
        source_prompt="Wandkonsole mit Schrauben",
        material=Material(preset=MaterialPreset.PLA, infill_pct=80),
        load_cases=[
            LoadCase(LoadType.FORCE, 78.48,
                     direction=[0, 0, -1],
                     application_point=[450, 0, 180]),
        ],
        constraints=[
            Constraint(ConstraintType.FIXED_FACE, face="x0"),
        ],
        interfaces=interfaces or [],
    )


# ======================================================================
# InterfaceFeature -- Grundeigenschaften
# ======================================================================

class TestInterfaceFeature:

    def test_screw_through_defaults(self):
        f = _screw_feature()
        assert f.feature_type == InterfaceType.SCREW_THROUGH
        assert f.diameter_mm == 6.0
        assert f.depth_mm == 22.0
        assert f.axis == "x"
        assert f.clearance_mm == 0.3
        assert f.interference_mm == 0.0

    def test_press_fit_defaults(self):
        f = _press_fit_feature()
        assert f.feature_type == InterfaceType.PRESS_FIT
        assert f.interference_mm == 0.05
        assert f.hub_outer_d_mm == 40.0

    def test_to_dict_roundtrip(self):
        f = _screw_feature(with_counterbore=True)
        d = f.to_dict()
        f2 = InterfaceFeature.from_dict(d)
        assert f2.feature_type == f.feature_type
        assert f2.diameter_mm == f.diameter_mm
        assert f2.counterbore_d_mm == f.counterbore_d_mm
        assert f2.thread_spec == "M6"

    def test_from_dict_optional_none(self):
        f = _screw_feature()
        d = f.to_dict()
        assert d["counterbore_d_mm"] is None
        f2 = InterfaceFeature.from_dict(d)
        assert f2.counterbore_d_mm is None

    def test_enum_value_str(self):
        f = _screw_feature()
        d = f.to_dict()
        assert d["feature_type"] == "screw_through"


# ======================================================================
# interface_to_cut_primitives
# ======================================================================

class TestInterfaceToCutPrimitives:

    def test_screw_through_x_produces_cylinder_x(self):
        f = _screw_feature(axis="x")
        cuts = interface_to_cut_primitives(f)
        assert len(cuts) == 1
        c = cuts[0]
        assert c["type"] == "cylinder_x"
        assert c["mode"] == "cut"
        assert c["radius"] == pytest.approx(6.3 / 2, rel=1e-3)   # d + clearance / 2

    def test_screw_through_z_produces_cylinder(self):
        f = InterfaceFeature(
            feature_type=InterfaceType.SCREW_THROUGH,
            position=[0.0, 0.0, 50.0],
            diameter_mm=4.0,
            depth_mm=15.0,
            axis="z",
        )
        cuts = interface_to_cut_primitives(f)
        assert cuts[0]["type"] == "cylinder"
        assert cuts[0]["mode"] == "cut"

    def test_screw_through_y_produces_cylinder_y(self):
        f = InterfaceFeature(
            feature_type=InterfaceType.SCREW_THROUGH,
            position=[0.0, 20.0, 0.0],
            diameter_mm=5.0,
            depth_mm=18.0,
            axis="y",
        )
        cuts = interface_to_cut_primitives(f)
        assert cuts[0]["type"] == "cylinder_y"
        assert cuts[0]["mode"] == "cut"

    def test_counterbore_adds_second_cut(self):
        f = _screw_feature(with_counterbore=True)
        cuts = interface_to_cut_primitives(f)
        assert len(cuts) == 2
        # Second cut = counterbore, larger radius
        main_r   = cuts[0]["radius"]
        bore_r   = cuts[1]["radius"]
        assert bore_r > main_r

    def test_press_fit_bore_diameter_minus_interference(self):
        f = _press_fit_feature()
        cuts = interface_to_cut_primitives(f)
        assert len(cuts) == 1
        # Bore = d - interference = 20 - 0.05 = 19.95 => r = 9.975
        assert cuts[0]["radius"] == pytest.approx(9.975, rel=1e-3)

    def test_dowel_pin_uses_clearance(self):
        f = InterfaceFeature(
            feature_type=InterfaceType.DOWEL_PIN,
            position=[0.0, 0.0, 0.0],
            diameter_mm=8.0,
            depth_mm=20.0,
            axis="z",
            clearance_mm=0.1,
        )
        cuts = interface_to_cut_primitives(f)
        assert cuts[0]["radius"] == pytest.approx((8.0 + 0.1) / 2, rel=1e-3)


# ======================================================================
# screw_bearing_params / press_fit_params
# ======================================================================

class TestParamExtraction:

    def test_screw_bearing_params_sum_n_count(self):
        feats = [
            _screw_feature(n_count=2),
            _screw_feature(n_count=2),
        ]
        p = screw_bearing_params(feats)
        assert p["n_screws"] == 4
        assert p["screw_d_mm"] == 6.0
        assert p["plate_t_mm"] == 22.0

    def test_screw_bearing_params_empty(self):
        p = screw_bearing_params([])
        assert p == {}

    def test_screw_bearing_params_ignores_press_fit(self):
        feats = [_press_fit_feature()]
        p = screw_bearing_params(feats)
        assert p == {}

    def test_press_fit_params_basic(self):
        f = _press_fit_feature()
        p = press_fit_params(f)
        assert p["interference_mm"] == 0.05
        assert p["shaft_d_mm"] == 20.0
        assert p["hub_outer_d_mm"] == 40.0
        assert p["engagement_l_mm"] == 30.0
        assert p["mu_friction"] == 0.30

    def test_press_fit_params_wrong_type_raises(self):
        with pytest.raises(ValueError):
            press_fit_params(_screw_feature())


# ======================================================================
# ScrewBearingRequirement
# ======================================================================

class TestScrewBearingRequirement:

    def test_derive_returns_plate_t_min(self):
        brief = _bracket_brief_with_ifaces([_screw_feature(n_count=4)])
        req = ScrewBearingRequirement()
        d = req.derive(brief)
        assert "plate_t_min_mm" in d
        assert d["plate_t_min_mm"] > 0.0

    def test_derive_plate_increases_with_load(self):
        brief_light = _bracket_brief_with_ifaces([_screw_feature(n_count=4)])
        brief_light.load_cases[0] = LoadCase(
            LoadType.FORCE, 10.0, direction=[0, 0, -1], application_point=[100, 0, 0])
        brief_heavy = _bracket_brief_with_ifaces([_screw_feature(n_count=4)])
        brief_heavy.load_cases[0] = LoadCase(
            LoadType.FORCE, 200.0, direction=[0, 0, -1], application_point=[100, 0, 0])

        d_light = ScrewBearingRequirement().derive(brief_light)
        d_heavy = ScrewBearingRequirement().derive(brief_heavy)
        assert d_heavy["plate_t_min_mm"] > d_light["plate_t_min_mm"]

    def test_verify_ok_thick_plate(self):
        req = ScrewBearingRequirement()
        ctx = {
            "load_mass_g": 8000,
            "n_screws":    4,
            "screw_d_mm":  6.0,
            "plate_t_mm":  22.0,   # thick plate => low bearing stress
            "yield_mpa":   55.0,
        }
        result = req.verify(ctx)
        assert result.passed
        assert result.sf >= req.sf_required

    def test_verify_fail_thin_plate(self):
        req = ScrewBearingRequirement()
        ctx = {
            "load_mass_g": 50_000,  # very heavy
            "n_screws":    1,
            "screw_d_mm":  3.0,
            "plate_t_mm":  1.0,     # very thin => high bearing stress
            "yield_mpa":   55.0,
        }
        result = req.verify(ctx)
        assert not result.passed


# ======================================================================
# PressFitRetentionRequirement
# ======================================================================

class TestPressFitRetentionRequirement:

    def test_derive_returns_interference_and_engagement(self):
        f = _press_fit_feature()
        brief = _bracket_brief_with_ifaces([f])
        req = PressFitRetentionRequirement()
        d = req.derive(brief)
        assert "interference_min_mm" in d
        assert "engagement_l_min_mm" in d
        assert d["interference_min_mm"] > 0.0

    def test_verify_ok_with_generous_interference(self):
        req = PressFitRetentionRequirement()
        ctx = {
            "E_mpa":            3500.0,
            "interference_mm":  0.10,
            "shaft_d_mm":       20.0,
            "hub_outer_d_mm":   40.0,
            "engagement_l_mm":  30.0,
            "axial_force_N":    200.0,
            "mu_friction":      0.30,
        }
        result = req.verify(ctx)
        assert result.passed

    def test_verify_fail_no_interference(self):
        req = PressFitRetentionRequirement()
        ctx = {
            "E_mpa":            3500.0,
            "interference_mm":  0.0,    # no interference => no retention
            "shaft_d_mm":       20.0,
            "hub_outer_d_mm":   40.0,
            "engagement_l_mm":  30.0,
            "axial_force_N":    500.0,
            "mu_friction":      0.30,
        }
        result = req.verify(ctx)
        assert not result.passed


# ======================================================================
# Mapper-Erweiterungen
# ======================================================================

class TestMapper:

    def test_brief_to_interface_primitives_empty(self):
        brief = _bracket_brief_with_ifaces([])
        cuts = brief_to_interface_primitives(brief)
        assert cuts == []

    def test_brief_to_interface_primitives_four_screws(self):
        ifaces = [
            InterfaceFeature(InterfaceType.SCREW_THROUGH, [10, +55, 240], 6.0, 22, axis="x"),
            InterfaceFeature(InterfaceType.SCREW_THROUGH, [10, -55, 240], 6.0, 22, axis="x"),
            InterfaceFeature(InterfaceType.SCREW_THROUGH, [10, +55,  40], 6.0, 22, axis="x"),
            InterfaceFeature(InterfaceType.SCREW_THROUGH, [10, -55,  40], 6.0, 22, axis="x"),
        ]
        brief = _bracket_brief_with_ifaces(ifaces)
        cuts = brief_to_interface_primitives(brief)
        assert len(cuts) == 4
        assert all(c["mode"] == "cut" for c in cuts)

    def test_brief_to_requirements_adds_screw_bearing(self):
        brief = _bracket_brief_with_ifaces([_screw_feature()])
        reqs = brief_to_requirements(brief)
        names = [type(r).__name__ for r in reqs]
        assert "ScrewBearingRequirement" in names

    def test_brief_to_requirements_adds_press_fit(self):
        brief = _bracket_brief_with_ifaces([_press_fit_feature()])
        reqs = brief_to_requirements(brief)
        names = [type(r).__name__ for r in reqs]
        assert "PressFitRetentionRequirement" in names

    def test_brief_to_requirements_no_interfaces_no_new_reqs(self):
        brief = _bracket_brief_with_ifaces([])
        reqs = brief_to_requirements(brief)
        names = [type(r).__name__ for r in reqs]
        assert "ScrewBearingRequirement" not in names
        assert "PressFitRetentionRequirement" not in names
