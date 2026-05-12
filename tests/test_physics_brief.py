"""Tests fuer PhysicsBrief, Mapper und Requirements."""
import math
import pytest

from picogk_mp.physics.brief import (
    ComponentType, ConstraintType, DesignIntent, FailureCriteria,
    GeometryLanguage, LoadCase, LoadCombination, LoadType, Material,
    MaterialPreset, PhysicsBrief, Constraint,
)
from picogk_mp.physics.brief_mapper import (
    brief_to_requirements, brief_to_sim_engine, brief_to_topopt_kwargs,
    suggest_geometry,
)
from picogk_mp.physics.requirement import (
    BendingRequirement, BucklingRequirement, DragRequirement,
    TensionRequirement, TippingRequirement, TorsionRequirement,
)


# ======================================================================
# Fixtures
# ======================================================================

def _bracket_brief(keywords=None) -> PhysicsBrief:
    """Wandkonsole: FORCE 49N, FIXED_FACE x0, PLA."""
    return PhysicsBrief(
        source_prompt="Wandkonsole 5kg PLA",
        material=Material(preset=MaterialPreset.PLA, infill_pct=20),
        load_cases=[
            LoadCase(LoadType.FORCE, 49.05, direction=[0, 0, -1],
                     application_point=[120, 0, 0])
        ],
        constraints=[Constraint(ConstraintType.FIXED_FACE, face="x0")],
        intent=DesignIntent(
            component_type=ComponentType.BRACKET,
            keywords=keywords or [],
        ),
    )


def _stand_brief() -> PhysicsBrief:
    """Kopfhoeretaender: FORCE 3.924N, FIXED_DISC 48mm, PLA."""
    return PhysicsBrief(
        source_prompt="Kopfhoeretaender 400g PLA",
        material=Material(preset=MaterialPreset.PLA, infill_pct=20),
        load_cases=[
            LoadCase(LoadType.FORCE, 3.924, direction=[0, 0, -1],
                     application_point=[-82, 0, 244])
        ],
        constraints=[Constraint(ConstraintType.FIXED_DISC, disc_radius_mm=48.0)],
        intent=DesignIntent(component_type=ComponentType.STAND),
    )


# ======================================================================
# Material tests
# ======================================================================

class TestMaterial:
    def test_pla_defaults(self):
        m = Material()
        assert m.resolved("E_mpa")    == pytest.approx(3500)
        assert m.resolved("nu")       == pytest.approx(0.36)
        assert m.resolved("yield_mpa") == pytest.approx(55)

    def test_petg_defaults(self):
        m = Material(preset=MaterialPreset.PETG)
        assert m.resolved("E_mpa") == pytest.approx(2100)
        assert m.resolved("yield_mpa") == pytest.approx(50)

    def test_tpu_defaults(self):
        m = Material(preset=MaterialPreset.TPU)
        assert m.resolved("E_mpa") == pytest.approx(50)

    def test_override_wins(self):
        m = Material(preset=MaterialPreset.PLA, yield_mpa=80.0)
        assert m.resolved("yield_mpa") == pytest.approx(80.0)

    def test_effective_density_20pct(self):
        m = Material(preset=MaterialPreset.PLA, infill_pct=20)
        assert m.effective_density_g_cm3() == pytest.approx(1.24 * 0.20, rel=1e-3)

    def test_effective_density_100pct(self):
        m = Material(preset=MaterialPreset.PLA, infill_pct=100)
        assert m.effective_density_g_cm3() == pytest.approx(1.24, rel=1e-3)

    def test_roundtrip(self):
        m = Material(preset=MaterialPreset.PETG, infill_pct=50, yield_mpa=60.0)
        m2 = Material.from_dict(m.to_dict())
        assert m2.preset == MaterialPreset.PETG
        assert m2.infill_pct == pytest.approx(50)
        assert m2.yield_mpa == pytest.approx(60.0)

    def test_unknown_key_raises(self):
        m = Material()
        with pytest.raises(KeyError):
            m.resolved("does_not_exist")


# ======================================================================
# LoadCase tests
# ======================================================================

class TestLoadCase:
    def test_design_magnitude(self):
        lc = LoadCase(LoadType.FORCE, 10.0, sf_static=2.0)
        assert lc.design_magnitude == pytest.approx(20.0)

    def test_force_vector_normalized(self):
        lc = LoadCase(LoadType.FORCE, 10.0, direction=[1, 0, 0], sf_static=2.0)
        fv = lc.force_vector_N
        assert abs(fv[0]) == pytest.approx(20.0)
        assert fv[1] == pytest.approx(0.0)
        assert fv[2] == pytest.approx(0.0)

    def test_force_vector_default_down(self):
        lc = LoadCase(LoadType.GRAVITY, 9.81)
        fv = lc.force_vector_N
        assert fv[2] < 0

    def test_force_vector_diagonal(self):
        lc = LoadCase(LoadType.FORCE, 1.0, direction=[1, 1, 0], sf_static=1.0)
        fv = lc.force_vector_N
        mag = math.sqrt(fv[0]**2 + fv[1]**2 + fv[2]**2)
        assert mag == pytest.approx(1.0, rel=1e-5)

    def test_roundtrip(self):
        lc = LoadCase(LoadType.TORSION, 5000.0, description="Schraube")
        lc2 = LoadCase.from_dict(lc.to_dict())
        assert lc2.load_type == LoadType.TORSION
        assert lc2.magnitude == pytest.approx(5000.0)
        assert lc2.description == "Schraube"


# ======================================================================
# Constraint tests
# ======================================================================

class TestConstraint:
    def test_fixed_face_roundtrip(self):
        c = Constraint(ConstraintType.FIXED_FACE, face="z0")
        c2 = Constraint.from_dict(c.to_dict())
        assert c2.constraint_type == ConstraintType.FIXED_FACE
        assert c2.face == "z0"

    def test_fixed_disc_roundtrip(self):
        c = Constraint(ConstraintType.FIXED_DISC, disc_radius_mm=48.0)
        c2 = Constraint.from_dict(c.to_dict())
        assert c2.disc_radius_mm == pytest.approx(48.0)


# ======================================================================
# FailureCriteria tests
# ======================================================================

class TestFailureCriteria:
    def test_defaults(self):
        f = FailureCriteria()
        assert f.sf_bending == pytest.approx(3.0)
        assert f.sf_torsion == pytest.approx(3.0)
        assert f.sf_buckling == pytest.approx(3.0)
        assert f.sf_tension == pytest.approx(2.0)
        assert f.min_wall_thickness_mm == pytest.approx(1.2)

    def test_roundtrip(self):
        f = FailureCriteria(sf_bending=4.0, max_Cd=1.5)
        f2 = FailureCriteria.from_dict(f.to_dict())
        assert f2.sf_bending == pytest.approx(4.0)
        assert f2.max_Cd == pytest.approx(1.5)


# ======================================================================
# PhysicsBrief validation tests
# ======================================================================

class TestPhysicsBriefValidation:
    def test_valid_bracket(self):
        b = _bracket_brief()
        assert b.is_valid(), b.validate()

    def test_valid_stand(self):
        b = _stand_brief()
        assert b.is_valid(), b.validate()

    def test_empty_load_cases_invalid(self):
        b = _bracket_brief()
        b.load_cases = []
        errors = b.validate()
        assert any("load_cases" in e for e in errors)

    def test_zero_magnitude_invalid(self):
        b = _bracket_brief()
        b.load_cases[0].magnitude = 0.0
        errors = b.validate()
        assert any("magnitude" in e for e in errors)

    def test_force_without_direction_invalid(self):
        b = _bracket_brief()
        b.load_cases[0].direction = None
        errors = b.validate()
        assert any("direction" in e for e in errors)

    def test_empty_constraints_invalid(self):
        b = _bracket_brief()
        b.constraints = []
        errors = b.validate()
        assert any("constraints" in e for e in errors)

    def test_fixed_disc_without_radius_invalid(self):
        b = _stand_brief()
        b.constraints[0].disc_radius_mm = None
        errors = b.validate()
        assert any("disc_radius_mm" in e for e in errors)

    def test_fixed_face_without_face_invalid(self):
        b = _bracket_brief()
        b.constraints[0].face = None
        errors = b.validate()
        assert any("face" in e for e in errors)

    def test_low_sf_tipping_invalid(self):
        b = _bracket_brief()
        b.failure.sf_tipping = 0.5
        errors = b.validate()
        assert any("sf_tipping" in e for e in errors)

    def test_brief_id_auto_generated(self):
        b = _bracket_brief()
        assert len(b.brief_id) == 8

    def test_two_briefs_have_different_ids(self):
        b1 = _bracket_brief()
        b2 = _bracket_brief()
        assert b1.brief_id != b2.brief_id


# ======================================================================
# PhysicsBrief JSON round-trip
# ======================================================================

class TestPhysicsBriefRoundTrip:
    def test_to_dict_contains_required_keys(self):
        b = _bracket_brief()
        d = b.to_dict()
        for key in ("brief_id", "source_prompt", "material", "load_cases", "constraints", "failure", "intent"):
            assert key in d

    def test_roundtrip_preserves_brief_id(self):
        b = _bracket_brief()
        b2 = PhysicsBrief.from_dict(b.to_dict())
        assert b2.brief_id == b.brief_id

    def test_roundtrip_preserves_load_case(self):
        b = _bracket_brief()
        b2 = PhysicsBrief.from_dict(b.to_dict())
        assert b2.load_cases[0].magnitude == pytest.approx(49.05)
        assert b2.load_cases[0].load_type == LoadType.FORCE

    def test_roundtrip_preserves_constraint(self):
        b = _stand_brief()
        b2 = PhysicsBrief.from_dict(b.to_dict())
        assert b2.constraints[0].constraint_type == ConstraintType.FIXED_DISC
        assert b2.constraints[0].disc_radius_mm == pytest.approx(48.0)

    def test_roundtrip_preserves_material_preset(self):
        b = _bracket_brief()
        b2 = PhysicsBrief.from_dict(b.to_dict())
        assert b2.material.preset == MaterialPreset.PLA

    def test_load_combination_roundtrip(self):
        b = _bracket_brief()
        b.load_combination = LoadCombination.OR
        b2 = PhysicsBrief.from_dict(b.to_dict())
        assert b2.load_combination == LoadCombination.OR


# ======================================================================
# brief_to_requirements tests
# ======================================================================

class TestBriefToRequirements:
    def test_bracket_has_bending(self):
        reqs = brief_to_requirements(_bracket_brief())
        names = [type(r).__name__ for r in reqs]
        assert "BendingRequirement" in names

    def test_bracket_has_no_tipping(self):
        reqs = brief_to_requirements(_bracket_brief())
        assert not any(isinstance(r, TippingRequirement) for r in reqs)

    def test_stand_has_tipping(self):
        reqs = brief_to_requirements(_stand_brief())
        assert any(isinstance(r, TippingRequirement) for r in reqs)

    def test_stand_has_bending(self):
        reqs = brief_to_requirements(_stand_brief())
        assert any(isinstance(r, BendingRequirement) for r in reqs)

    def test_torsion_load_adds_torsion_req(self):
        b = _bracket_brief()
        b.load_cases.append(
            LoadCase(LoadType.TORSION, 2000.0, direction=None)
        )
        reqs = brief_to_requirements(b)
        assert any(isinstance(r, TorsionRequirement) for r in reqs)

    def test_flow_load_adds_drag_req(self):
        b = _bracket_brief()
        b.load_cases.append(LoadCase(LoadType.FLOW, 1.0))
        reqs = brief_to_requirements(b)
        assert any(isinstance(r, DragRequirement) for r in reqs)

    def test_sf_bending_propagated(self):
        b = _bracket_brief()
        b.failure.sf_bending = 4.0
        reqs = brief_to_requirements(b)
        bending = next(r for r in reqs if isinstance(r, BendingRequirement))
        assert bending.sf_required == pytest.approx(4.0)


# ======================================================================
# derive() tests  (Physics -> Geometrie)
# ======================================================================

class TestDerive:
    def test_bending_derive_returns_section_r_min(self):
        b = _bracket_brief()
        req = BendingRequirement()
        result = req.derive(b)
        assert "section_r_min_mm" in result
        assert result["section_r_min_mm"] > 0

    def test_bending_derive_scales_with_force(self):
        b_light = _bracket_brief()
        b_light.load_cases[0].magnitude = 10.0
        b_heavy = _bracket_brief()
        b_heavy.load_cases[0].magnitude = 100.0

        r_light = BendingRequirement().derive(b_light)["section_r_min_mm"]
        r_heavy = BendingRequirement().derive(b_heavy)["section_r_min_mm"]
        assert r_heavy > r_light

    def test_tipping_derive_returns_base_r_min(self):
        b = _stand_brief()
        req = TippingRequirement()
        result = req.derive(b)
        assert "base_r_min_mm" in result
        assert result["base_r_min_mm"] > 0

    def test_tension_derive_returns_area_and_radius(self):
        b = _bracket_brief()
        b.load_cases[0].application_point = None   # kein Hebelarm
        req = TensionRequirement()
        result = req.derive(b)
        assert "section_r_min_mm" in result
        assert "section_area_min_mm2" in result


# ======================================================================
# brief_to_topopt_kwargs tests
# ======================================================================

class TestTopoptKwargs:
    def test_default_vol_frac(self):
        kw = brief_to_topopt_kwargs(_bracket_brief())
        assert kw["vol_frac"] == pytest.approx(0.65)

    def test_leichtgewichtig_reduces_vol_frac(self):
        b = _bracket_brief(keywords=["leichtgewichtig"])
        kw = brief_to_topopt_kwargs(b)
        assert kw["vol_frac"] == pytest.approx(0.40)

    def test_steifigkeit_increases_vol_frac(self):
        b = _bracket_brief(keywords=["maximale Steifigkeit"])
        kw = brief_to_topopt_kwargs(b)
        assert kw["vol_frac"] == pytest.approx(0.85)

    def test_material_pla_e(self):
        kw = brief_to_topopt_kwargs(_bracket_brief())
        assert kw["E0"] == pytest.approx(3500)
        assert kw["nu"] == pytest.approx(0.36)


# ======================================================================
# suggest_geometry tests
# ======================================================================

class TestSuggestGeometry:
    def test_force_suggests_kragarm(self):
        geo = suggest_geometry(_bracket_brief())
        assert "Kragarm" in geo["geometry_class"] or "CapsuleShape" in geo["geometry_class"]

    def test_tpms_hint_on_lightweight(self):
        b = _bracket_brief(keywords=["leichtgewichtig"])
        geo = suggest_geometry(b)
        assert "Gyroid" in geo["geometry_class"] or "TPMS" in geo["geometry_class"]

    def test_moment_load_suggests_i_profile(self):
        b = _bracket_brief()
        b.load_cases[0] = LoadCase(LoadType.MOMENT, 5000.0)
        geo = suggest_geometry(b)
        assert "I-Profil" in geo["cross_section"] or "Kastenprofil" in geo["cross_section"]

    def test_pressure_suggests_gewoelbe(self):
        b = _bracket_brief()
        b.load_cases[0] = LoadCase(LoadType.PRESSURE, 0.1)
        geo = suggest_geometry(b)
        assert "Gewoelb" in geo["geometry_class"] or "Kuppel" in geo["geometry_class"] or "RevolveShape" in geo["geometry_class"]

    def test_min_wall_from_failure_criteria(self):
        geo = suggest_geometry(_bracket_brief())
        assert geo["min_wall_mm"] == pytest.approx(1.2)
