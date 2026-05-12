"""Multiphysics simulation engine for PicoGK pipeline.

Public API:
    Param       -- declares a physical parameter (known or unknown)
    SimEngine   -- orchestrates parameter resolution + check execution
    CheckResult -- result of a single physics check (= RequirementResult)

Built-in requirements (import from picogk_mp.physics.requirement):
    PhysicsRequirement, RequirementResult
    BendingRequirement, TippingRequirement, TorsionRequirement
    BucklingRequirement, TensionRequirement, DragRequirement

Backward-compat aliases:
    TippingCheck, SectionBendingCheck, StemBendingCheck, BaseCheck, CheckResult

Physics Brief (Physics-First workflow):
    PhysicsBrief, Material, LoadCase, Constraint, FailureCriteria, DesignIntent
    MaterialPreset, LoadType, LoadCombination, ConstraintType, ComponentType,
    GeometryLanguage

Brief Mapper:
    brief_to_requirements, brief_to_sim_engine, brief_to_boundary_conditions
    brief_to_topopt_kwargs, suggest_geometry
"""
from picogk_mp.physics.params import Param
from picogk_mp.physics.checks import (
    CheckResult, BaseCheck,
    TippingCheck, SectionBendingCheck, StemBendingCheck,
    BendingRequirement, TorsionRequirement, BucklingRequirement,
    TensionRequirement, DragRequirement,
)
from picogk_mp.physics.requirement import (
    PhysicsRequirement,
    RequirementResult,
    TippingRequirement,
)
from picogk_mp.physics.engine import SimEngine
from picogk_mp.physics.brief import (
    PhysicsBrief, Material, LoadCase, Constraint,
    FailureCriteria, DesignIntent,
    MaterialPreset, LoadType, LoadCombination,
    ConstraintType, ComponentType, GeometryLanguage,
)
from picogk_mp.physics.brief_mapper import (
    brief_to_requirements,
    brief_to_sim_engine,
    brief_to_boundary_conditions,
    brief_to_topopt_kwargs,
    suggest_geometry,
)

__all__ = [
    # Core engine
    "Param",
    "SimEngine",
    # Result types
    "CheckResult",
    "RequirementResult",
    # Base classes
    "BaseCheck",
    "PhysicsRequirement",
    # Concrete requirements
    "BendingRequirement",
    "TippingRequirement",
    "TorsionRequirement",
    "BucklingRequirement",
    "TensionRequirement",
    "DragRequirement",
    # Backward-compat aliases
    "TippingCheck",
    "SectionBendingCheck",
    "StemBendingCheck",
    # Brief schema
    "PhysicsBrief",
    "Material",
    "LoadCase",
    "Constraint",
    "FailureCriteria",
    "DesignIntent",
    # Enums
    "MaterialPreset",
    "LoadType",
    "LoadCombination",
    "ConstraintType",
    "ComponentType",
    "GeometryLanguage",
    # Mapper
    "brief_to_requirements",
    "brief_to_sim_engine",
    "brief_to_boundary_conditions",
    "brief_to_topopt_kwargs",
    "suggest_geometry",
]
