"""Built-in physics checks — compatibility shim.

All logic has moved to requirement.py (PhysicsRequirement / RequirementResult).
This module re-exports the same names so that existing imports continue to work:

    from picogk_mp.physics.checks import (
        CheckResult, BaseCheck,
        TippingCheck, SectionBendingCheck, StemBendingCheck,
    )
"""
from picogk_mp.physics.requirement import (  # noqa: F401  (re-export)
    RequirementResult as CheckResult,
    PhysicsRequirement as BaseCheck,
    BendingRequirement,
    BendingRequirement as SectionBendingCheck,
    BendingRequirement as StemBendingCheck,
    TippingRequirement as TippingCheck,
    TorsionRequirement,
    BucklingRequirement,
    TensionRequirement,
    DragRequirement,
    DragRequirement as DragCheck,
)

__all__ = [
    "CheckResult",
    "BaseCheck",
    "BendingRequirement",
    "SectionBendingCheck",
    "StemBendingCheck",
    "TippingCheck",
    "TorsionRequirement",
    "BucklingRequirement",
    "TensionRequirement",
    "DragRequirement",
    "DragCheck",
]
