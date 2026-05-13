"""Multiphysics simulation engine for PicoGK pipeline.

Public API:
    Param       -- declares a physical parameter (known or unknown)
    SimEngine   -- orchestrates parameter resolution + check execution
    CheckResult -- result of a single physics check

Built-in checks (import from picogk_mp.physics.checks):
    TippingCheck, CantileverBendingCheck
"""
from picogk_mp.physics.params import Param
from picogk_mp.physics.checks import (
    CheckResult,
    BaseCheck,
    TippingCheck,
    CantileverBendingCheck,
    StemBendingCheck,  # deprecated alias
)
from picogk_mp.physics.engine import SimEngine

__all__ = [
    "Param",
    "SimEngine",
    "CheckResult",
    "BaseCheck",
    "TippingCheck",
    "CantileverBendingCheck",
    "StemBendingCheck",
]
