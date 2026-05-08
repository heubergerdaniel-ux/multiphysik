"""Aerodynamic physics check for the multiphysik pipeline.

DragCheck is informational: it reports Cd and Re but does not enforce a
mandatory safety factor (aerodynamic drag rarely drives failure for static
3D-printed parts).  A warning is issued when Cd exceeds a configurable
threshold.
"""
from __future__ import annotations

from typing import Any, Dict

from picogk_mp.physics.checks import BaseCheck, CheckResult


class DragCheck(BaseCheck):
    """Aerodynamic drag coefficient check (informational).

    Required context keys
    ---------------------
    Cd      : Drag coefficient (dimensionless).
    Re      : Reynolds number.
    Cd_warn : Threshold above which a warning is issued (default: 2.0).
    """

    name = "Aerodynamischer Widerstand (Cd)"
    sf_required = 0.0     # no mandatory SF -- informational only
    required_params = ("Cd", "Re")

    def __init__(self, Cd_warn: float = 2.0) -> None:
        self.Cd_warn = Cd_warn

    def evaluate(self, ctx: Dict[str, Any]) -> CheckResult:
        self._verify_ctx(ctx)

        Cd = float(ctx["Cd"])
        Re = float(ctx["Re"])
        Cd_warn = float(ctx.get("Cd_warn", self.Cd_warn))

        passed = Cd <= Cd_warn
        sf = Cd_warn / Cd if Cd > 0 else float("inf")

        detail = (
            f"Cd = {Cd:.3f}  Re = {Re:.0f}  "
            f"(Grenzwert Cd_warn = {Cd_warn:.1f})"
        )
        if not passed:
            detail += "  -- hoher Widerstand, aerodynamische Optimierung empfohlen"

        return CheckResult(
            name=self.name,
            passed=passed,
            sf=round(sf, 3),
            sf_required=self.sf_required,
            detail=detail,
        )
