"""Built-in physics checks.

Each check is a stateless callable: BaseCheck.evaluate(ctx) -> CheckResult.
*ctx* is a plain dict[str, Any] of resolved parameter values.

Checks declare which keys they need via *required_params*.  The engine
verifies all keys are present before calling evaluate().

Built-in checks
---------------
TippingCheck      -- static tip-over stability under vertical point load
StemBendingCheck  -- bending stress at the narrowest stem cross-section
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple


# ======================================================================
# Result container
# ======================================================================

@dataclass
class CheckResult:
    """Outcome of a single physics check."""

    name: str
    passed: bool
    sf: float          # actual safety factor (> sf_required = pass)
    sf_required: float
    detail: str        # human-readable summary line

    def __str__(self) -> str:
        status = "OK  " if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.detail}"


# ======================================================================
# Base class
# ======================================================================

class BaseCheck:
    """Abstract base for all physics checks."""

    name: str = "BaseCheck"
    sf_required: float = 1.0
    required_params: Tuple[str, ...] = ()

    def evaluate(self, ctx: Dict[str, Any]) -> CheckResult:
        raise NotImplementedError

    def _verify_ctx(self, ctx: Dict[str, Any]) -> None:
        missing = [k for k in self.required_params if k not in ctx]
        if missing:
            raise KeyError(
                f"Check '{self.name}' benoetigt fehlende Parameter: {missing}"
            )


# ======================================================================
# Tipping stability
# ======================================================================

class TippingCheck(BaseCheck):
    """Static tip-over check under vertical point load at arm tip.

    Tipping axis = outermost base edge in the direction of the load.
    Safety factor = restoring_moment / tipping_moment.

    Required context keys
    ---------------------
    head_mass_g     : design load [g]
    base_r_mm       : base radius [mm]
    arm_reach_mm    : horizontal distance from base axis to load point [mm]
    volume_mm3      : voxel volume of the printed part [mm^3]
    infill_pct      : print infill percentage [%]  (e.g. 20)
    density_g_cm3   : raw filament density [g/cm^3] (e.g. 1.24 for PLA)
    """

    name = "Kippstabilitaet"
    sf_required = 1.5
    required_params = (
        "head_mass_g", "base_r_mm", "arm_reach_mm",
        "volume_mm3", "infill_pct", "density_g_cm3",
    )

    def evaluate(self, ctx: Dict[str, Any]) -> CheckResult:
        self._verify_ctx(ctx)

        head_g   = ctx["head_mass_g"]
        base_r   = ctx["base_r_mm"]
        reach    = ctx["arm_reach_mm"]
        vol      = ctx["volume_mm3"]
        infill   = ctx["infill_pct"] / 100.0
        rho      = ctx["density_g_cm3"] * infill          # eff. g/cm^3
        stand_g  = vol * rho / 1000.0                     # cm^3 * g/cm^3 = g

        lever_load  = reach - base_r    # [mm]  positive = arm beyond base
        lever_stand = base_r            # [mm]  conservative: CG at centre

        if lever_load <= 0:
            return CheckResult(
                self.name, True, 999.0, self.sf_required,
                f"Arm innerhalb Basis (reach={reach:.0f} mm <= base_r={base_r:.0f} mm) -- kein Kipprisiko",
            )

        tip_moment     = head_g  * lever_load   # [g*mm]
        restore_moment = stand_g * lever_stand  # [g*mm]
        sf = restore_moment / tip_moment

        # Minimum base radius for SF=sf_required
        min_r = (head_g * self.sf_required * reach) / (stand_g + head_g * self.sf_required)

        detail = (
            f"SF={sf:.2f} (Mindest {self.sf_required})  |  "
            f"Kippmoment {tip_moment:.0f} g*mm  "
            f"Gegenmoment {restore_moment:.0f} g*mm  "
            f"Staendermasse {stand_g:.1f} g"
        )
        if not sf >= self.sf_required:
            detail += f"  |  --> min. Basisradius: {min_r:.0f} mm"

        return CheckResult(self.name, sf >= self.sf_required, sf, self.sf_required, detail)


# ======================================================================
# Stem bending stress
# ======================================================================

class StemBendingCheck(BaseCheck):
    """Bending stress at the narrowest cross-section of the stem.

    Assumes a solid circular cross-section.  Bending is caused by the
    horizontal moment arm from the arm attachment point to the stem foot.

    Required context keys
    ---------------------
    head_mass_g     : design load [g]
    arm_reach_mm    : horizontal reach [mm]
    stem_r_min_mm   : radius at the narrowest stem cross-section [mm]
    yield_mpa       : yield strength of filament material [MPa]
    """

    name = "Stammstiel Biegung"
    sf_required = 3.0
    required_params = (
        "head_mass_g", "arm_reach_mm", "stem_r_min_mm", "yield_mpa",
    )

    def evaluate(self, ctx: Dict[str, Any]) -> CheckResult:
        self._verify_ctx(ctx)

        F_N    = ctx["head_mass_g"] * 9.81e-3           # N
        arm_m  = ctx["arm_reach_mm"] * 1e-3             # m
        r_m    = ctx["stem_r_min_mm"] * 1e-3            # m
        sig_y  = ctx["yield_mpa"]                       # MPa

        M_Nm   = F_N * arm_m                            # bending moment [N*m]
        I_m4   = math.pi * r_m**4 / 4.0                # 2nd moment of area [m^4]
        sigma  = M_Nm * r_m / I_m4 / 1e6               # bending stress [MPa]

        sf = sig_y / sigma if sigma > 0 else float("inf")

        detail = (
            f"SF={sf:.1f} (Mindest {self.sf_required})  |  "
            f"Moment {M_Nm*1000:.1f} N*mm  "
            f"Spannung {sigma:.2f} MPa  "
            f"Streckgrenze {sig_y} MPa"
        )

        return CheckResult(self.name, sf >= self.sf_required, sf, self.sf_required, detail)
