"""Built-in physics checks.

Each check is a stateless callable: BaseCheck.evaluate(ctx) -> CheckResult.
*ctx* is a plain dict[str, Any] of resolved parameter values.

Checks declare which keys they need via *required_params*.  The engine
verifies all keys are present before calling evaluate().

Built-in checks
---------------
TippingCheck      -- static tip-over stability under vertical point load
                     at an offset from a disc base.
CantileverBendingCheck
                  -- bending stress at the narrowest cross-section of a
                     cantilever from the base.
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
    """Static tip-over check under vertical point load at offset.

    Tipping axis = outermost base edge in the direction of the load.
    Safety factor = restoring_moment / tipping_moment.

    Required context keys
    ---------------------
    load_mass_g     : design load [g]
    base_r_mm       : base radius [mm]
    load_offset_mm  : horizontal distance from base axis to load point [mm]
    volume_mm3      : voxel volume of the printed part [mm^3]
    infill_pct      : print infill percentage [%]  (e.g. 20)
    density_g_cm3   : raw filament density [g/cm^3] (e.g. 1.24 for PLA)
    """

    name = "TippingStability"
    sf_required = 1.5
    required_params = (
        "load_mass_g", "base_r_mm", "load_offset_mm",
        "volume_mm3", "infill_pct", "density_g_cm3",
    )

    def evaluate(self, ctx: Dict[str, Any]) -> CheckResult:
        self._verify_ctx(ctx)

        load_g   = ctx["load_mass_g"]
        base_r   = ctx["base_r_mm"]
        offset   = ctx["load_offset_mm"]
        vol      = ctx["volume_mm3"]
        infill   = ctx["infill_pct"] / 100.0
        rho      = ctx["density_g_cm3"] * infill          # eff. g/cm^3
        part_g   = vol * rho / 1000.0                     # cm^3 * g/cm^3 = g

        lever_load = offset - base_r    # [mm]  positive = load beyond base
        lever_part = base_r             # [mm]  conservative: CG at centre

        if lever_load <= 0:
            return CheckResult(
                self.name, True, 999.0, self.sf_required,
                f"Load inside base (offset={offset:.0f} mm <= base_r={base_r:.0f} mm) -- no tipping risk",
            )

        tip_moment     = load_g * lever_load    # [g*mm]
        restore_moment = part_g * lever_part    # [g*mm]
        sf = restore_moment / tip_moment

        # Minimum base radius for SF=sf_required
        min_r = (load_g * self.sf_required * offset) / (part_g + load_g * self.sf_required)

        detail = (
            f"SF={sf:.2f} (min {self.sf_required})  |  "
            f"tip moment {tip_moment:.0f} g*mm  "
            f"restore moment {restore_moment:.0f} g*mm  "
            f"part mass {part_g:.1f} g"
        )
        if not sf >= self.sf_required:
            detail += f"  |  --> min base radius: {min_r:.0f} mm"

        return CheckResult(self.name, sf >= self.sf_required, sf, self.sf_required, detail)


# ======================================================================
# Cantilever bending stress
# ======================================================================

class CantileverBendingCheck(BaseCheck):
    """Bending stress at the narrowest cross-section of a cantilever.

    Assumes a solid circular cross-section.  Bending is caused by the
    horizontal moment arm from the load point to the section.

    Required context keys
    ---------------------
    load_mass_g          : design load [g]
    load_offset_mm       : horizontal offset from cross-section to load [mm]
    min_section_radius_mm: radius at the narrowest cross-section [mm]
    yield_mpa            : yield strength of filament material [MPa]
    """

    name = "CantileverBending"
    sf_required = 3.0
    required_params = (
        "load_mass_g", "load_offset_mm", "min_section_radius_mm", "yield_mpa",
    )

    def evaluate(self, ctx: Dict[str, Any]) -> CheckResult:
        self._verify_ctx(ctx)

        F_N    = ctx["load_mass_g"] * 9.81e-3           # N
        arm_m  = ctx["load_offset_mm"] * 1e-3           # m
        r_m    = ctx["min_section_radius_mm"] * 1e-3    # m
        sig_y  = ctx["yield_mpa"]                       # MPa

        M_Nm   = F_N * arm_m                            # bending moment [N*m]
        I_m4   = math.pi * r_m**4 / 4.0                # 2nd moment of area [m^4]
        sigma  = M_Nm * r_m / I_m4 / 1e6               # bending stress [MPa]

        sf = sig_y / sigma if sigma > 0 else float("inf")

        detail = (
            f"SF={sf:.1f} (min {self.sf_required})  |  "
            f"moment {M_Nm*1000:.1f} N*mm  "
            f"stress {sigma:.2f} MPa  "
            f"yield {sig_y} MPa"
        )

        return CheckResult(self.name, sf >= self.sf_required, sf, self.sf_required, detail)


# Backwards-compat alias (deprecated -- use CantileverBendingCheck)
StemBendingCheck = CantileverBendingCheck
