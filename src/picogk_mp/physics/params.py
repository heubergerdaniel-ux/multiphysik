"""Parameter registry: known values, defaults, and unknowns.

A Param is RESOLVED when .value is set (programmatically or via query).
Until then it is UNKNOWN and the engine will ask the user for it.

Usage::

    p = Param("head_mass_g", "Kopfhoerermasse", unit="g", lo=50, hi=1000)
    p.is_resolved   # False -- must be queried
    p.set(350.0)
    p.resolved_value  # 350.0

    q = Param("infill_pct", "Infill", unit="%", default=20, lo=5, hi=100)
    q.is_resolved   # False -- no explicit value yet
    q.resolved_value  # 20.0 -- but default is used when running checks
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, List


@dataclass
class Param:
    """One physical parameter with optional default and validation bounds.

    Attributes
    ----------
    key:      unique identifier used as dict key in checks
    label:    human-readable name shown in CLI prompt
    unit:     physical unit string (shown in prompt, e.g. "g", "mm", "%")
    default:  value used when user hits Enter without typing; None = required
    lo, hi:   inclusive valid range; None = unchecked
    choices:  discrete allowed values (for string/categorical params)
    dtype:    conversion type applied to raw input (default float)
    value:    set programmatically via inject() or set(); None = unknown
    """

    key: str
    label: str
    unit: str
    default: Any = None
    lo: Optional[float] = None
    hi: Optional[float] = None
    choices: Optional[List[Any]] = None
    dtype: type = float
    value: Any = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def is_resolved(self) -> bool:
        """True when an explicit value has been set (not just default)."""
        return self.value is not None

    @property
    def resolved_value(self) -> Any:
        """Return explicit value, then default.  Raise if neither exists."""
        if self.value is not None:
            return self.value
        if self.default is not None:
            return self.dtype(self.default)
        raise ValueError(
            f"Param '{self.key}' ('{self.label}') ist unaufgeloest und hat keinen Default."
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def set(self, raw: Any) -> None:
        """Convert *raw* to dtype, validate, store as .value."""
        try:
            val = self.dtype(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Param '{self.key}': ungueltige Eingabe '{raw}'") from exc

        if self.lo is not None and val < self.lo:
            raise ValueError(f"Param '{self.key}': {val} < Minimum {self.lo}")
        if self.hi is not None and val > self.hi:
            raise ValueError(f"Param '{self.key}': {val} > Maximum {self.hi}")
        if self.choices is not None and val not in self.choices:
            raise ValueError(
                f"Param '{self.key}': '{val}' nicht in Optionen {self.choices}"
            )
        self.value = val

    def reset(self) -> None:
        """Clear explicit value (back to unknown / default state)."""
        self.value = None

    # ------------------------------------------------------------------
    # Prompt helper
    # ------------------------------------------------------------------

    def prompt_text(self) -> str:
        """Build the CLI prompt string shown to the user."""
        parts = [f"  {self.label} [{self.unit}]"]
        if self.lo is not None and self.hi is not None:
            parts.append(f" ({self.lo}-{self.hi})")
        elif self.lo is not None:
            parts.append(f" (>={self.lo})")
        elif self.hi is not None:
            parts.append(f" (<={self.hi})")
        if self.choices:
            parts.append(f" [{'/'.join(str(c) for c in self.choices)}]")
        if self.default is not None:
            parts.append(f" [Enter={self.default}]")
        parts.append(": ")
        return "".join(parts)
