"""SimEngine: orchestrates parameter resolution and physics checks.

Workflow
--------
1. Register Param objects (some known, some unknown).
2. Register checks (TippingCheck, CantileverBendingCheck, ...).
3. Call engine.inject(**geometry_values) to push in computed geometry.
4. Call engine.run():
      a. For each unresolved Param:  ask user via CLI (or use resolver).
      b. Build context dict from all resolved values.
      c. Run every check; print pass/fail summary.
      d. Raise PhysicsFailure if any mandatory check failed.

Non-interactive mode (tests, CI)
---------------------------------
Pass a *resolver* callable to the constructor::

    def my_resolver(param: Param):
        return {"load_mass_g": 400, "infill_pct": 15}[param.key]

    engine = SimEngine(resolver=my_resolver)

Or pass a plain dict shorthand::

    engine = SimEngine(resolver={"load_mass_g": 400})
"""
from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Optional, Union

from picogk_mp.physics.params import Param
from picogk_mp.physics.checks import BaseCheck, CheckResult


# ======================================================================
# Exceptions
# ======================================================================

class PhysicsFailure(Exception):
    """Raised when one or more mandatory checks fail."""


# ======================================================================
# Engine
# ======================================================================

class SimEngine:
    """Multiphysics simulation engine.

    Parameters
    ----------
    resolver:
        Optional override for the interactive CLI query.
        Can be:
          - a callable (Param) -> Any
          - a dict {key: value}  (convenience shorthand)
          - None  (default: interactive stdin)
    """

    def __init__(
        self,
        resolver: Optional[Union[Callable[[Param], Any], Dict[str, Any]]] = None,
    ) -> None:
        self._params: Dict[str, Param] = {}
        self._checks: List[BaseCheck] = []

        if isinstance(resolver, dict):
            _d = resolver
            self._resolver: Optional[Callable[[Param], Any]] = lambda p: _d[p.key]
        else:
            self._resolver = resolver  # callable or None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, *params: Param) -> "SimEngine":
        """Add Param definitions to the engine (fluent API)."""
        for p in params:
            self._params[p.key] = p
        return self

    def add_check(self, check: BaseCheck) -> "SimEngine":
        """Register a physics check (fluent API)."""
        self._checks.append(check)
        return self

    # ------------------------------------------------------------------
    # Inject known geometry values after build
    # ------------------------------------------------------------------

    def inject(self, **kwargs: Any) -> "SimEngine":
        """Push computed geometry values into registered params.

        Only updates params that have been registered; unknown keys are
        silently ignored so callers don't need to filter themselves.
        """
        for key, val in kwargs.items():
            if key in self._params:
                self._params[key].set(val)
        return self

    # ------------------------------------------------------------------
    # Core: resolve unknowns
    # ------------------------------------------------------------------

    def _query_param(self, p: Param) -> Any:
        """Ask user for a single param value and store it."""
        if self._resolver is not None:
            raw = self._resolver(p)
            p.set(raw)
            return p.value

        # Interactive CLI
        while True:
            try:
                raw = input(p.prompt_text()).strip()
            except EOFError:
                # Non-interactive stdin (e.g. piped empty input): use default or fail
                if p.default is not None:
                    print(f"(kein Input -- verwende Default: {p.default})")
                    p.set(p.default)
                    return p.value
                raise

            if raw == "" and p.default is not None:
                p.set(p.default)
                return p.value
            try:
                p.set(raw)
                return p.value
            except ValueError as exc:
                print(f"  Fehler: {exc}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, raise_on_failure: bool = True) -> List[CheckResult]:
        """Resolve unknowns, execute all checks, print summary.

        Parameters
        ----------
        raise_on_failure:
            If True (default), raise PhysicsFailure when any check fails.
            Set False to collect results without aborting.

        Returns
        -------
        List of CheckResult, one per registered check.
        """
        # 1. Query params that have neither an explicit value nor a default.
        # Params with a default are used as-is; they are NOT queried unless
        # the caller wants to override them (which they can do via inject()).
        unknowns = [
            p for p in self._params.values()
            if p.value is None and p.default is None
        ]
        if unknowns:
            print()
            print("=" * 60)
            print("  MULTIPHYSIK ENGINE -- fehlende Parameter")
            print("=" * 60)
            for p in unknowns:
                self._query_param(p)
            print()

        # 2. Build context
        ctx: Dict[str, Any] = {}
        for key, p in self._params.items():
            try:
                ctx[key] = p.resolved_value
            except ValueError as exc:
                raise PhysicsFailure(str(exc)) from exc

        # 3. Run checks
        print("=" * 60)
        print("  PHYSIK-CHECKS")
        print("=" * 60)
        results: List[CheckResult] = []
        for check in self._checks:
            try:
                r = check.evaluate(ctx)
            except KeyError as exc:
                raise PhysicsFailure(f"Check '{check.name}' fehlgeschlagen: {exc}") from exc
            print(f"  {r}")
            results.append(r)

        failed = [r for r in results if not r.passed]
        if failed:
            print()
            print(f"  !! {len(failed)} Check(s) NICHT bestanden !!")
        else:
            print()
            print("  Alle Checks bestanden.")
        print("=" * 60)
        print()

        if failed and raise_on_failure:
            names = ", ".join(r.name for r in failed)
            raise PhysicsFailure(
                f"Design abgelehnt -- fehlgeschlagene Checks: {names}"
            )

        return results

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a short text summary of registered params and checks."""
        lines = ["SimEngine"]
        lines.append(f"  Params ({len(self._params)}):")
        for p in self._params.values():
            state = f"= {p.resolved_value}" if p.is_resolved else "(unbekannt)"
            dflt  = f", default={p.default}" if p.default is not None else ""
            lines.append(f"    {p.key}: {state}{dflt}  [{p.unit}]")
        lines.append(f"  Checks ({len(self._checks)}):")
        for c in self._checks:
            lines.append(f"    {c.name}  (SF>={c.sf_required})")
        return "\n".join(lines)
