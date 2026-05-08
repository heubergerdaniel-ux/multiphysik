"""Parametric modulation classes for varying shape properties along a path."""
from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np
from scipy.interpolate import CubicSpline

_Scalar = Union[float, int]


class ControlPointSpline:
    """Cubic spline through (t, value) control points.

    Parameters
    ----------
    control_points : sequence of (t, value) tuples, t values in [0, 1].
                     At least 2 points required; t values need not be sorted.
    """

    def __init__(self, control_points: Sequence[tuple[float, float]]) -> None:
        if len(control_points) < 2:
            raise ValueError("ControlPointSpline needs at least 2 control points")
        ts = np.array([p[0] for p in control_points], dtype=float)
        vs = np.array([p[1] for p in control_points], dtype=float)
        idx = np.argsort(ts)
        self._spline = CubicSpline(ts[idx], vs[idx], extrapolate=True)

    def evaluate(self, t: float) -> float:
        return float(self._spline(np.clip(t, 0.0, 1.0)))

    def evaluate_array(self, ts: np.ndarray) -> np.ndarray:
        return self._spline(np.clip(ts, 0.0, 1.0))


class LineModulation:
    """Parametric 1D variation over [0, 1] -> float.

    Three modes selected by which keyword argument is supplied:

    Constant  : LineModulation(value=5.0)
    Function  : LineModulation(func=lambda t: 5 + 2*t)
    Spline    : LineModulation(spline=ControlPointSpline(...))

    Convenience constructors
    ------------------------
    LineModulation.constant(v)
    LineModulation.from_function(f)
    LineModulation.from_endpoints(v0, v1)
    LineModulation.from_control_points([(t0,v0), ...])
    """

    def __init__(
        self,
        *,
        value:  _Scalar | None = None,
        func:   Callable[[float], float] | None = None,
        spline: ControlPointSpline | None = None,
    ) -> None:
        given = sum(x is not None for x in (value, func, spline))
        if given != 1:
            raise ValueError("Exactly one of value/func/spline must be provided")
        self._value  = float(value) if value is not None else None
        self._func   = func
        self._spline = spline

    # --- Constructors --------------------------------------------------

    @classmethod
    def constant(cls, value: _Scalar) -> "LineModulation":
        return cls(value=float(value))

    @classmethod
    def from_function(cls, func: Callable[[float], float]) -> "LineModulation":
        return cls(func=func)

    @classmethod
    def from_endpoints(cls, v0: _Scalar, v1: _Scalar) -> "LineModulation":
        """Linear taper from v0 at t=0 to v1 at t=1."""
        return cls(spline=ControlPointSpline([(0.0, float(v0)), (1.0, float(v1))]))

    @classmethod
    def from_control_points(
        cls, points: Sequence[tuple[float, float]]
    ) -> "LineModulation":
        return cls(spline=ControlPointSpline(points))

    # --- Evaluation ----------------------------------------------------

    def at(self, t: float) -> float:
        """Evaluate at scalar t in [0, 1]."""
        if self._value is not None:
            return self._value
        if self._func is not None:
            return float(self._func(float(t)))
        return self._spline.evaluate(t)  # type: ignore[union-attr]

    def at_array(self, ts: np.ndarray) -> np.ndarray:
        """Evaluate at an array of t values -- vectorised."""
        if self._value is not None:
            return np.full(len(ts), self._value)
        if self._func is not None:
            return np.vectorize(self._func)(ts.astype(float))
        return self._spline.evaluate_array(ts)  # type: ignore[union-attr]


class SurfaceModulation:
    """Parametric 2D variation over (u, v) in [0, 1]^2 -> float.

    For surface-following thickness or property variations.
    Currently supports constant or callable f(u, v) -> float.
    """

    def __init__(
        self,
        *,
        value: _Scalar | None = None,
        func:  Callable[[float, float], float] | None = None,
    ) -> None:
        given = sum(x is not None for x in (value, func))
        if given != 1:
            raise ValueError("Exactly one of value/func must be provided")
        self._value = float(value) if value is not None else None
        self._func  = func

    @classmethod
    def constant(cls, value: _Scalar) -> "SurfaceModulation":
        return cls(value=float(value))

    def at(self, u: float, v: float) -> float:
        if self._value is not None:
            return self._value
        return float(self._func(float(u), float(v)))  # type: ignore[misc]
