"""TPMS (Triply Periodic Minimal Surface) implicit functions for pycogk.

Both surfaces are defined as sheet-TPMSs: the solid region is where
    |f(x,y,z)| < isovalue
so fSignedDistance = |f| - isovalue  (negative = material, positive = void).

'isovalue' is dimensionless (function-space threshold, NOT mm wall thickness).
Typical range: 0.2 (thin walls) … 1.0 (thick walls / high volume fraction).

For a sense of scale:
  Gyroid    f ranges ≈ ±1.73  →  isovalue 0.5 ≈ 35-40 % volume fraction
  Schwartz-P f ranges ≈ ±3.0  →  isovalue 1.0 ≈ 35-40 % volume fraction
"""
from __future__ import annotations

import math

from picogk import IBoundedImplicit

Vector3 = tuple[float, float, float]


class Gyroid(IBoundedImplicit):
    """Gyroid TPMS sheet: sin(kx)cos(ky) + sin(ky)cos(kz) + sin(kz)cos(kx) = 0."""

    def __init__(
        self,
        cell_size_mm: float,
        isovalue: float,
        bounds_mm: tuple[float, float, float],
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self._k = 2.0 * math.pi / cell_size_mm
        self._iso = isovalue
        self._half = (bounds_mm[0] / 2.0, bounds_mm[1] / 2.0, bounds_mm[2] / 2.0)
        self._cx, self._cy, self._cz = center

    def fSignedDistance(self, vecPt) -> float:  # noqa: N802
        x = vecPt[0] - self._cx
        y = vecPt[1] - self._cy
        z = vecPt[2] - self._cz
        k = self._k
        f = (
            math.sin(k * x) * math.cos(k * y)
            + math.sin(k * y) * math.cos(k * z)
            + math.sin(k * z) * math.cos(k * x)
        )
        return abs(f) - self._iso

    @property
    def oBounds(self) -> tuple[Vector3, Vector3]:  # noqa: N802
        hx, hy, hz = self._half
        cx, cy, cz = self._cx, self._cy, self._cz
        return (
            (cx - hx, cy - hy, cz - hz),
            (cx + hx, cy + hy, cz + hz),
        )


class SchwartzP(IBoundedImplicit):
    """Schwartz-P TPMS sheet: cos(kx) + cos(ky) + cos(kz) = 0."""

    def __init__(
        self,
        cell_size_mm: float,
        isovalue: float,
        bounds_mm: tuple[float, float, float],
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self._k = 2.0 * math.pi / cell_size_mm
        self._iso = isovalue
        self._half = (bounds_mm[0] / 2.0, bounds_mm[1] / 2.0, bounds_mm[2] / 2.0)
        self._cx, self._cy, self._cz = center

    def fSignedDistance(self, vecPt) -> float:  # noqa: N802
        x = vecPt[0] - self._cx
        y = vecPt[1] - self._cy
        z = vecPt[2] - self._cz
        k = self._k
        f = math.cos(k * x) + math.cos(k * y) + math.cos(k * z)
        return abs(f) - self._iso

    @property
    def oBounds(self) -> tuple[Vector3, Vector3]:  # noqa: N802
        hx, hy, hz = self._half
        cx, cy, cz = self._cx, self._cy, self._cz
        return (
            (cx - hx, cy - hy, cz - hz),
            (cx + hx, cy + hy, cz + hz),
        )
