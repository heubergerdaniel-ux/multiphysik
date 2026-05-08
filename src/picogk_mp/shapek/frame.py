"""LocalFrame -- position + orthonormal basis for SDF evaluation."""
from __future__ import annotations

from typing import Sequence

import numpy as np

Vec3 = np.ndarray  # shape (3,), dtype float64


class LocalFrame:
    """Position + orthonormal basis (tangent T, normal N, binormal B).

    Convention: T = primary axis (spine direction), N = secondary,
    B = T x N (right-hand rule, computed automatically).

    Parameters
    ----------
    origin  : (3,) position in world mm
    tangent : (3,) primary axis (will be normalised)
    normal  : (3,) secondary axis (will be Gram-Schmidt orthogonalised
              against T, then normalised)
    """

    def __init__(
        self,
        origin:  Sequence[float] = (0.0, 0.0, 0.0),
        tangent: Sequence[float] = (0.0, 0.0, 1.0),
        normal:  Sequence[float] = (1.0, 0.0, 0.0),
    ) -> None:
        self.origin = np.asarray(origin, dtype=float)
        T = np.asarray(tangent, dtype=float)
        N = np.asarray(normal,  dtype=float)
        norm_T = np.linalg.norm(T)
        if norm_T < 1e-12:
            raise ValueError("tangent vector must be non-zero")
        self.T = T / norm_T
        N = N - np.dot(N, self.T) * self.T  # Gram-Schmidt
        norm_N = np.linalg.norm(N)
        if norm_N < 1e-12:
            raise ValueError("normal vector must not be parallel to tangent")
        self.N = N / norm_N
        self.B = np.cross(self.T, self.N)

    def to_local(self, pts: np.ndarray) -> np.ndarray:
        """Transform (N,3) world points to frame-local coordinates.

        Returns (N,3): columns are [along T, along N, along B].
        """
        d = pts - self.origin
        return np.stack([d @ self.T, d @ self.N, d @ self.B], axis=1)

    def to_world(self, local: np.ndarray) -> np.ndarray:
        """Transform (N,3) local coordinates back to world space."""
        return (
            self.origin
            + local[:, 0:1] * self.T
            + local[:, 1:2] * self.N
            + local[:, 2:3] * self.B
        )

    @classmethod
    def along_segment(
        cls,
        a: Sequence[float],
        b: Sequence[float],
    ) -> "LocalFrame":
        """Frame at midpoint of segment a->b, T pointing from a to b."""
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        T = b_arr - a_arr
        # Choose N perpendicular to T using the most stable axis
        T_norm = T / np.linalg.norm(T)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(T_norm, axis)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        N = np.cross(T, axis)
        return cls(origin=(a_arr + b_arr) / 2, tangent=T, normal=N)

    @classmethod
    def world(cls) -> "LocalFrame":
        """Identity frame at origin, T=+Z, N=+X."""
        return cls((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
