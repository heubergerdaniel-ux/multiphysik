"""Extended SDF primitives for the shapek module.

All functions have signature:  f(pts: (N,3) ndarray, ...) -> (N,) ndarray
  - Negative = inside solid, positive = outside.
  - SDF union = np.minimum(a, b).
  - All coordinates in mm.

This module consolidates the private SDF helpers that were duplicated
across generators/shape.py and generators/holder.py and adds:
  sdf_cone     -- solid cone from apex to base circle
  sdf_torus    -- torus (ring tube) in arbitrary orientation
  sdf_revolve  -- arbitrary solid of revolution from a 2D profile SDF
  sdf_pipe     -- pipe along a polyline spine with modulated radius
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from picogk_mp.shapek.frame import LocalFrame
from picogk_mp.shapek.modulation import LineModulation


# ---------------------------------------------------------------------------
# Consolidated base primitives  (replace private helpers in generators/)
# ---------------------------------------------------------------------------

def sdf_sphere(pts: np.ndarray, center: Sequence[float], radius: float) -> np.ndarray:
    """Exact sphere SDF."""
    c = np.asarray(center, dtype=float)
    return np.linalg.norm(pts - c, axis=1) - float(radius)


def sdf_box(pts: np.ndarray, mn: Sequence[float], mx: Sequence[float]) -> np.ndarray:
    """Axis-aligned solid box SDF (exact interior + exterior)."""
    lo = np.asarray(mn, dtype=float)
    hi = np.asarray(mx, dtype=float)
    c  = (lo + hi) / 2
    h  = (hi - lo) / 2
    q  = np.abs(pts - c) - h
    outer = np.sqrt((np.maximum(q, 0.0)**2).sum(axis=1))
    inner = np.minimum(np.max(q, axis=1), 0.0)
    return outer + inner


def sdf_capsule(
    pts: np.ndarray,
    a:   Sequence[float],
    b:   Sequence[float],
    ra:  float,
    rb:  float,
) -> np.ndarray:
    """Tapered capsule (conical beam) SDF -- exact."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    ra, rb = float(ra), float(rb)
    ab  = b_arr - a_arr
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-12:
        return sdf_sphere(pts, a_arr, ra)
    ap   = pts - a_arr
    t    = np.clip(ap @ ab / ab2, 0.0, 1.0)
    r_t  = ra + t * (rb - ra)
    closest = a_arr + t[:, None] * ab
    return np.linalg.norm(pts - closest, axis=1) - r_t


def sdf_cylinder(
    pts:        np.ndarray,
    center_xy:  Sequence[float],
    z_range:    Sequence[float],
    radius:     float,
) -> np.ndarray:
    """Upright capped cylinder (z-axis aligned) SDF -- exact."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    z0, z1 = float(z_range[0]), float(z_range[1])
    r      = float(radius)
    d_r    = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2) - r
    d_z    = np.maximum(z0 - pts[:, 2], pts[:, 2] - z1)
    outer  = np.sqrt(np.maximum(d_r, 0)**2 + np.maximum(d_z, 0)**2)
    inner  = np.minimum(np.maximum(d_r, d_z), 0.0)
    return outer + inner


# ---------------------------------------------------------------------------
# New primitives
# ---------------------------------------------------------------------------

def sdf_cone(
    pts:    np.ndarray,
    apex:   Sequence[float],
    base:   Sequence[float],
    r_base: float,
) -> np.ndarray:
    """Solid cone from apex point to a circular base disc.

    Parameters
    ----------
    pts    : (N,3) query points
    apex   : [x,y,z] tip of the cone
    base   : [x,y,z] centre of the circular base
    r_base : radius of the base circle [mm]

    The cone axis can point in any direction.
    Uses a 2D signed-distance formulation in (radial, axial) space.
    """
    apex_a  = np.asarray(apex,  dtype=float)
    base_c  = np.asarray(base,  dtype=float)
    axis    = base_c - apex_a
    h       = float(np.linalg.norm(axis))
    r_base  = float(r_base)
    if h < 1e-12:
        return np.linalg.norm(pts - apex_a, axis=1) - 0.0

    T  = axis / h
    ap = pts - apex_a
    z  = ap @ T                                          # (N,) along axis
    r_axis = ap - z[:, None] * T
    r  = np.linalg.norm(r_axis, axis=1)                 # (N,) radial from axis

    # Lateral edge: unit vector in 2D (r, z) plane
    d_norm = np.sqrt(r_base**2 + h**2)
    lr, lz = r_base / d_norm, h / d_norm

    # Closest point on lateral edge (clipped to [apex, base circle])
    t_lat = np.clip(r * lr + z * lz, 0.0, d_norm)
    d_lat = np.sqrt(
        np.maximum((r - t_lat * lr)**2 + (z - t_lat * lz)**2, 0.0)
    )

    # Closest point on base disc (z=h, r in [0, r_base])
    r_clamp = np.clip(r, 0.0, r_base)
    d_base  = np.sqrt(
        np.maximum((r - r_clamp)**2 + (z - h)**2, 0.0)
    )

    d_bnd = np.minimum(d_lat, d_base)

    # Interior: 0 <= z <= h  AND  r <= r_base * z / h
    inside = (z >= -1e-9) & (z <= h + 1e-9) & (r <= r_base * z / h + 1e-9)
    return np.where(inside, -d_bnd, d_bnd)


def sdf_torus(
    pts:     np.ndarray,
    center:  Sequence[float],
    major_r: float,
    minor_r: float,
    frame:   LocalFrame | None = None,
) -> np.ndarray:
    """Solid torus (ring tube) SDF.

    Parameters
    ----------
    pts     : (N,3) query points
    center  : [x,y,z] torus centre
    major_r : distance from centre to tube centre-line [mm]
    minor_r : tube cross-section radius [mm]
    frame   : if provided, the torus lies in the frame's N-B plane
              (T is the torus axis of revolution). Default: Z-axis.
    """
    c = np.asarray(center, dtype=float)
    if frame is None:
        frame = LocalFrame(origin=c, tangent=(0, 0, 1), normal=(1, 0, 0))
    loc      = frame.to_local(pts)           # (N,3): [along T, along N, along B]
    axial    = loc[:, 0]                     # distance along torus axis
    radial   = np.sqrt(loc[:, 1]**2 + loc[:, 2]**2)   # radial in N-B plane
    return np.sqrt(
        np.maximum((radial - float(major_r))**2 + axial**2, 0.0)
    ) - float(minor_r)


def sdf_revolve(
    pts:         np.ndarray,
    profile_fn:  Callable[[np.ndarray], np.ndarray],
    axis_origin: Sequence[float] = (0.0, 0.0, 0.0),
    axis_dir:    Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Solid of revolution of an arbitrary 2D profile.

    Parameters
    ----------
    pts        : (N,3) query points
    profile_fn : callable(pts_2d: (N,2)) -> (N,) SDF
                 where pts_2d columns are [r_radial, z_axial].
    axis_origin: point on the rotation axis [mm]
    axis_dir   : direction of the rotation axis

    Example -- revolve a rectangular 2D cross-section into a thick ring:
        def rect_profile(pts2d):
            r, z = pts2d[:,0], pts2d[:,1]
            qr = np.abs(r - 15.0) - 3.0
            qz = np.abs(z) - 5.0
            outer = np.sqrt(np.maximum(qr,0)**2 + np.maximum(qz,0)**2)
            inner = np.minimum(np.maximum(qr, qz), 0.0)
            return outer + inner
        sdf = sdf_revolve(pts, rect_profile)
    """
    frm    = LocalFrame(axis_origin, tangent=axis_dir, normal=(1, 0, 0))
    loc    = frm.to_local(pts)
    axial  = loc[:, 0]
    radial = np.sqrt(loc[:, 1]**2 + loc[:, 2]**2)
    pts_2d = np.stack([radial, axial], axis=1)
    return profile_fn(pts_2d)


def sdf_cylinder_x(
    pts:       np.ndarray,
    center_yz: Sequence[float],
    x_range:   Sequence[float],
    radius:    float,
) -> np.ndarray:
    """x-axis aligned capped cylinder SDF (exact)."""
    cy, cz = float(center_yz[0]), float(center_yz[1])
    x0, x1 = float(x_range[0]), float(x_range[1])
    r      = float(radius)
    d_r    = np.sqrt((pts[:, 1] - cy)**2 + (pts[:, 2] - cz)**2) - r
    d_x    = np.maximum(x0 - pts[:, 0], pts[:, 0] - x1)
    outer  = np.sqrt(np.maximum(d_r, 0)**2 + np.maximum(d_x, 0)**2)
    inner  = np.minimum(np.maximum(d_r, d_x), 0.0)
    return outer + inner


def sdf_cylinder_y(
    pts:       np.ndarray,
    center_xz: Sequence[float],
    y_range:   Sequence[float],
    radius:    float,
) -> np.ndarray:
    """y-axis aligned capped cylinder SDF (exact)."""
    cx, cz = float(center_xz[0]), float(center_xz[1])
    y0, y1 = float(y_range[0]), float(y_range[1])
    r      = float(radius)
    d_r    = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 2] - cz)**2) - r
    d_y    = np.maximum(y0 - pts[:, 1], pts[:, 1] - y1)
    outer  = np.sqrt(np.maximum(d_r, 0)**2 + np.maximum(d_y, 0)**2)
    inner  = np.minimum(np.maximum(d_r, d_y), 0.0)
    return outer + inner


def sdf_pipe(
    pts:        np.ndarray,
    spine:      np.ndarray,
    radius_mod: LineModulation,
    closed:     bool = False,
) -> np.ndarray:
    """Pipe following a polyline spine with modulated radius.

    The pipe is the SDF union of tapered capsule segments along the spine.

    Parameters
    ----------
    pts        : (N,3) query points
    spine      : (M,3) array of world-space waypoints. M >= 2.
    radius_mod : LineModulation giving radius at each t in [0,1].
    closed     : if True, connect last point back to first (not yet used).
    """
    spine = np.asarray(spine, dtype=float)
    M = len(spine)
    if M < 2:
        raise ValueError("Pipe spine needs at least 2 points")

    # Precompute cumulative arc length to get t values per waypoint
    seg_lens = np.array([
        np.linalg.norm(spine[i+1] - spine[i]) for i in range(M - 1)
    ])
    total_len = float(seg_lens.sum())
    if total_len < 1e-12:
        return np.full(len(pts), np.inf)

    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    ts  = cum / total_len                       # t value at each waypoint

    result = np.full(len(pts), np.inf)
    for i in range(M - 1):
        ra  = radius_mod.at(ts[i])
        rb  = radius_mod.at(ts[i + 1])
        seg = sdf_capsule(pts, spine[i], spine[i + 1], ra, rb)
        result = np.minimum(result, seg)
    return result
