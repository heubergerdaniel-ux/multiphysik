"""Parametric L-bracket generator (SDF + marching cubes).

Builds a wall-mounted L-bracket entirely from SDF primitives -- no picogk
runtime required.  The result is ready for BESO topology optimisation with
fixture_type="face", fixed_face="x0", load_direction="gravity".

Coordinate convention
---------------------
  x  : depth (0 = wall surface, positive = away from wall)
  y  : width (lateral)
  z  : height (up)

  Wall face  = x0 (the part bolts to the wall there)
  Shelf face = top surface of the horizontal arm
  Load point = centre of the arm tip: (back_thickness + arm_length, width/2, arm_z)

Anatomy
-------
   ___________________________
  |  back plate (vertical)   |
  |                          |  <- wall at x = 0
  |_________ arm ____________|
             |
             v  gravity load

  Optional diagonal gusset fills the inside corner for rigidity.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# SDF primitives
# ---------------------------------------------------------------------------

def _sdf_box(pts: np.ndarray,
             x0: float, x1: float,
             y0: float, y1: float,
             z0: float, z1: float) -> np.ndarray:
    """Exact SDF for an axis-aligned box [x0,x1] x [y0,y1] x [z0,z1]."""
    cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    hx, hy, hz = (x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2

    qx = np.abs(pts[:, 0] - cx) - hx
    qy = np.abs(pts[:, 1] - cy) - hy
    qz = np.abs(pts[:, 2] - cz) - hz

    outer = np.sqrt(
        np.maximum(qx, 0.0) ** 2 +
        np.maximum(qy, 0.0) ** 2 +
        np.maximum(qz, 0.0) ** 2
    )
    inner = np.minimum(np.maximum(np.maximum(qx, qy), qz), 0.0)
    return outer + inner


def _sdf_wedge(pts: np.ndarray,
               x0: float, x1: float,
               y0: float, y1: float,
               z_top_at_x0: float,
               z_top_at_x1: float,
               z_bot: float) -> np.ndarray:
    """Approximate SDF for a triangular prism (gusset wedge).

    The wedge has:
      - rectangular cross-section in y direction (y0..y1)
      - triangular cross-section in xz plane:
          bottom edge at z=z_bot (x0..x1),
          top-left corner at (x0, z_top_at_x0),
          top-right corner at (x1, z_top_at_x1).

    Implemented as a half-space intersection (good enough for SDF union).
    """
    # Inside if: y in [y0,y1]  AND  z >= z_bot  AND  below the slope line
    slope = (z_top_at_x1 - z_top_at_x0) / max(x1 - x0, 1e-6)
    z_slope = z_top_at_x0 + slope * (pts[:, 0] - x0)  # height of slope at each x

    # SDF approximation via smooth max of three constraints:
    d_y0  =  y0 - pts[:, 1]         # positive outside y0 edge
    d_y1  =  pts[:, 1] - y1         # positive outside y1 edge
    d_zb  =  z_bot - pts[:, 2]      # positive below bottom
    d_top =  pts[:, 2] - z_slope    # positive above slope
    d_x0  =  x0 - pts[:, 0]
    d_x1  =  pts[:, 0] - x1

    # SDF = max of all signed distances to each half-space boundary
    # (union of outsides = intersection of insides)
    return np.maximum.reduce([d_y0, d_y1, d_zb, d_top, d_x0, d_x1])


# ---------------------------------------------------------------------------
# Bracket SDF
# ---------------------------------------------------------------------------

def _bracket_sdf(
    pts: np.ndarray,
    T: float,    # back plate thickness (x direction)
    W: float,    # width (y direction)
    H: float,    # back plate height (z direction)
    ah: float,   # arm cross-section height
    al: float,   # arm length (x direction, beyond back plate)
    gusset: bool,
) -> np.ndarray:
    """Union SDF of all bracket components."""

    # 1. Vertical back plate: x=[0,T], y=[0,W], z=[0,H]
    d = _sdf_box(pts, 0.0, T, 0.0, W, 0.0, H)

    # 2. Horizontal arm: x=[T, T+al], y=[0,W], z=[H-ah, H]
    d = np.minimum(d, _sdf_box(pts, T, T + al, 0.0, W, H - ah, H))

    # 3. Optional diagonal gusset fills the inside corner
    #    Wedge: x=[T, T+al*0.6], y=[0,W], z from H-ah (at x=T) to H-ah (at x=T+al*0.6)
    #    Actually: a right triangle with the right angle at (T, H-ah),
    #    the hypotenuse going from (T, H) to (T + al*0.6, H-ah).
    if gusset:
        gx1 = T + al * 0.55
        d = np.minimum(d, _sdf_wedge(
            pts,
            x0=T,   x1=gx1,
            y0=0.0, y1=W,
            z_top_at_x0=H,       # at wall: gusset goes all the way to top
            z_top_at_x1=H - ah,  # at gusset tip: just meets the arm bottom
            z_bot=H - ah,        # bottom of gusset = bottom of arm
        ))

    return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_bracket_stl(
    back_height_mm:     float = 80.0,
    back_width_mm:      float = 60.0,
    back_thickness_mm:  float = 8.0,
    arm_length_mm:      float = 120.0,
    arm_height_mm:      float = 20.0,
    gusset:             bool  = True,
    resolution_mm:      float = 1.0,
    out_stl:            Optional[str | Path] = None,
) -> dict:
    """Generate a parametric wall-mounted L-bracket STL.

    The bracket mounts with its x0 face against the wall.  Gravity load
    acts downward (-z) at the tip of the arm (far end, centre of width).

    Use with run_topopt:
        fixture_type   = "face"
        fixed_face     = "x0"
        load_direction = "gravity"
        load_point_x/y/z_mm from the returned values.

    Parameters
    ----------
    back_height_mm    : Height of the back (wall-mounting) plate [mm].
    back_width_mm     : Width of the bracket (y direction) [mm].
    back_thickness_mm : Thickness of the back plate and arm [mm].
    arm_length_mm     : Length of the horizontal arm [mm].
    arm_height_mm     : Height of the arm cross-section [mm].
    gusset            : Add a diagonal gusset at the inside corner.
    resolution_mm     : SDF grid pitch [mm].  1 mm ~1-2 s.
    out_stl           : Output path.  Default: docs/generated_bracket.stl.

    Returns
    -------
    dict with status, stl_path, volume_mm3, load_point_x/y/z_mm,
    fixture_type, fixed_face, elapsed_s.
    """
    from skimage.measure import marching_cubes as _mc

    T  = back_thickness_mm
    W  = back_width_mm
    H  = back_height_mm
    ah = arm_height_mm
    al = arm_length_mm
    res = float(resolution_mm)

    t0 = time.time()

    # Bounding box with padding
    pad = 5.0
    x_min, x_max = -pad, T + al + pad
    y_min, y_max = -pad, W + pad
    z_min, z_max = -pad, H + pad

    xs = np.arange(x_min, x_max + res, res)
    ys = np.arange(y_min, y_max + res, res)
    zs = np.arange(z_min, z_max + res, res)
    Nx, Ny, Nz = len(xs), len(ys), len(zs)

    print(
        f"Generating bracket SDF at {res} mm: "
        f"{Nx} x {Ny} x {Nz} = {Nx*Ny*Nz:,} pts ..."
    )

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    sdf = _bracket_sdf(pts, T=T, W=W, H=H, ah=ah, al=al, gusset=gusset)
    sdf = sdf.reshape(Nx, Ny, Nz)

    # Marching cubes
    verts, faces, normals, _ = _mc(sdf, level=0.0, spacing=(res, res, res))
    verts += np.array([x_min, y_min, z_min])

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    if mesh.volume < 0:
        mesh.invert()

    volume_mm3 = float(mesh.volume)

    # Output path
    if out_stl is None:
        _root = Path(__file__).resolve().parent.parent.parent.parent
        out_path = _root / "docs" / "generated_bracket.stl"
    else:
        out_path = Path(out_stl)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))

    elapsed = time.time() - t0
    size_kb = out_path.stat().st_size // 1024
    print(f"  Generated STL -> {out_path}  ({size_kb} KB, {volume_mm3:.0f} mm3, {elapsed:.1f}s)")

    # Load point: centre of arm tip face, mid-height of arm
    lx = T + al                 # tip of arm (far x)
    ly = W / 2.0                # centre of width
    lz = H - ah / 2.0           # mid-height of arm

    return {
        "status":          "ok",
        "stl_path":        str(out_path),
        "volume_mm3":      round(volume_mm3),
        "load_point_x_mm": round(lx, 1),
        "load_point_y_mm": round(ly, 1),
        "load_point_z_mm": round(lz, 1),
        "fixture_type":    "face",
        "fixed_face":      "x0",
        "load_direction":  "gravity",
        "elapsed_s":       round(elapsed, 1),
    }
