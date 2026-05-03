"""Parametric headphone-holder generator (SDF + marching cubes).

Builds the full holder geometry from a signed-distance-field on a numpy
grid and extracts the mesh with scikit-image marching cubes.  No
picogk.go / VTK / C++ runtime required -- works headlessly inside the
MCP server.

Anatomy
-------
   end-cap sphere   <--- arm tip (hook hangs headphones here)
         |
   arm segment 3    (gently hooks back down to tip)
         |
   arm segment 2    (plateau: headband rests here)
         |
   arm segment 1    (sweeps out and rises from stem top)
         |
   top junction sphere
         |
   tapered stem     (slim column, base_r -> top_r over full height)
         |
   bottom junction sphere
         |
   base disc        (heavy cylinder for tipping stability)

SDF sign convention: negative = inside solid, positive = outside.
Union = element-wise minimum over individual SDF fields.

Key parameters (all in mm)
--------------------------
base_radius_mm      : base disc radius            default  48
base_height_mm      : base disc thickness         default  14
stem_height_mm      : height where arm starts     default 234
stem_radius_base_mm : stem radius at base         default   9
stem_radius_top_mm  : stem radius at stem top     default   7
arm_reach_mm        : horizontal reach of tip     default  82
arm_tip_z_mm        : Z of the hook tip           default 244
arm_radius_mm       : arm beam radius             default 8.5
end_cap_radius_mm   : end-cap sphere radius       default   9
resolution_mm       : SDF grid pitch              default   1
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# SDF primitives (vectorised, pts.shape = (N,3))
# ---------------------------------------------------------------------------

def _sdf_sphere(pts: np.ndarray, cx: float, cy: float, cz: float,
                r: float) -> np.ndarray:
    """Exact SDF for a sphere centred at (cx, cy, cz) with radius r."""
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    dz = pts[:, 2] - cz
    return np.sqrt(dx * dx + dy * dy + dz * dz) - r


def _sdf_cylinder(pts: np.ndarray, cx: float, cy: float,
                  z_lo: float, z_hi: float, r: float) -> np.ndarray:
    """Exact SDF for an upright capped cylinder (axis-aligned to Z)."""
    d_r = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2) - r
    d_z = np.maximum(z_lo - pts[:, 2], pts[:, 2] - z_hi)
    # Standard "rounded box" style exterior + interior formula:
    return (np.sqrt(np.maximum(d_r, 0.0) ** 2 + np.maximum(d_z, 0.0) ** 2)
            + np.minimum(np.maximum(d_r, d_z), 0.0))


def _sdf_capsule(pts: np.ndarray,
                 ax: float, ay: float, az: float,
                 bx: float, by: float, bz: float,
                 ra: float, rb: float) -> np.ndarray:
    """Exact SDF for a tapered capsule (beam) from A to B.

    The cross-section radius lerps linearly from ra at A to rb at B.
    This is the standard tapered-capsule formula.
    """
    a = np.array([ax, ay, az], dtype=float)
    b = np.array([bx, by, bz], dtype=float)
    ab = b - a
    ab2 = float(np.dot(ab, ab))

    if ab2 < 1e-12:          # degenerate: A == B, fall back to sphere
        return _sdf_sphere(pts, ax, ay, az, ra)

    # Project onto segment, clamp t to [0,1]
    ap = pts - a                            # (N,3)
    t  = np.clip(ap @ ab / ab2, 0.0, 1.0)  # (N,)

    # Interpolated radius at each t
    r_t = ra + t * (rb - ra)               # (N,)

    closest = a + t[:, None] * ab          # (N,3)
    dist    = np.linalg.norm(pts - closest, axis=1)  # (N,)
    return dist - r_t


# ---------------------------------------------------------------------------
# Holder SDF
# ---------------------------------------------------------------------------

def _holder_sdf(
    pts: np.ndarray,
    base_r: float,
    base_h: float,
    stem_h: float,
    srb: float,
    srt: float,
    reach: float,
    tip_z: float,
    arm_r: float,
    ecr: float,
) -> np.ndarray:
    """Compute the union SDF of all holder components at the given points."""

    # 1. Base disc
    d = _sdf_cylinder(pts, 0.0, 0.0, 0.0, base_h, base_r)

    # 2. Bottom junction sphere (smooth transition base -> stem)
    d = np.minimum(d, _sdf_sphere(pts, 0.0, 0.0, base_h, srb + 2.0))

    # 3. Tapered stem
    d = np.minimum(d, _sdf_capsule(pts, 0.0, 0.0, base_h, 0.0, 0.0, stem_h, srb, srt))

    # 4. Top junction sphere (smooth transition stem -> arm)
    d = np.minimum(d, _sdf_sphere(pts, 0.0, 0.0, stem_h, srt + 1.5))

    # 5. Arm: 3-segment S-curve
    #    Proportions derived from the reference v2 design (reach=82, stem_h=234, tip_z=244)
    #    rise = tip_z - stem_h (+10 in reference)
    rise = tip_z - stem_h
    elbow   = np.array([-reach * 0.341, 0.0, stem_h + rise * 1.20])  # sweeps out + up
    plateau = np.array([-reach * 0.732, 0.0, stem_h + rise * 1.60])  # peak of arc
    tip     = np.array([-reach,         0.0, tip_z])

    # arm segment 1: stem top -> elbow (srt+2.5 -> arm_r for smooth junction)
    d = np.minimum(d, _sdf_capsule(pts, 0.0, 0.0, stem_h,
                                   elbow[0],   elbow[1],   elbow[2],
                                   srt + 2.5,  arm_r))
    # arm segment 2: elbow -> plateau (uniform radius)
    d = np.minimum(d, _sdf_capsule(pts, elbow[0],   elbow[1],   elbow[2],
                                        plateau[0], plateau[1], plateau[2],
                                        arm_r, arm_r))
    # arm segment 3: plateau -> tip (gently tapers)
    d = np.minimum(d, _sdf_capsule(pts, plateau[0], plateau[1], plateau[2],
                                        tip[0],     tip[1],     tip[2],
                                        arm_r, arm_r - 1.0))

    # 6. End cap sphere (distributes contact force, prevents slip)
    d = np.minimum(d, _sdf_sphere(pts, tip[0], tip[1], tip[2], ecr))

    return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_holder_stl(
    base_radius_mm:      float = 48.0,
    base_height_mm:      float = 14.0,
    stem_height_mm:      float = 234.0,
    stem_radius_base_mm: float = 9.0,
    stem_radius_top_mm:  float = 7.0,
    arm_reach_mm:        float = 82.0,
    arm_tip_z_mm:        float = 244.0,
    arm_radius_mm:       float = 8.5,
    end_cap_radius_mm:   float = 9.0,
    resolution_mm:       float = 1.0,
    out_stl:             Optional[str | Path] = None,
) -> dict:
    """Generate a parametric headphone-holder STL from design parameters.

    Builds the geometry via SDF union + marching cubes; no external CAD
    kernel required.  Returns a dict with path, volume, and arm-tip
    coordinates suitable for passing directly to run_topopt / check_physics.

    Parameters
    ----------
    base_radius_mm      : Radius of the base disc [mm].
    base_height_mm      : Thickness of the base disc [mm].
    stem_height_mm      : Height at which the arm leaves the stem [mm].
                          The stand's total height is roughly stem_height_mm + 20.
    stem_radius_base_mm : Stem radius at the base junction [mm].
    stem_radius_top_mm  : Stem radius at the top junction [mm] (taper).
    arm_reach_mm        : Horizontal reach of the hook tip [mm].
    arm_tip_z_mm        : Z-height of the hook tip [mm].
    arm_radius_mm       : Arm beam radius [mm].
    end_cap_radius_mm   : End-cap sphere radius [mm].
    resolution_mm       : SDF grid pitch [mm].  1 mm gives a clean mesh in
                          ~5 s; 0.5 mm is finer but takes ~40 s.
    out_stl             : Output STL path.  Default: docs/generated_holder.stl.

    Returns
    -------
    dict with keys:
        status            : "ok" | "error"
        stl_path          : absolute path to the generated STL
        volume_mm3        : solid volume
        arm_tip_x_mm      : X of hook tip (for run_topopt)
        arm_tip_y_mm      : Y of hook tip (always 0)
        arm_tip_z_mm      : Z of hook tip (for run_topopt)
        base_radius_mm    : base disc radius (for check_physics)
        elapsed_s         : wall-clock generation time
    """
    from skimage.measure import marching_cubes as _mc

    t0 = time.time()

    # ------------------------------------------------------------------
    # Bounding box (+ generous padding so MC boundary is always void)
    # ------------------------------------------------------------------
    pad = max(end_cap_radius_mm + 5.0, 15.0)
    x_min = -arm_reach_mm - pad
    x_max =  pad
    y_min = -pad
    y_max =  pad
    z_min = -pad
    z_max =  stem_height_mm + pad + 20.0   # arm peaks above stem_height

    res = float(resolution_mm)
    xs = np.arange(x_min, x_max + res, res)
    ys = np.arange(y_min, y_max + res, res)
    zs = np.arange(z_min, z_max + res, res)
    Nx, Ny, Nz = len(xs), len(ys), len(zs)

    print(
        f"Generating holder SDF at {res} mm: "
        f"{Nx} x {Ny} x {Nz} = {Nx*Ny*Nz:,} pts ..."
    )

    # ------------------------------------------------------------------
    # Evaluate SDF (flat then reshape)
    # ------------------------------------------------------------------
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    sdf = _holder_sdf(
        pts,
        base_r=base_radius_mm,
        base_h=base_height_mm,
        stem_h=stem_height_mm,
        srb=stem_radius_base_mm,
        srt=stem_radius_top_mm,
        reach=arm_reach_mm,
        tip_z=arm_tip_z_mm,
        arm_r=arm_radius_mm,
        ecr=end_cap_radius_mm,
    ).reshape(Nx, Ny, Nz)

    # ------------------------------------------------------------------
    # Marching cubes at iso = 0  (negative SDF = inside)
    # ------------------------------------------------------------------
    verts, faces, normals, _ = _mc(sdf, level=0.0, spacing=(res, res, res))
    # translate verts from grid-index space back to world (mm) coords
    verts += np.array([x_min, y_min, z_min])

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    if mesh.volume < 0:
        mesh.invert()

    volume_mm3 = float(mesh.volume)

    # ------------------------------------------------------------------
    # Export STL
    # ------------------------------------------------------------------
    if out_stl is None:
        from pathlib import Path as _P
        # Three levels up from this file -> project root
        _root = _P(__file__).resolve().parent.parent.parent.parent
        out_path = _root / "docs" / "generated_holder.stl"
    else:
        out_path = Path(out_stl)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))

    elapsed = time.time() - t0
    size_kb = out_path.stat().st_size // 1024
    print(f"  Generated STL -> {out_path}  ({size_kb} KB, {volume_mm3:.0f} mm3, {elapsed:.1f}s)")

    return {
        "status":         "ok",
        "stl_path":       str(out_path),
        "volume_mm3":     round(volume_mm3),
        "arm_tip_x_mm":  -arm_reach_mm,      # negative X (arm extends left)
        "arm_tip_y_mm":   0.0,
        "arm_tip_z_mm":   arm_tip_z_mm,
        "base_radius_mm": base_radius_mm,
        "elapsed_s":      round(elapsed, 1),
    }
