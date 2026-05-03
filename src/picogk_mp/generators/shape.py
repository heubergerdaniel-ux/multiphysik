"""General SDF-based shape generator.

Claude specifies a list of geometric primitives in JSON; this module
unions them into a single watertight mesh via marching cubes.

Primitive types
---------------
sphere   : {"type":"sphere",   "center":[x,y,z], "radius":r}
box      : {"type":"box",      "min":[x0,y0,z0], "max":[x1,y1,z1]}
capsule  : {"type":"capsule",  "from":[x,y,z], "to":[x,y,z],
                               "radius_from":r0, "radius_to":r1}
           Set radius_from == radius_to for a uniform capsule (rod/beam).
cylinder : {"type":"cylinder", "center_xy":[cx,cy], "z_range":[z0,z1],
                               "radius":r}
           Always upright (z-axis aligned).

SDF sign convention: negative = inside solid, positive = outside.
Union = element-wise minimum.

Design rules
------------
- All coordinates in millimetres.
- Smooth blending between adjacent primitives is automatic (SDF minimum
  already produces a C1-continuous surface at intersections).
- For organic, rounded joints add an overlapping sphere at each junction.
- resolution_mm=1 gives clean meshes in 1-5 s for parts up to ~300mm.
  Use 0.5 for fine detail (~10-40 s), 2 for quick previews (~0.1 s).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Vectorised SDF primitives  (pts: (N,3) float64 array)
# ---------------------------------------------------------------------------

def _sphere(pts: np.ndarray, center, radius: float) -> np.ndarray:
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    dz = pts[:, 2] - cz
    return np.sqrt(dx*dx + dy*dy + dz*dz) - float(radius)


def _box(pts: np.ndarray, mn, mx) -> np.ndarray:
    cx = (float(mn[0]) + float(mx[0])) / 2
    cy = (float(mn[1]) + float(mx[1])) / 2
    cz = (float(mn[2]) + float(mx[2])) / 2
    hx = (float(mx[0]) - float(mn[0])) / 2
    hy = (float(mx[1]) - float(mn[1])) / 2
    hz = (float(mx[2]) - float(mn[2])) / 2

    qx = np.abs(pts[:, 0] - cx) - hx
    qy = np.abs(pts[:, 1] - cy) - hy
    qz = np.abs(pts[:, 2] - cz) - hz

    outer = np.sqrt(np.maximum(qx, 0)**2 + np.maximum(qy, 0)**2 + np.maximum(qz, 0)**2)
    inner = np.minimum(np.maximum(np.maximum(qx, qy), qz), 0.0)
    return outer + inner


def _capsule(pts: np.ndarray, a, b, ra: float, rb: float) -> np.ndarray:
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    bx, by, bz = float(b[0]), float(b[1]), float(b[2])
    ra, rb = float(ra), float(rb)

    ab  = np.array([bx-ax, by-ay, bz-az])
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-12:
        return _sphere(pts, a, ra)

    ap = pts - np.array([ax, ay, az])
    t  = np.clip(ap @ ab / ab2, 0.0, 1.0)
    r_t = ra + t * (rb - ra)
    closest = np.array([ax, ay, az]) + t[:, None] * ab
    return np.linalg.norm(pts - closest, axis=1) - r_t


def _cylinder(pts: np.ndarray, center_xy, z_range, radius: float) -> np.ndarray:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    z0, z1 = float(z_range[0]), float(z_range[1])
    r = float(radius)

    d_r = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2) - r
    d_z = np.maximum(z0 - pts[:, 2], pts[:, 2] - z1)

    outer = np.sqrt(np.maximum(d_r, 0)**2 + np.maximum(d_z, 0)**2)
    inner = np.minimum(np.maximum(d_r, d_z), 0.0)
    return outer + inner


_DISPATCH = {
    "sphere":   lambda pts, p: _sphere(pts,
                    p["center"], p["radius"]),
    "box":      lambda pts, p: _box(pts,
                    p["min"], p["max"]),
    "capsule":  lambda pts, p: _capsule(pts,
                    p["from"], p["to"],
                    p.get("radius_from", p.get("radius", 5.0)),
                    p.get("radius_to",   p.get("radius", 5.0))),
    "cylinder": lambda pts, p: _cylinder(pts,
                    p["center_xy"], p["z_range"], p["radius"]),
}


# ---------------------------------------------------------------------------
# Bounding box helper
# ---------------------------------------------------------------------------

def _primitive_bounds(p: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return (min_xyz, max_xyz) bounding box for a single primitive."""
    t = p["type"].lower()
    if t == "sphere":
        c = np.array(p["center"], dtype=float)
        r = float(p["radius"])
        return c - r, c + r
    if t == "box":
        return np.array(p["min"], dtype=float), np.array(p["max"], dtype=float)
    if t == "capsule":
        a = np.array(p["from"], dtype=float)
        b = np.array(p["to"],   dtype=float)
        ra = float(p.get("radius_from", p.get("radius", 5.0)))
        rb = float(p.get("radius_to",   p.get("radius", 5.0)))
        r = max(ra, rb)
        return np.minimum(a, b) - r, np.maximum(a, b) + r
    if t == "cylinder":
        cx, cy = p["center_xy"]
        z0, z1 = p["z_range"]
        r = float(p["radius"])
        return np.array([cx-r, cy-r, z0]), np.array([cx+r, cy+r, z1])
    raise ValueError(f"Unknown primitive type: '{t}'")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_shape_stl(
    primitives:    list[dict[str, Any]],
    resolution_mm: float = 1.0,
    out_stl:       Optional[str | Path] = None,
) -> dict:
    """Generate an arbitrary watertight STL from a list of SDF primitives.

    Parameters
    ----------
    primitives    : List of primitive dicts (see module docstring).
    resolution_mm : SDF grid pitch [mm].  1 = good quality in 1-5 s.
                    Use 2 for fast preview, 0.5 for fine detail.
    out_stl       : Output path.  Default: docs/generated_shape.stl.

    Returns
    -------
    dict with keys: status, stl_path, volume_mm3, bounds_min, bounds_max,
    elapsed_s.  bounds_* are [x,y,z] lists of the mesh bounding box.
    """
    from skimage.measure import marching_cubes as _mc

    if not primitives:
        return {"status": "error", "message": "No primitives provided."}

    # Validate types
    for i, p in enumerate(primitives):
        if "type" not in p:
            return {"status": "error", "message": f"Primitive {i} missing 'type' key."}
        if p["type"].lower() not in _DISPATCH:
            return {"status": "error",
                    "message": f"Unknown type '{p['type']}'. Use: {list(_DISPATCH)}"}

    t0 = time.time()

    # Union bounding box (+ padding so MC boundary is always void)
    pad = 10.0
    mn = np.array([np.inf,  np.inf,  np.inf])
    mx = np.array([-np.inf, -np.inf, -np.inf])
    for p in primitives:
        lo, hi = _primitive_bounds(p)
        mn = np.minimum(mn, lo)
        mx = np.maximum(mx, hi)
    mn -= pad
    mx += pad

    res = float(resolution_mm)
    xs = np.arange(mn[0], mx[0] + res, res)
    ys = np.arange(mn[1], mx[1] + res, res)
    zs = np.arange(mn[2], mx[2] + res, res)
    Nx, Ny, Nz = len(xs), len(ys), len(zs)

    print(
        f"Generating shape SDF at {res} mm: "
        f"{Nx} x {Ny} x {Nz} = {Nx*Ny*Nz:,} pts, "
        f"{len(primitives)} primitive(s) ..."
    )

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # Union of all primitives
    sdf = np.full(len(pts), np.inf)
    for p in primitives:
        fn = _DISPATCH[p["type"].lower()]
        sdf = np.minimum(sdf, fn(pts, p))

    sdf = sdf.reshape(Nx, Ny, Nz)

    # Marching cubes
    verts, faces, normals, _ = _mc(sdf, level=0.0, spacing=(res, res, res))
    verts += mn                            # translate to world coords

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    if mesh.volume < 0:
        mesh.invert()

    volume_mm3 = float(mesh.volume)
    b_mn = mesh.bounds[0].tolist()
    b_mx = mesh.bounds[1].tolist()

    # Output path
    if out_stl is None:
        _root = Path(__file__).resolve().parent.parent.parent.parent
        out_path = _root / "docs" / "generated_shape.stl"
    else:
        out_path = Path(out_stl)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))

    elapsed = time.time() - t0
    size_kb = out_path.stat().st_size // 1024
    print(f"  Generated STL -> {out_path}  ({size_kb} KB, {volume_mm3:.0f} mm3, {elapsed:.1f}s)")

    return {
        "status":     "ok",
        "stl_path":   str(out_path),
        "volume_mm3": round(volume_mm3),
        "bounds_min": [round(v, 1) for v in b_mn],
        "bounds_max": [round(v, 1) for v in b_mx],
        "elapsed_s":  round(elapsed, 1),
    }
