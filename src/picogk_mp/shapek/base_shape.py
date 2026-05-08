"""BaseShape abstraction -- unified SDF evaluation + voxelization interface."""
from __future__ import annotations

import os
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import trimesh


class BaseShape(ABC):
    """Abstract base for all shapek shapes.

    Subclasses implement sdf_at_points() and bounds().  This class provides:
      mesh_stl()  -- SDF -> marching cubes -> STL  (Path A, pure numpy)
      voxelize()  -- STL -> picogk Voxels          (Path B, requires picogk.go)

    The voxelize() result is a picogk.Voxels object compatible with
    csg.union / csg.difference / csg.intersection.
    """

    # --- Abstract interface --------------------------------------------

    @abstractmethod
    def sdf_at_points(self, pts: np.ndarray) -> np.ndarray:
        """Evaluate SDF at (N,3) world points. Returns (N,) signed distances."""
        ...

    @abstractmethod
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (min_xyz, max_xyz) bounding box as (3,) float64 arrays.

        Must be a conservative outer bound (all solid material inside).
        """
        ...

    # --- Concrete methods ----------------------------------------------

    def mesh_stl(
        self,
        resolution_mm: float = 1.0,
        out_stl: Optional[str | Path] = None,
    ) -> dict:
        """Evaluate SDF on a grid and extract surface with marching cubes.

        Returns a dict matching the generate_shape_stl() format:
        {status, stl_path, volume_mm3, bounds_min, bounds_max, elapsed_s}
        """
        from skimage.measure import marching_cubes as _mc

        t0  = time.time()
        mn, mx = self.bounds()
        pad = 5.0
        mn = mn - pad
        mx = mx + pad
        res = float(resolution_mm)

        xs = np.arange(mn[0], mx[0] + res, res)
        ys = np.arange(mn[1], mx[1] + res, res)
        zs = np.arange(mn[2], mx[2] + res, res)
        Nx, Ny, Nz = len(xs), len(ys), len(zs)

        print(
            f"mesh_stl {type(self).__name__} at {res} mm: "
            f"{Nx}x{Ny}x{Nz}={Nx*Ny*Nz:,} pts ..."
        )

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        sdf = self.sdf_at_points(pts).reshape(Nx, Ny, Nz)

        verts, faces, normals, _ = _mc(sdf, level=0.0, spacing=(res, res, res))
        verts += mn

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        if mesh.volume < 0:
            mesh.invert()

        if out_stl is None:
            fd, out_stl = tempfile.mkstemp(suffix=".stl")
            os.close(fd)
        out_path = Path(out_stl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(out_path))

        elapsed = time.time() - t0
        size_kb = out_path.stat().st_size // 1024
        print(f"  -> {out_path}  ({size_kb} KB, {mesh.volume:.0f} mm3, {elapsed:.1f}s)")

        return {
            "status":     "ok",
            "stl_path":   str(out_path),
            "volume_mm3": round(float(mesh.volume)),
            "bounds_min": [round(v, 1) for v in mesh.bounds[0].tolist()],
            "bounds_max": [round(v, 1) for v in mesh.bounds[1].tolist()],
            "elapsed_s":  round(elapsed, 1),
        }

    def voxelize(self, voxel_size_mm: float = 0.5):
        """Convert to picogk.Voxels (requires active picogk.go context).

        Uses the same STL round-trip pattern as csg.cylinder_voxels().
        Safe: does NOT call Voxels.offset() on large geometries.
        Returns a picogk.Voxels object usable with csg.union() etc.
        """
        from picogk import Mesh, Voxels  # type: ignore[import]

        fd, tmp = tempfile.mkstemp(suffix=".stl")
        os.close(fd)
        try:
            self.mesh_stl(resolution_mm=voxel_size_mm, out_stl=tmp)
            pmesh = Mesh.mshFromStlFile(tmp)
        finally:
            os.unlink(tmp)
        return Voxels.from_mesh(pmesh)


# ---------------------------------------------------------------------------
# Boolean composition shapes
# ---------------------------------------------------------------------------

class DifferenceShape(BaseShape):
    """SDF difference: A minus B.  sdf = max(sdf_A, -sdf_B)."""

    def __init__(self, a: BaseShape, b: BaseShape) -> None:
        self._a = a
        self._b = b

    def sdf_at_points(self, pts: np.ndarray) -> np.ndarray:
        return np.maximum(self._a.sdf_at_points(pts), -self._b.sdf_at_points(pts))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._a.bounds()


class IntersectionShape(BaseShape):
    """SDF intersection: A intersect B.  sdf = max(sdf_A, sdf_B)."""

    def __init__(self, a: BaseShape, b: BaseShape) -> None:
        self._a = a
        self._b = b

    def sdf_at_points(self, pts: np.ndarray) -> np.ndarray:
        return np.maximum(self._a.sdf_at_points(pts), self._b.sdf_at_points(pts))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        mn_a, mx_a = self._a.bounds()
        mn_b, mx_b = self._b.bounds()
        return np.maximum(mn_a, mn_b), np.minimum(mx_a, mx_b)


# ---------------------------------------------------------------------------
# CompoundShape -- SDF union of multiple BaseShape instances
# ---------------------------------------------------------------------------

class CompoundShape(BaseShape):
    """SDF union (element-wise minimum) of multiple BaseShape children."""

    def __init__(self, *children: BaseShape) -> None:
        self._children: list[BaseShape] = list(children)

    def add(self, shape: BaseShape) -> "CompoundShape":
        self._children.append(shape)
        return self

    def sdf_at_points(self, pts: np.ndarray) -> np.ndarray:
        result = np.full(len(pts), np.inf)
        for child in self._children:
            result = np.minimum(result, child.sdf_at_points(pts))
        return result

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        all_mn = np.array([c.bounds()[0] for c in self._children])
        all_mx = np.array([c.bounds()[1] for c in self._children])
        return all_mn.min(axis=0), all_mx.max(axis=0)


# ---------------------------------------------------------------------------
# Concrete shape classes
# ---------------------------------------------------------------------------

class SphereShape(BaseShape):
    def __init__(self, center: Sequence[float], radius: float) -> None:
        self._c = np.asarray(center, dtype=float)
        self._r = float(radius)

    def sdf_at_points(self, pts):
        from picogk_mp.shapek.primitives import sdf_sphere
        return sdf_sphere(pts, self._c, self._r)

    def bounds(self):
        return self._c - self._r - 1, self._c + self._r + 1


class BoxShape(BaseShape):
    def __init__(self, mn: Sequence[float], mx: Sequence[float]) -> None:
        self._mn = np.asarray(mn, dtype=float)
        self._mx = np.asarray(mx, dtype=float)

    def sdf_at_points(self, pts):
        from picogk_mp.shapek.primitives import sdf_box
        return sdf_box(pts, self._mn, self._mx)

    def bounds(self):
        return self._mn - 1, self._mx + 1


class CapsuleShape(BaseShape):
    def __init__(
        self,
        a: Sequence[float], b: Sequence[float],
        ra: float, rb: float,
    ) -> None:
        self._a  = np.asarray(a, dtype=float)
        self._b  = np.asarray(b, dtype=float)
        self._ra = float(ra)
        self._rb = float(rb)

    def sdf_at_points(self, pts):
        from picogk_mp.shapek.primitives import sdf_capsule
        return sdf_capsule(pts, self._a, self._b, self._ra, self._rb)

    def bounds(self):
        r = max(self._ra, self._rb) + 1
        return np.minimum(self._a, self._b) - r, np.maximum(self._a, self._b) + r


class CylinderShape(BaseShape):
    def __init__(
        self,
        center_xy: Sequence[float],
        z_range:   Sequence[float],
        radius:    float,
    ) -> None:
        self._cxy = (float(center_xy[0]), float(center_xy[1]))
        self._z   = (float(z_range[0]), float(z_range[1]))
        self._r   = float(radius)

    def sdf_at_points(self, pts):
        from picogk_mp.shapek.primitives import sdf_cylinder
        return sdf_cylinder(pts, self._cxy, self._z, self._r)

    def bounds(self):
        r = self._r + 1
        return (
            np.array([self._cxy[0]-r, self._cxy[1]-r, self._z[0]-1]),
            np.array([self._cxy[0]+r, self._cxy[1]+r, self._z[1]+1]),
        )


class ConeShape(BaseShape):
    def __init__(
        self,
        apex:   Sequence[float],
        base:   Sequence[float],
        r_base: float,
    ) -> None:
        self._apex   = np.asarray(apex, dtype=float)
        self._base   = np.asarray(base, dtype=float)
        self._r_base = float(r_base)

    def sdf_at_points(self, pts):
        from picogk_mp.shapek.primitives import sdf_cone
        return sdf_cone(pts, self._apex, self._base, self._r_base)

    def bounds(self):
        pad = self._r_base + 1
        return (
            np.minimum(self._apex, self._base) - pad,
            np.maximum(self._apex, self._base) + pad,
        )


class TorusShape(BaseShape):
    def __init__(
        self,
        center:  Sequence[float],
        major_r: float,
        minor_r: float,
        frame=None,
    ) -> None:
        from picogk_mp.shapek.frame import LocalFrame
        self._center  = np.asarray(center, dtype=float)
        self._major_r = float(major_r)
        self._minor_r = float(minor_r)
        self._frame   = frame or LocalFrame(
            origin=self._center, tangent=(0, 0, 1), normal=(1, 0, 0)
        )

    def sdf_at_points(self, pts):
        from picogk_mp.shapek.primitives import sdf_torus
        return sdf_torus(pts, self._center, self._major_r, self._minor_r, self._frame)

    def bounds(self):
        extent = self._major_r + self._minor_r + 1
        return self._center - extent, self._center + extent


class PipeShape(BaseShape):
    def __init__(
        self,
        spine:      np.ndarray,
        radius_mod,
    ) -> None:
        from picogk_mp.shapek.modulation import LineModulation
        self._spine  = np.asarray(spine, dtype=float)
        self._rmod   = radius_mod

    def sdf_at_points(self, pts):
        from picogk_mp.shapek.primitives import sdf_pipe
        return sdf_pipe(pts, self._spine, self._rmod)

    def bounds(self):
        ts   = np.linspace(0, 1, len(self._spine))
        radii = self._rmod.at_array(ts)
        r_max = float(radii.max()) + 1
        return self._spine.min(axis=0) - r_max, self._spine.max(axis=0) + r_max


class RevolveShape(BaseShape):
    def __init__(
        self,
        profile_fn,
        axis_origin: Sequence[float] = (0.0, 0.0, 0.0),
        axis_dir:    Sequence[float] = (0.0, 0.0, 1.0),
        bounds_hint: float = 100.0,
    ) -> None:
        self._profile    = profile_fn
        self._axis_o     = np.asarray(axis_origin, dtype=float)
        self._axis_d     = np.asarray(axis_dir,    dtype=float)
        self._bounds_hint = float(bounds_hint)

    def sdf_at_points(self, pts):
        from picogk_mp.shapek.primitives import sdf_revolve
        return sdf_revolve(pts, self._profile, self._axis_o, self._axis_d)

    def bounds(self):
        h = self._bounds_hint
        return self._axis_o - h, self._axis_o + h


# ---------------------------------------------------------------------------
# Factory: build CompoundShape from MCP spec dicts
# ---------------------------------------------------------------------------

def build_compound_from_spec(shapes: list[dict]) -> CompoundShape:
    """Build a CompoundShape from a list of primitive spec dicts.

    Supports all generate_shape types plus: cone, torus, pipe.
    """
    from picogk_mp.shapek.modulation import LineModulation

    compound = CompoundShape()
    for p in shapes:
        t = p.get("type", "").lower()
        if t == "sphere":
            compound.add(SphereShape(p["center"], p["radius"]))
        elif t == "box":
            compound.add(BoxShape(p["min"], p["max"]))
        elif t == "capsule":
            compound.add(CapsuleShape(
                p["from"], p["to"],
                p.get("radius_from", p.get("radius", 5.0)),
                p.get("radius_to",   p.get("radius", 5.0)),
            ))
        elif t == "cylinder":
            compound.add(CylinderShape(p["center_xy"], p["z_range"], p["radius"]))
        elif t == "cone":
            compound.add(ConeShape(p["apex"], p["base"], p["r_base"]))
        elif t == "torus":
            compound.add(TorusShape(p["center"], p["major_r"], p["minor_r"]))
        elif t == "pipe":
            spine = np.asarray(p["spine"], dtype=float)
            if "radii" in p:
                radii = p["radii"]
                ts    = np.linspace(0.0, 1.0, len(radii))
                rmod  = LineModulation.from_control_points(list(zip(ts, radii)))
            else:
                rmod = LineModulation.constant(float(p.get("radius", 5.0)))
            compound.add(PipeShape(spine, rmod))
        else:
            raise ValueError(f"Unknown primitive type: '{t}'")
    return compound
