"""EngineeringLattice -- beam-node lattice using picogk.Lattice as backend.

Mirrors ShapeKernel's Lattice class patterns.

Nodes  = spheres (junction points)
Beams  = tapered capsules (structural members)

Requires an active picogk.go context for voxelization.
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

Vec3 = Sequence[float]


@dataclass
class _Node:
    position: np.ndarray
    radius:   float


@dataclass
class _Beam:
    a:   np.ndarray
    b:   np.ndarray
    r_a: float
    r_b: float


class EngineeringLattice:
    """Fluent builder for beam-node lattice structures.

    Uses picogk.Lattice natively (C++, sub-second for typical structures).

    Example
    -------
    lat = (EngineeringLattice()
           .add_node([0,0,0], 3.0)
           .add_node([0,0,30], 3.0)
           .add_beam([0,0,0], [0,0,30], 2.0))
    vox = lat.voxelize()   # requires picogk.go context
    """

    def __init__(self) -> None:
        self._nodes: list[_Node] = []
        self._beams: list[_Beam] = []

    # --- Builder API --------------------------------------------------

    def add_node(self, position: Vec3, radius: float) -> "EngineeringLattice":
        """Add a sphere node."""
        self._nodes.append(_Node(np.asarray(position, float), float(radius)))
        return self

    def add_beam(
        self,
        a:   Vec3,
        b:   Vec3,
        r_a: float,
        r_b: float | None = None,
    ) -> "EngineeringLattice":
        """Add a tapered capsule beam from a (radius r_a) to b (radius r_b).

        r_b defaults to r_a (uniform cylinder).
        """
        r_b = float(r_a) if r_b is None else float(r_b)
        self._beams.append(_Beam(
            np.asarray(a, float), np.asarray(b, float), float(r_a), r_b
        ))
        return self

    def add_strut(
        self,
        a:          Vec3,
        b:          Vec3,
        radius_mod,
        n_segments: int = 1,
    ) -> "EngineeringLattice":
        """Modulated-radius strut (LineModulation).

        n_segments=1 -> single tapered capsule.
        n_segments>1 -> polyline of capsules approximating modulation curve.
        """
        a_arr = np.asarray(a, float)
        b_arr = np.asarray(b, float)
        ts    = np.linspace(0.0, 1.0, n_segments + 1)
        radii = radius_mod.at_array(ts)
        pts   = a_arr + ts[:, None] * (b_arr - a_arr)
        for i in range(n_segments):
            self.add_beam(pts[i], pts[i + 1], radii[i], radii[i + 1])
        return self

    # --- Factory methods ----------------------------------------------

    @classmethod
    def cubic_unit_cell(
        cls,
        origin:       Vec3,
        cell_size_mm: float,
        strut_radius: float,
    ) -> "EngineeringLattice":
        """Build a single cubic unit cell (12 edges + 8 corner nodes)."""
        lat = cls()
        o = np.asarray(origin, float)
        s = float(cell_size_mm)
        corners = np.array([
            [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
            [0, 0, s], [s, 0, s], [s, s, s], [0, s, s],
        ], dtype=float) + o
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for c in corners:
            lat.add_node(c, strut_radius)
        for i, j in edges:
            lat.add_beam(corners[i], corners[j], strut_radius)
        return lat

    def fill_box(
        self,
        min_xyz:      Vec3,
        max_xyz:      Vec3,
        cell_size_mm: float,
        strut_radius: float,
    ) -> "EngineeringLattice":
        """Fill a box with cubic unit cells (in-place merge)."""
        mn  = np.asarray(min_xyz, float)
        mx  = np.asarray(max_xyz, float)
        s   = float(cell_size_mm)
        nx  = max(1, int(np.ceil((mx[0] - mn[0]) / s)))
        ny  = max(1, int(np.ceil((mx[1] - mn[1]) / s)))
        nz  = max(1, int(np.ceil((mx[2] - mn[2]) / s)))
        cell = EngineeringLattice.cubic_unit_cell([0, 0, 0], s, strut_radius)
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    off = mn + np.array([ix, iy, iz]) * s
                    self.merge(cell.translate_and_merge(off))
        return self

    def translate_and_merge(self, offset: Vec3) -> "EngineeringLattice":
        """Return a new lattice that is a copy of self shifted by offset."""
        off = np.asarray(offset, float)
        new = EngineeringLattice()
        for n in self._nodes:
            new.add_node(n.position + off, n.radius)
        for b in self._beams:
            new.add_beam(b.a + off, b.b + off, b.r_a, b.r_b)
        return new

    def merge(self, other: "EngineeringLattice") -> "EngineeringLattice":
        """In-place: add all nodes and beams from other."""
        self._nodes.extend(other._nodes)
        self._beams.extend(other._beams)
        return self

    # --- Output -------------------------------------------------------

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def beam_count(self) -> int:
        return len(self._beams)

    def to_picogk_lattice(self):
        """Build and return a raw picogk.Lattice object.

        Requires picogk imported but NOT necessarily a .go() context.
        """
        from picogk import Lattice  # type: ignore[import]

        lat = Lattice()
        for n in self._nodes:
            lat.add_sphere(n.position.tolist(), n.radius)
        for b in self._beams:
            lat.add_beam(b.a.tolist(), b.b.tolist(), b.r_a, b.r_b)
        return lat

    def voxelize(self):
        """Voxelize to picogk.Voxels (requires active picogk.go context)."""
        from picogk import Voxels  # type: ignore[import]

        return Voxels.from_lattice(self.to_picogk_lattice())

    def mesh_stl(self, out_stl: str | Path) -> dict:
        """Voxelize then export to STL (requires active picogk.go context)."""
        from picogk import Mesh  # type: ignore[import]

        t0 = time.time()
        vox  = self.voxelize()
        mesh = Mesh.from_voxels(vox)
        out  = Path(out_stl)
        out.parent.mkdir(parents=True, exist_ok=True)
        mesh.SaveToStlFile(str(out))
        vol, _ = vox.calculate_properties()
        elapsed = time.time() - t0
        size_kb = out.stat().st_size // 1024
        print(
            f"EngineeringLattice -> {out}  ({size_kb} KB, {vol:.0f} mm3, "
            f"{self.node_count} nodes, {self.beam_count} beams, {elapsed:.1f}s)"
        )
        return {
            "status":     "ok",
            "stl_path":   str(out),
            "volume_mm3": round(float(vol)),
            "node_count": self.node_count,
            "beam_count": self.beam_count,
            "elapsed_s":  round(elapsed, 1),
        }
