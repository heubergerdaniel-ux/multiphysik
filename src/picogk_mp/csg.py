"""CSG helpers: box primitive and non-destructive boolean wrappers.

All pycogk boolean ops (bool_add, bool_subtract, bool_intersect) are in-place:
they mutate 'self' and return 'self'.  The wrappers here always duplicate 'a'
first so callers can reuse their Voxels objects safely.
"""
from __future__ import annotations

import os
import tempfile
from typing import Sequence

import trimesh
import trimesh.creation

from picogk import Mesh, Voxels

Vector3 = Sequence[float]


def box_voxels(center: Vector3, size: Vector3) -> Voxels:
    """Voxelise an axis-aligned box.

    Routes through trimesh -> tmp STL -> picogk Mesh -> Voxels so we
    don't have to write a hand-rolled implicit or vertex loop.
    """
    tm: trimesh.Trimesh = trimesh.creation.box(extents=size)
    tm.apply_translation(center)
    fd, tmp = tempfile.mkstemp(suffix=".stl")
    os.close(fd)
    try:
        tm.export(tmp)
        pmesh = Mesh.mshFromStlFile(tmp)
    finally:
        os.unlink(tmp)
    return Voxels.from_mesh(pmesh)


def union(a: Voxels, b: Voxels) -> Voxels:
    return a.duplicate().bool_add(b)


def difference(a: Voxels, b: Voxels) -> Voxels:
    return a.duplicate().bool_subtract(b)


def intersection(a: Voxels, b: Voxels) -> Voxels:
    return a.duplicate().bool_intersect(b)


def smooth_union(a: Voxels, b: Voxels, blend_mm: float = 2.0) -> Voxels:
    # Voxels_BoolAddSmooth is absent in pycogk 0.3.0 runtime; falls back to plain union.
    try:
        return a.duplicate().bool_add_smooth(b, blend_mm)
    except NotImplementedError:
        return a.duplicate().bool_add(b)
