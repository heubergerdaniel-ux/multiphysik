"""shapek -- ShapeKernel-inspired geometry abstractions for the multiphysik pipeline.

Python re-implementation of LEAP71 ShapeKernel (C#) key patterns,
using the existing numpy / scipy / pycogk stack.

Public API
----------
LocalFrame            -- coordinate frames and transforms
LineModulation        -- parametric 1D variation over [0,1]
SurfaceModulation     -- parametric 2D variation over [0,1]^2
ControlPointSpline    -- cubic spline interpolation through control points

sdf_sphere, sdf_box, sdf_capsule, sdf_cylinder  -- consolidated base primitives
sdf_cone, sdf_torus, sdf_revolve, sdf_pipe      -- new extended primitives

BaseShape             -- abstract: sdf_at_points() + bounds() + mesh_stl() + voxelize()
CompoundShape         -- SDF union of multiple BaseShape children
SphereShape, BoxShape, CapsuleShape, CylinderShape  -- base primitive shapes
ConeShape, TorusShape, PipeShape, RevolveShape       -- extended shapes
build_compound_from_spec  -- factory from MCP JSON spec list

EngineeringLattice    -- beam-node lattice via picogk.Lattice (Phase C)

Measure               -- volume, CoG, moment of inertia from STL or Voxels
ShapeMeasure          -- result dataclass
"""
from picogk_mp.shapek.frame      import LocalFrame
from picogk_mp.shapek.modulation import (
    ControlPointSpline,
    LineModulation,
    SurfaceModulation,
)
from picogk_mp.shapek.primitives import (
    sdf_box,
    sdf_capsule,
    sdf_cone,
    sdf_cylinder,
    sdf_pipe,
    sdf_revolve,
    sdf_sphere,
    sdf_torus,
)
from picogk_mp.shapek.base_shape import (
    BaseShape,
    BoxShape,
    CapsuleShape,
    CompoundShape,
    ConeShape,
    CylinderShape,
    DifferenceShape,
    IntersectionShape,
    PipeShape,
    RevolveShape,
    SphereShape,
    TorusShape,
    build_compound_from_spec,
)
from picogk_mp.shapek.measure import Measure, ShapeMeasure

# Lazy import for lattice (requires picogk.go context at construction time,
# but import itself is safe).  Expose at module level for convenience.
try:
    from picogk_mp.shapek.lattice import EngineeringLattice
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Extend generators/shape.py _DISPATCH so the existing generate_shape MCP
# tool immediately gains cone and torus support without any other code change.
# ---------------------------------------------------------------------------
try:
    from picogk_mp.generators.shape import _DISPATCH as _SD  # type: ignore[attr-defined]
    from picogk_mp.shapek.primitives import sdf_cone as _sc, sdf_torus as _st

    if "cone" not in _SD:
        _SD["cone"] = lambda pts, p: _sc(pts, p["apex"], p["base"], p["r_base"])
    if "torus" not in _SD:
        _SD["torus"] = lambda pts, p: _st(pts, p["center"], p["major_r"], p["minor_r"])
except Exception:
    # generators may not be on path in test environments; not fatal.
    pass


__all__ = [
    "LocalFrame",
    "ControlPointSpline",
    "LineModulation",
    "SurfaceModulation",
    "sdf_sphere",
    "sdf_box",
    "sdf_capsule",
    "sdf_cylinder",
    "sdf_cone",
    "sdf_torus",
    "sdf_revolve",
    "sdf_pipe",
    "BaseShape",
    "CompoundShape",
    "DifferenceShape",
    "IntersectionShape",
    "SphereShape",
    "BoxShape",
    "CapsuleShape",
    "CylinderShape",
    "ConeShape",
    "TorusShape",
    "PipeShape",
    "RevolveShape",
    "build_compound_from_spec",
    "Measure",
    "ShapeMeasure",
]
