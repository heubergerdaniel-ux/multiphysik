"""Geometry generators -- produce STLs from design parameters or primitives.

All generators work headlessly (pure numpy / scikit-image / trimesh),
so they run safely inside the MCP server without picogk.go.
"""
from picogk_mp.generators.shape  import generate_shape_stl

__all__ = ["generate_shape_stl"]
