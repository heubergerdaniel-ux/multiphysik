"""Topology optimisation module for PicoGK Multiphysik pipeline.

BESO (Bi-directional Evolutionary Structural Optimization) on regular
voxel grids.  One voxel = one Hex8 FEM element.

Public API
----------
TopoptPipeline          -- end-to-end STL-in / STL-out optimisation runner
BoundaryConditions      -- container for fixed DOFs + force vector
BoundaryConditions.headphone_holder  -- convenience constructor for stands
element_stiffness       -- 24x24 Hex8 stiffness matrix (for unit tests)
"""
from picogk_mp.topopt.pipeline   import TopoptPipeline
from picogk_mp.topopt.boundary   import BoundaryConditions
from picogk_mp.topopt.fem        import element_stiffness

__all__ = [
    "TopoptPipeline",
    "BoundaryConditions",
    "element_stiffness",
]
