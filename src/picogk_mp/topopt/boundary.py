"""Boundary condition helpers for regular voxel-grid FEM meshes.

All functions operate in physical millimetres.  The voxel grid spans
x in [0, Nx*h],  y in [0, Ny*h],  z in [0, Nz*h]  with origin at the
grid corner.  The *offset* vector shifts this origin to align with the
physical model coordinate system.

Node global index:  n = ix + iy*(Nx+1) + iz*(Nx+1)*(Ny+1)
DOF index for node n:  [3n, 3n+1, 3n+2]  = [u_x, u_y, u_z]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Mesh geometry helpers
# ---------------------------------------------------------------------------

def node_index(ix: int, iy: int, iz: int, Nx: int, Ny: int) -> int:
    """Global node index for grid node (ix, iy, iz)."""
    return int(ix + iy * (Nx + 1) + iz * (Nx + 1) * (Ny + 1))


def dof_of_node(n: int, axis: int) -> int:
    """DOF index: axis=0->ux, 1->uy, 2->uz."""
    return 3 * n + axis


# ---------------------------------------------------------------------------
# Fixed-DOF generators
# ---------------------------------------------------------------------------

def fixed_face_dofs(
    Nx: int, Ny: int, Nz: int,
    face: str,
    axes: Sequence[int] = (0, 1, 2),
) -> np.ndarray:
    """Return DOF indices for all nodes on a named face of the grid.

    Parameters
    ----------
    face : one of "x0","x1","y0","y1","z0","z1"
           where 0/1 is the low/high end of that axis.
    axes : which DOFs to fix (default: all three = fully clamped).
    """
    face_nodes = []
    if face == "z0":
        for ix in range(Nx + 1):
            for iy in range(Ny + 1):
                face_nodes.append(node_index(ix, iy, 0, Nx, Ny))
    elif face == "z1":
        for ix in range(Nx + 1):
            for iy in range(Ny + 1):
                face_nodes.append(node_index(ix, iy, Nz, Nx, Ny))
    elif face == "x0":
        for iy in range(Ny + 1):
            for iz in range(Nz + 1):
                face_nodes.append(node_index(0, iy, iz, Nx, Ny))
    elif face == "x1":
        for iy in range(Ny + 1):
            for iz in range(Nz + 1):
                face_nodes.append(node_index(Nx, iy, iz, Nx, Ny))
    elif face == "y0":
        for ix in range(Nx + 1):
            for iz in range(Nz + 1):
                face_nodes.append(node_index(ix, 0, iz, Nx, Ny))
    elif face == "y1":
        for ix in range(Nx + 1):
            for iz in range(Nz + 1):
                face_nodes.append(node_index(ix, Ny, iz, Nx, Ny))
    else:
        raise ValueError(f"Unknown face '{face}'. Use x0/x1/y0/y1/z0/z1.")

    dofs = []
    for n in face_nodes:
        for ax in axes:
            dofs.append(dof_of_node(n, ax))
    return np.unique(dofs).astype(np.int64)


def fixed_cylinder_base_dofs(
    Nx: int, Ny: int, Nz: int, h: float,
    offset: Sequence[float],
    radius: float,
    axes: Sequence[int] = (0, 1, 2),
) -> np.ndarray:
    """Fix all z=0 nodes within *radius* of the grid axis.

    More accurate than fixing the whole z0 face for a cylindrical base.

    Parameters
    ----------
    offset : (ox, oy, oz) physical origin of the voxel grid [mm]
    radius : base disc radius [mm]
    """
    ox, oy, _ = offset
    dofs = []
    for ix in range(Nx + 1):
        for iy in range(Ny + 1):
            x_phys = ox + ix * h
            y_phys = oy + iy * h
            if x_phys**2 + y_phys**2 <= radius**2:      # within disc
                n = node_index(ix, iy, 0, Nx, Ny)
                for ax in axes:
                    dofs.append(dof_of_node(n, ax))
    return np.unique(dofs).astype(np.int64)


# ---------------------------------------------------------------------------
# Force vector builders
# ---------------------------------------------------------------------------

def point_load_dof(
    Nx: int, Ny: int, Nz: int, h: float,
    offset: Sequence[float],
    position_mm: Sequence[float],
    force_N: Sequence[float],
) -> np.ndarray:
    """Force vector for a single point load at the nearest grid node.

    Parameters
    ----------
    offset      : (ox, oy, oz) physical origin of the voxel grid [mm]
    position_mm : (x, y, z) physical position of the load [mm]
    force_N     : (Fx, Fy, Fz) force components [N]
    """
    ox, oy, oz = offset
    px, py, pz = position_mm

    # Nearest node indices
    ix = int(np.clip(round((px - ox) / h), 0, Nx))
    iy = int(np.clip(round((py - oy) / h), 0, Ny))
    iz = int(np.clip(round((pz - oz) / h), 0, Nz))

    ndof = 3 * (Nx + 1) * (Ny + 1) * (Nz + 1)
    f    = np.zeros(ndof)
    n    = node_index(ix, iy, iz, Nx, Ny)
    f[3*n]   = force_N[0]
    f[3*n+1] = force_N[1]
    f[3*n+2] = force_N[2]
    return f


@dataclass
class BoundaryConditions:
    """Container for fixed DOFs and force vector of a mesh problem.

    Build with the helper functions above, then pass to TopoptPipeline.
    """
    fixed_dofs: np.ndarray      # int64 array of fixed global DOF indices
    force_vec:  np.ndarray      # (ndof,) float64 force vector

    @classmethod
    def headphone_holder(
        cls,
        Nx: int, Ny: int, Nz: int,
        h: float,
        offset: Sequence[float],
        base_radius_mm: float,
        arm_tip_mm: Sequence[float],
        head_mass_g: float,
    ) -> "BoundaryConditions":
        """Standard BCs for a headphone holder standing on its base.

        Fixed: all nodes on z=0 within base_radius (clamped disc base).
        Load:  vertical gravity load at arm tip node (downward z).
        """
        g_ms2  = 9.81
        F_N    = head_mass_g * 1e-3 * g_ms2          # [N] downward

        fixed = fixed_cylinder_base_dofs(
            Nx, Ny, Nz, h, offset, base_radius_mm,
        )
        f = point_load_dof(
            Nx, Ny, Nz, h, offset,
            position_mm=arm_tip_mm,
            force_N=(0.0, 0.0, -F_N),
        )
        ndof_expected = 3 * (Nx + 1) * (Ny + 1) * (Nz + 1)
        if len(f) != ndof_expected:
            f = np.zeros(ndof_expected)
        return cls(fixed_dofs=fixed, force_vec=f)
