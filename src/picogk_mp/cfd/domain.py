"""STL -> 2D flow domain for LBM simulation.

Loads the STL, voxelises it, extracts a 2D cross-section perpendicular to
the flow axis, and pads the domain with fluid cells for inlet/outlet/wake.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class FlowDomain:
    """Geometry and physical parameters for one 2D flow simulation."""

    mask: np.ndarray       # (Ny, Nx) bool, True = solid
    dx_m: float            # physical cell size [m]
    char_length_m: float   # characteristic length (bounding box in flow dir) [m]
    U_lb: float            # inlet velocity in lattice units (Ma = U_lb/cs)
    nu_lb: float           # kinematic viscosity in lattice units
    Re: float              # Reynolds number (physical)
    velocity_m_s: float    # physical inlet velocity [m/s]
    nu_air_m2s: float      # kinematic viscosity of fluid [m2/s]

    @property
    def Ny(self) -> int:
        return self.mask.shape[0]

    @property
    def Nx(self) -> int:
        return self.mask.shape[1]


def build_domain(
    stl_path: str | Path,
    velocity_m_s: float = 0.5,
    flow_axis: int = 0,        # 0=x, 1=y, 2=z
    resolution_mm: float = 3.0,
    pad_upstream: int = 8,
    pad_downstream: int = 12,
    pad_transverse: int = 8,
    slice_fraction: float = 0.5,
    nu_air_m2s: float = 1.5e-5,
    U_lb: float = 0.1,
) -> FlowDomain:
    """Build a 2D flow domain from an STL file.

    Parameters
    ----------
    stl_path        : Path to the input STL.
    velocity_m_s    : Physical inlet velocity [m/s].
    flow_axis       : Axis aligned with the flow direction (0=x, 1=y, 2=z).
    resolution_mm   : Voxel size [mm].
    pad_upstream    : Fluid cells added upstream (inlet side).
    pad_downstream  : Fluid cells added downstream (outlet/wake).
    pad_transverse  : Fluid cells on each side perpendicular to flow.
    slice_fraction  : 0..1, fractional position of the 2D slice along
                      the axis perpendicular to the 2D plane.
    nu_air_m2s      : Kinematic viscosity of the fluid [m2/s].
    U_lb            : Target lattice Mach number (keep <= 0.15 for stability).

    Returns
    -------
    FlowDomain with mask, physical parameters, and lattice unit conversions.
    """
    mesh = trimesh.load(str(stl_path), force="mesh")

    dx_m = resolution_mm * 1e-3

    # Voxelise the mesh
    vox = mesh.voxelized(pitch=resolution_mm).fill()
    matrix = vox.matrix           # (Nvox_x, Nvox_y, Nvox_z) bool

    # Choose which axis is the "out-of-plane" direction for the 2D slice.
    # We always simulate flow in 2D: (flow_axis, transverse_axis).
    # The slice axis is the remaining one.
    all_axes = [0, 1, 2]
    all_axes.remove(flow_axis)
    # Prefer z as the slice axis; otherwise take the second transverse
    if 2 in all_axes:
        slice_axis = 2
    else:
        slice_axis = all_axes[1]
    transverse_axis = [a for a in all_axes if a != slice_axis][0]

    # Extract 2D slice
    n_slices = matrix.shape[slice_axis]
    idx = int(np.clip(slice_fraction * (n_slices - 1), 0, n_slices - 1))

    slicer: list = [slice(None), slice(None), slice(None)]
    slicer[slice_axis] = idx
    slab = matrix[tuple(slicer)]     # 2D array: shape (Na, Nb)

    # Reorder axes so that 2D mask is (transverse, flow) = (Ny, Nx)
    axes_remaining = [a for a in [0, 1, 2] if a != slice_axis]
    if axes_remaining[0] == flow_axis:
        slab = slab.T     # ensure rows = transverse, cols = flow

    mask_raw = slab       # (Ny_raw, Nx_raw)

    # Characteristic length in flow direction [cells]
    char_length_cells = float(mask_raw.shape[1])   # width of bounding box
    char_length_m = char_length_cells * dx_m

    # Pad the domain
    mask_padded = np.pad(
        mask_raw,
        ((pad_transverse, pad_transverse),
         (pad_upstream, pad_downstream)),
        constant_values=False,
    )

    # Physical -> lattice unit conversion
    # dt = dx_m * U_lb / velocity_m_s
    dt = dx_m * U_lb / velocity_m_s
    nu_lb = nu_air_m2s * dt / dx_m**2

    Re = velocity_m_s * char_length_m / nu_air_m2s

    return FlowDomain(
        mask=mask_padded,
        dx_m=dx_m,
        char_length_m=char_length_m,
        U_lb=U_lb,
        nu_lb=nu_lb,
        Re=Re,
        velocity_m_s=velocity_m_s,
        nu_air_m2s=nu_air_m2s,
    )
