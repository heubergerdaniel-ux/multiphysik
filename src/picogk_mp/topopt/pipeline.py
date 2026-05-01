"""TopoptPipeline: BESO topology optimisation for voxel-based CAD parts.

Workflow
--------
1.  Load geometry from an STL file (the picogk STL export).
2.  Voxelise at the topopt resolution (default 3 mm -- much coarser than
    the final 0.5 mm picogk voxels; topopt does not need fine resolution).
3.  Run the BESO loop:
      a. Assemble K from the current solid mask.
      b. Apply boundary conditions (fixed base + arm-tip load).
      c. Solve K*u = f.
      d. Compute per-element sensitivity (strain energy).
      e. BESO update mask toward target volume fraction.
      f. Check convergence.
4.  Export the optimised mask as a new STL (via trimesh marching cubes).

Usage::

    from picogk_mp.topopt import TopoptPipeline
    from picogk_mp.topopt.boundary import BoundaryConditions

    pipeline = TopoptPipeline(
        stl_path="tests/fixtures/headphone_holder_v2.stl",
        topopt_h_mm=3.0,
        vol_frac=0.40,
        max_iter=40,
    )
    bc = BoundaryConditions.headphone_holder(
        *pipeline.grid_shape, pipeline.h, pipeline.offset,
        base_radius_mm=48, arm_tip_mm=(-82, 0, 244), head_mass_g=400,
    )
    result_stl = pipeline.run(bc, out_stl="docs/holder_optimised.stl")
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import trimesh

from picogk_mp.topopt.fem import (
    element_dof_indices,
    element_stiffness,
    element_strain_energy,
    fem_solve,
)
from picogk_mp.topopt.beso import BESOState, beso_step, is_converged
from picogk_mp.topopt.boundary import BoundaryConditions


class TopoptPipeline:
    """End-to-end BESO topology optimisation from STL to STL.

    Parameters
    ----------
    stl_path      : input STL (full-density geometry from picogk)
    topopt_h_mm   : voxel size for topopt mesh [mm]  (default 3 mm)
    vol_frac      : target solid volume fraction after optimisation (0-1)
    max_iter      : maximum BESO iterations
    E0            : Young's modulus of filament [MPa]  (PLA: ~3500)
    nu            : Poisson ratio                       (PLA: ~0.36)
    er            : BESO evolutionary removal ratio per step
    r_filter      : sensitivity filter radius [elements]
    add_ratio     : bidirectional addition threshold
    conv_tol      : convergence tolerance on relative compliance change
    conv_patience : iterations of compliance stability required
    """

    def __init__(
        self,
        stl_path: str | Path,
        topopt_h_mm: float = 3.0,
        vol_frac: float = 0.40,
        max_iter: int = 40,
        E0: float = 3500.0,          # MPa -- PLA typical
        nu: float = 0.36,
        er: float = 0.02,
        r_filter: float = 1.5,
        add_ratio: float = 0.01,
        conv_tol: float = 1e-3,
        conv_patience: int = 5,
    ) -> None:
        self.stl_path     = Path(stl_path)
        self.h            = float(topopt_h_mm)
        self.vol_frac     = float(vol_frac)
        self.max_iter     = int(max_iter)
        self.E0           = float(E0)
        self.nu           = float(nu)
        self.er           = float(er)
        self.r_filter     = float(r_filter)
        self.add_ratio    = float(add_ratio)
        self.conv_tol     = float(conv_tol)
        self.conv_patience = int(conv_patience)

        # Filled by _voxelise()
        self._mask: Optional[np.ndarray] = None
        self._offset: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Properties (available after first call to run() or _voxelise())
    # ------------------------------------------------------------------

    @property
    def mask(self) -> np.ndarray:
        if self._mask is None:
            self._voxelise()
        return self._mask                           # type: ignore[return-value]

    @property
    def grid_shape(self):
        """(Nx, Ny, Nz) element counts."""
        return self.mask.shape

    @property
    def offset(self) -> np.ndarray:
        """Physical (x,y,z) of the grid origin [mm]."""
        if self._offset is None:
            self._voxelise()
        return self._offset                         # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Voxelisation
    # ------------------------------------------------------------------

    def _voxelise(self) -> None:
        """Load STL and rasterise to a boolean voxel mask at self.h resolution."""
        mesh = trimesh.load(str(self.stl_path), force="mesh")
        bounds   = mesh.bounds                      # (2,3): [[min],[max]]
        origin   = bounds[0]
        extent   = bounds[1] - bounds[0]
        Nx, Ny, Nz = np.ceil(extent / self.h).astype(int)

        print(
            f"Voxelising '{self.stl_path.name}' at {self.h} mm: "
            f"{Nx} x {Ny} x {Nz} = {Nx*Ny*Nz:,} elements..."
        )
        t0 = time.time()

        # Sample element centres and test inside/outside
        cx = origin[0] + (np.arange(Nx) + 0.5) * self.h
        cy = origin[1] + (np.arange(Ny) + 0.5) * self.h
        cz = origin[2] + (np.arange(Nz) + 0.5) * self.h

        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
        pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

        inside = mesh.contains(pts)
        mask = inside.reshape(Nx, Ny, Nz)

        print(
            f"  Voxelisation done in {time.time()-t0:.1f}s -- "
            f"{mask.sum():,} solid elements "
            f"({100*mask.mean():.1f}% fill)"
        )

        self._mask   = mask
        self._offset = origin

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------

    def run(
        self,
        bc: BoundaryConditions,
        out_stl: Optional[str | Path] = None,
    ) -> Optional[Path]:
        """Run BESO optimisation.

        Parameters
        ----------
        bc      : boundary conditions (fixed DOFs + force vector)
        out_stl : optional path to write the optimised STL

        Returns
        -------
        Path to the written STL, or None if out_stl is None.
        """
        mask   = self.mask.copy()
        Nx, Ny, Nz = mask.shape

        KE    = element_stiffness(self.h, self.E0, self.nu)
        edofs = element_dof_indices(Nx, Ny, Nz)

        state = BESOState(mask=mask.copy())

        print()
        print("=" * 62)
        print("  BESO TOPOLOGY OPTIMISATION")
        print(f"  Target volume fraction : {self.vol_frac:.0%}")
        print(f"  Max iterations         : {self.max_iter}")
        print(f"  Voxel size             : {self.h} mm")
        print("=" * 62)
        print(f"  {'Iter':>4}  {'Vol%':>6}  {'Compliance':>12}  {'dC%':>8}  Time")

        t_total = time.time()
        for it in range(1, self.max_iter + 1):
            t0 = time.time()

            # FEM solve
            u = fem_solve(
                state.mask, edofs, KE,
                bc.fixed_dofs, bc.force_vec,
            )

            # Sensitivity
            alpha = element_strain_energy(u, edofs, KE)

            # BESO update
            state = beso_step(
                state, alpha,
                vol_target=self.vol_frac,
                er=self.er,
                r_filter=self.r_filter,
                add_ratio=self.add_ratio,
            )

            # Reporting
            comp = state.compliance_history[-1]
            hist = state.compliance_history
            if len(hist) >= 2 and abs(hist[-2]) > 1e-30:
                dc_pct = 100 * abs(hist[-1] - hist[-2]) / abs(hist[-2])
            else:
                dc_pct = float("nan")

            elapsed = time.time() - t0
            print(
                f"  {it:>4}  {state.volume_fraction:>6.1%}  "
                f"{comp:>12.4e}  {dc_pct:>7.2f}%  {elapsed:.1f}s"
            )

            if is_converged(state, tol=self.conv_tol, patience=self.conv_patience):
                print(f"\n  Converged at iteration {it}.")
                break
        else:
            print(f"\n  Reached max iterations ({self.max_iter}).")

        print(f"  Total time: {time.time()-t_total:.1f}s")
        print("=" * 62)
        print()

        self._final_mask = state.mask

        if out_stl is not None:
            return self._export_stl(state.mask, Path(out_stl))
        return None

    # ------------------------------------------------------------------
    # STL export from optimised mask (marching cubes via trimesh)
    # ------------------------------------------------------------------

    def _export_stl(self, mask: np.ndarray, path: Path) -> Path:
        """Convert boolean voxel mask to STL via marching cubes."""
        # trimesh voxel grid -> mesh
        pitch = self.h
        ox, oy, oz = self.offset

        vg = trimesh.voxel.VoxelGrid(
            trimesh.voxel.encoding.DenseEncoding(mask),
            transform=trimesh.transformations.scale_and_translate(
                scale=pitch,
                translate=[ox, oy, oz],
            ),
        )
        mesh_out = vg.marching_cubes
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh_out.export(str(path))
        size_kb = path.stat().st_size // 1024
        print(f"  Optimised STL -> {path}  ({size_kb} KB)")
        return path
