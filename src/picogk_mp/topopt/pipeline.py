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
from scipy.ndimage import label as _label

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
        """Load STL and rasterise to a boolean voxel mask at self.h resolution.

        Uses trimesh.voxel.creation.voxelize (scanline flood-fill) which is
        memory-efficient and does not require rtree or ray-casting 130K points.
        """
        mesh = trimesh.load(str(self.stl_path), force="mesh")
        bounds = mesh.bounds
        extent = bounds[1] - bounds[0]
        Nx, Ny, Nz = np.ceil(extent / self.h).astype(int)

        print(
            f"Voxelising '{self.stl_path.name}' at {self.h} mm: "
            f"~{Nx} x {Ny} x {Nz} elements..."
        )
        t0 = time.time()

        # trimesh native voxelization: surface rasterisation + flood fill.
        # Returns VoxelGrid whose .matrix is a dense boolean occupancy array.
        vox = trimesh.voxel.creation.voxelize(mesh, pitch=self.h)

        # Extract dense mask.  trimesh stores it as (Nz, Ny, Nx) in some
        # versions and (Nx, Ny, Nz) in others; we normalise via the transform.
        raw = np.asarray(vox.matrix, dtype=bool)

        # VoxelGrid.transform maps voxel indices (integers) to voxel CENTRES.
        # We want our FEM grid to use the same axes as physical space (x,y,z).
        # trimesh typically stores with first axis = x in 4.x.
        mask = raw  # assumed (Nx, Ny, Nz)

        # Physical origin = centre of voxel (0,0,0) minus half pitch
        centre_000  = vox.transform[:3, 3]
        origin      = centre_000 - self.h / 2.0

        Nx_v, Ny_v, Nz_v = mask.shape
        print(
            f"  Voxelisation done in {time.time()-t0:.1f}s -- "
            f"{Nx_v} x {Ny_v} x {Nz_v} grid, "
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

        # vol_frac: "keep this fraction of the INITIAL solid material".
        # Convert both vol_target and er to grid-fractions so beso_step
        # receives consistent fractions of the total voxel count.
        initial_n_solid = int(mask.sum())
        n_total         = mask.size

        # Target: e.g. keep 50% of 4702 solid → 2351 elements → 1.74% of grid
        vol_target_grid = self.vol_frac * initial_n_solid / n_total

        # er: evolutionary ratio relative to INITIAL solid count.
        # e.g. er=0.02 → remove/add at most 2% * 4702 = 94 elements per step.
        # Expressed as a grid fraction: 94/134937 = 0.0007.
        # This keeps the structure connected by limiting per-step material change.
        er_grid = self.er * initial_n_solid / n_total

        n_target_solid  = int(round(self.vol_frac * initial_n_solid))
        n_steps_est     = max(1, int(round(
            (initial_n_solid - n_target_solid) / (self.er * initial_n_solid)
        )))

        print()
        print("=" * 62)
        print("  BESO TOPOLOGY OPTIMISATION")
        print(f"  Initial solid elements : {initial_n_solid:,} "
              f"({initial_n_solid/n_total:.1%} of grid)")
        print(f"  Target (keep {self.vol_frac:.0%} of solid) : "
              f"{n_target_solid:,} elements "
              f"({vol_target_grid:.2%} of grid)")
        print(f"  Steps to target (er={self.er:.0%}) : ~{n_steps_est}")
        print(f"  Max iterations         : {self.max_iter}")
        print(f"  Voxel size             : {self.h} mm")
        print("=" * 62)
        n_fixed = len(bc.fixed_dofs)
        force_mag = float(np.linalg.norm(bc.force_vec))
        print(f"  Fixed DOFs     : {n_fixed:,}")
        print(f"  Force magnitude: {force_mag:.3f} N")
        print(f"  {'Iter':>4}  {'Vol%':>6}  {'NSolid':>7}  {'Compliance':>12}  {'dC%':>8}  Time")
        print(f"         (of init)  (abs)")

        t_total = time.time()
        prev_state = state  # last state with valid FEM solution
        for it in range(1, self.max_iter + 1):
            t0 = time.time()

            # FEM solve
            u = fem_solve(
                state.mask, edofs, KE,
                bc.fixed_dofs, bc.force_vec,
            )

            # Sensitivity
            alpha = element_strain_energy(u, edofs, KE)

            # Guard against garbage FEM solutions (e.g. when the direct-solver
            # fallback also fails).  Strain energy is always non-negative for a
            # positive semi-definite KE, so negative compliance signals that
            # the solver returned a non-physical displacement field.
            solid_compliance = float(alpha[state.mask.ravel()].sum())
            if solid_compliance < 0 and it > 1:
                print(f"  [FEM] Negative compliance ({solid_compliance:.2e}) -- "
                      f"solver diverged; reverting to iter {it-1} topology.")
                state = prev_state
                break

            prev_state = state   # snapshot of last verified-good topology

            # ESO update -- use er_grid so removal rate is ~er% of initial solid
            state = beso_step(
                state, alpha,
                vol_target=vol_target_grid,
                er=er_grid,
                r_filter=self.r_filter,
                add_ratio=self.add_ratio,
            )

            # Connectivity enforcement (6-connectivity / face-adjacent only):
            # Remove solid elements not face-connected to the base (iz=0 face).
            # Using 6-connectivity instead of the default 26 means that
            # elements touching only at an edge or corner are treated as
            # disconnected -- they share at most 6 DOFs and create near-rigid
            # body modes that make K nearly singular.
            _struct6 = np.zeros((3, 3, 3), dtype=bool)
            _struct6[1, 1, 0] = _struct6[1, 1, 2] = True   # ±z
            _struct6[1, 0, 1] = _struct6[1, 2, 1] = True   # ±y
            _struct6[0, 1, 1] = _struct6[2, 1, 1] = True   # ±x
            _struct6[1, 1, 1] = True                        # self

            conn_mask = state.mask
            labeled, n_comp = _label(conn_mask, structure=_struct6)
            if n_comp > 1:
                # Identify connected components touching the bottom face (iz=0)
                base_labels = set(labeled[:, :, 0].ravel()) - {0}
                if base_labels:
                    keep = np.zeros_like(conn_mask, dtype=bool)
                    for lbl in base_labels:
                        keep |= labeled == lbl
                    n_pruned = int(conn_mask.sum()) - int(keep.sum())
                    if n_pruned > 0:
                        print(f"  [CONN] {n_pruned} disconnected elements pruned")
                    state = BESOState(
                        mask=keep,
                        alpha_history=state.alpha_history,
                        compliance_history=state.compliance_history,
                    )

            # Reporting
            comp = state.compliance_history[-1]
            hist = state.compliance_history
            if len(hist) >= 2 and abs(hist[-2]) > 1e-30:
                dc_pct = 100 * abs(hist[-1] - hist[-2]) / abs(hist[-2])
            else:
                dc_pct = float("nan")

            elapsed = time.time() - t0
            vol_of_init = (state.n_solid / initial_n_solid
                           if initial_n_solid > 0 else 0.0)
            print(
                f"  {it:>4}  {vol_of_init:>6.1%}  {state.n_solid:>7,}  "
                f"{comp:>12.4e}  {dc_pct:>7.2f}%  {elapsed:.1f}s"
            )

            # Only accept convergence once volume is within one step of target.
            # Early convergence at stable compliance while still far from target
            # would stop ESO prematurely (removed elements had near-zero energy).
            vol_at_target = state.n_solid <= n_target_solid + max(
                1, int(round(er_grid * n_total))
            )
            if vol_at_target and is_converged(
                state, tol=self.conv_tol, patience=self.conv_patience
            ):
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
        """Convert boolean voxel mask to STL via marching cubes.

        Uses scipy.ndimage for marching cubes to avoid trimesh VoxelGrid
        API differences across versions.
        """
        from scipy.ndimage import zoom as _zoom

        pitch = self.h
        ox, oy, oz = self.offset

        # Build voxel grid aligned with the physical space, run marching cubes
        try:
            # Try skimage first (fastest, best quality)
            from skimage.measure import marching_cubes as _mc
            verts, faces, normals, _ = _mc(
                mask.astype(np.float32), level=0.5, spacing=(pitch, pitch, pitch)
            )
            verts += np.array([ox, oy, oz])
            mesh_out = trimesh.Trimesh(vertices=verts, faces=faces,
                                       vertex_normals=normals)
            # Marching cubes winding direction depends on gradient sign;
            # ensure outward normals (positive volume).
            if mesh_out.volume < 0:
                mesh_out.invert()
        except ImportError:
            # Fall back to trimesh VoxelGrid marching cubes
            transform = np.eye(4)
            transform[0, 0] = transform[1, 1] = transform[2, 2] = pitch
            transform[:3, 3] = [ox, oy, oz]
            vg = trimesh.voxel.VoxelGrid(
                trimesh.voxel.encoding.DenseEncoding(mask),
                transform=transform,
            )
            mesh_out = vg.marching_cubes

        path.parent.mkdir(parents=True, exist_ok=True)
        mesh_out.export(str(path))
        size_kb = path.stat().st_size // 1024
        print(f"  Optimised STL -> {path}  ({size_kb} KB)")
        return path
