"""Physical measurement utilities for STL geometry."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh


@dataclass
class ShapeMeasure:
    """Physical measurements of a solid shape.

    Units (all derived from mm-coordinate geometry):
      volume             : mm^3
      surface_area       : mm^2
      center_of_gravity  : mm  (world coordinates)
      mass               : g   (requires density_g_cm3)
      inertia_tensor     : g*mm^2 at CoG (if density supplied, else mm^2)
    """
    volume_mm3:           float
    surface_area_mm2:     float
    center_of_gravity_mm: np.ndarray    # (3,) world mm
    inertia_tensor_g_mm2: np.ndarray    # (3,3) at CoG
    mass_g:               Optional[float] = None

    def summary(self) -> str:
        cg = self.center_of_gravity_mm
        lines = [
            f"Volume:       {self.volume_mm3:.1f} mm3",
            f"Surface area: {self.surface_area_mm2:.1f} mm2",
            f"CoG:          ({cg[0]:.2f}, {cg[1]:.2f}, {cg[2]:.2f}) mm",
        ]
        if self.mass_g is not None:
            lines.append(f"Mass:         {self.mass_g:.2f} g")
        return "\n".join(lines)


class Measure:
    """Compute physical measurements from geometry.

    Usage
    -----
    m = Measure.from_stl("docs/generated_shape.stl",
                          density_g_cm3=1.24, infill_pct=20.0)
    print(m.summary())
    """

    @staticmethod
    def from_stl(
        stl_path:        str | Path,
        density_g_cm3:   Optional[float] = None,
        infill_pct:      Optional[float] = None,
    ) -> ShapeMeasure:
        """Measure a closed STL mesh (trimesh-based, no picogk needed).

        Parameters
        ----------
        stl_path      : path to STL file
        density_g_cm3 : raw filament density (PLA: 1.24)
        infill_pct    : print infill percentage 0-100 (default: solid = 100)
        """
        mesh = trimesh.load(str(stl_path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Could not load mesh from {stl_path}")
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)

        vol  = float(mesh.volume)
        area = float(mesh.area)
        cog  = mesh.center_mass.astype(float)

        # Inertia tensor: trimesh gives (3,3) in mesh units^5 * unit_density
        # = mm^5 * (1/mm^3) = mm^2 when interpreted as normalised-by-mass.
        # To get g*mm^2: multiply by effective density [g/mm^3].
        I_raw = mesh.moment_inertia.astype(float)  # (3,3)

        mass_g = None
        eff_density_g_mm3 = None
        if density_g_cm3 is not None:
            ip = float(infill_pct) if infill_pct is not None else 100.0
            eff_density_g_mm3 = float(density_g_cm3) * (ip / 100.0) / 1000.0
            mass_g = vol * eff_density_g_mm3

        I_scaled = I_raw * eff_density_g_mm3 if eff_density_g_mm3 is not None else I_raw

        return ShapeMeasure(
            volume_mm3=round(vol, 3),
            surface_area_mm2=round(area, 3),
            center_of_gravity_mm=cog,
            inertia_tensor_g_mm2=I_scaled,
            mass_g=round(mass_g, 3) if mass_g is not None else None,
        )

    @staticmethod
    def from_voxels(
        vox,
        density_g_cm3:   Optional[float] = None,
        infill_pct:      Optional[float] = None,
    ) -> ShapeMeasure:
        """Measure from a picogk Voxels object (requires picogk.go context).

        Exports to a temporary STL and delegates to from_stl().
        """
        import os
        import tempfile

        from picogk import Mesh

        fd, tmp = tempfile.mkstemp(suffix=".stl")
        os.close(fd)
        try:
            mesh = Mesh.from_voxels(vox)
            mesh.SaveToStlFile(tmp)
            return Measure.from_stl(tmp, density_g_cm3, infill_pct)
        finally:
            os.unlink(tmp)

    @staticmethod
    def principal_axes(
        measure: ShapeMeasure,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Eigendecompose the inertia tensor at CoG.

        Returns
        -------
        eigenvalues  : (3,) principal moments of inertia [same units as tensor]
        eigenvectors : (3,3) columns are principal axes (sorted ascending)
        """
        vals, vecs = np.linalg.eigh(measure.inertia_tensor_g_mm2)
        return vals, vecs
