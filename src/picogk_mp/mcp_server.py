"""MCP server -- multiphysik design pipeline.

Exposes the BESO topology optimiser, physics checks, and STL renderer
as tools callable by Claude Code or any MCP-compatible client.

Claude handles natural-language understanding; these tools handle the
physics compute.  No SimEngine / parameter-resolver scaffolding is
needed -- Claude asks for missing information conversationally.

Registration (project-level .mcp.json)
---------------------------------------
{
  "mcpServers": {
    "multiphysik": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "picogk-mp-mcp"]
    }
  }
}

Tools
-----
run_topopt       Full BESO optimisation: STL in -> optimised STL + PNG out
check_physics    Structural safety-factor checks on any STL
render_stl       STL -> PNG preview image (returned inline)
list_stls        Discover STL files available in the project
"""
from __future__ import annotations

import base64
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from fastmcp import FastMCP
from fastmcp.utilities.types import Image

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "multiphysik",
    instructions=(
        "Physics-aware generative design pipeline for 3D-printed parts. "
        "Available tools: run_topopt (BESO topology optimisation), "
        "check_physics (structural safety factors), render_stl (preview), "
        "list_stls (discover available files). "
        "Always render the STL after optimisation so the user can see it."
    ),
)

# Project root: three levels up from this file
# src/picogk_mp/mcp_server.py -> src/picogk_mp -> src -> project root
_THIS = Path(__file__).resolve()
ROOT  = _THIS.parent.parent.parent
DOCS  = ROOT / "docs"
FIXTURES = ROOT / "tests" / "fixtures"


# ===========================================================================
# Tool 1 -- topology optimisation
# ===========================================================================

@mcp.tool()
def run_topopt(
    stl_path: str,
    load_mass_g: float,
    arm_tip_x_mm: float,
    arm_tip_y_mm: float,
    arm_tip_z_mm: float,
    base_radius_mm: float = 48.0,
    vol_frac: float = 0.75,
    safety_factor: float = 2.0,
    resolution_mm: float = 3.0,
    max_iter: int = 40,
    out_stl: Optional[str] = None,
) -> dict:
    """Run BESO topology optimisation on a 3D-printed part.

    Iteratively removes low-stress material while preserving structural
    integrity.  Returns the optimised STL path, achieved volume fraction,
    compliance trajectory, and structural physics check results.

    Parameters
    ----------
    stl_path       : Input STL (absolute path or relative to project root).
    load_mass_g    : Operating load mass [g] -- e.g. headphone mass, payload.
                     The design load is load_mass_g * safety_factor.
    arm_tip_x_mm   : X coordinate of the load application point [mm].
    arm_tip_y_mm   : Y coordinate of the load application point [mm].
    arm_tip_z_mm   : Z coordinate of the load application point [mm].
    base_radius_mm : Radius of the fixed support disc at z=0 [mm].
    vol_frac       : Fraction of initial solid material to KEEP after
                     optimisation (0.5 = aggressive, 0.85 = conservative).
    safety_factor  : Multiplier applied to load_mass_g for FEM sizing.
    resolution_mm  : Voxel pitch for the FEM mesh [mm].  3 mm balances
                     accuracy and speed (~10-15 s/iteration).
    max_iter       : Maximum BESO iterations (40 is usually sufficient).
    out_stl        : Output path for optimised STL.
                     Default: docs/optimised_<input-name>.stl
    """
    from picogk_mp.topopt import TopoptPipeline, BoundaryConditions

    # --- resolve paths ---
    stl_in = Path(stl_path) if Path(stl_path).is_absolute() else ROOT / stl_path
    if not stl_in.exists():
        return {"status": "error", "message": f"STL not found: {stl_in}"}

    stem = stl_in.stem
    out_stl_path = (
        Path(out_stl) if out_stl
        else DOCS / f"optimised_{stem}.stl"
    )
    out_png_path = out_stl_path.with_suffix(".png")
    out_stl_path.parent.mkdir(parents=True, exist_ok=True)

    # --- design load ---
    design_mass_g  = load_mass_g * safety_factor
    design_force_N = design_mass_g * 1e-3 * 9.81

    try:
        t0 = time.time()

        pipeline = TopoptPipeline(
            stl_in,
            topopt_h_mm=resolution_mm,
            vol_frac=vol_frac,
            max_iter=max_iter,
        )

        bc = BoundaryConditions.headphone_holder(
            *pipeline.grid_shape,
            pipeline.h,
            pipeline.offset,
            base_radius_mm=base_radius_mm,
            arm_tip_mm=(arm_tip_x_mm, arm_tip_y_mm, arm_tip_z_mm),
            head_mass_g=design_mass_g,
        )

        pipeline.run(bc, out_stl=out_stl_path)
        elapsed = time.time() - t0

        # --- compliance history ---
        final_state = getattr(pipeline, "_final_state", None)
        compliance_history = (
            [round(c, 4) for c in final_state.compliance_history]
            if final_state is not None
            else []
        )

        # --- element counts ---
        elements_initial = int(pipeline.mask.sum())
        elements_final   = int(pipeline._final_mask.sum())
        vol_achieved     = round(elements_final / elements_initial, 3)

        # --- render preview ---
        _render_to_file(out_stl_path, out_png_path)

        # --- physics checks (at operating load, not design load) ---
        arm_reach_mm = abs(arm_tip_x_mm)   # horizontal reach from base axis
        physics = _physics_checks(
            stl_path=str(out_stl_path),
            load_mass_g=load_mass_g,
            arm_reach_mm=arm_reach_mm,
            base_radius_mm=base_radius_mm,
        )

        return {
            "status": "ok",
            "stl_path": str(out_stl_path),
            "png_path": str(out_png_path),
            "elements_initial": elements_initial,
            "elements_final":   elements_final,
            "vol_frac_achieved": vol_achieved,
            "design_force_N":    round(design_force_N, 2),
            "elapsed_s":         round(elapsed, 1),
            "compliance_nm":     compliance_history,
            "physics":           physics,
        }

    except Exception:
        return {
            "status": "error",
            "message": traceback.format_exc(),
        }


# ===========================================================================
# Tool 2 -- physics checks only
# ===========================================================================

@mcp.tool()
def check_physics(
    stl_path: str,
    load_mass_g: float,
    arm_reach_mm: float,
    base_radius_mm: float = 48.0,
    stem_min_radius_mm: float = 7.0,
    infill_percent: float = 20.0,
    material_density_g_cm3: float = 1.24,
    material_yield_mpa: float = 55.0,
) -> dict:
    """Run structural physics checks on an STL file.

    Evaluates tipping stability and stem bending stress at the given
    operating load.  Returns safety factors and pass/fail per check.

    Parameters
    ----------
    stl_path              : STL to evaluate (absolute or project-relative).
    load_mass_g           : Operating load [g] (not factored).
    arm_reach_mm          : Horizontal distance from base axis to load [mm].
    base_radius_mm        : Base disc radius [mm].
    stem_min_radius_mm    : Radius at narrowest stem cross-section [mm].
    infill_percent        : FDM infill % -- affects stand mass (default 20).
    material_density_g_cm3: Raw filament density [g/cm3] (PLA: 1.24).
    material_yield_mpa    : Filament yield strength [MPa] (PLA: 55).
    """
    stl_p = Path(stl_path) if Path(stl_path).is_absolute() else ROOT / stl_path
    return _physics_checks(
        stl_path=str(stl_p),
        load_mass_g=load_mass_g,
        arm_reach_mm=arm_reach_mm,
        base_radius_mm=base_radius_mm,
        stem_min_radius_mm=stem_min_radius_mm,
        infill_percent=infill_percent,
        material_density_g_cm3=material_density_g_cm3,
        material_yield_mpa=material_yield_mpa,
    )


# ===========================================================================
# Tool 3 -- render STL -> PNG (returned inline)
# ===========================================================================

@mcp.tool()
def render_stl(
    stl_path: str,
    out_png: Optional[str] = None,
) -> Image:
    """Render an STL file to a PNG preview image.

    The image is returned directly so Claude can display it inline in
    the conversation without any extra steps.

    Parameters
    ----------
    stl_path : STL to render (absolute or project-relative path).
    out_png  : Optional path to save the PNG.  Defaults to same
               directory as the STL with a .png extension.
    """
    stl_p = Path(stl_path) if Path(stl_path).is_absolute() else ROOT / stl_path
    png_p = (
        Path(out_png) if out_png
        else stl_p.with_suffix(".png")
    )
    _render_to_file(stl_p, png_p)
    return Image(path=str(png_p))


# ===========================================================================
# Tool 4 -- list available STL files
# ===========================================================================

@mcp.tool()
def list_stls() -> dict:
    """List all STL files available in the project.

    Scans tests/fixtures/ and docs/ for STL files and returns their
    paths, names, and sizes.  Use the returned paths directly with
    run_topopt or check_physics.
    """
    seen: set[Path] = set()
    stls = []
    for search_dir in [FIXTURES, DOCS, ROOT]:
        for stl in sorted(search_dir.glob("**/*.stl")):
            resolved = stl.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                kb = resolved.stat().st_size // 1024
                stls.append({
                    "path": str(resolved),
                    "name": resolved.name,
                    "size_kb": kb,
                    "relative": str(resolved.relative_to(ROOT)),
                })
            except (OSError, ValueError):
                pass
    return {"stls": stls, "count": len(stls)}


# ===========================================================================
# Internal helpers
# ===========================================================================

def _physics_checks(
    stl_path: str,
    load_mass_g: float,
    arm_reach_mm: float,
    base_radius_mm: float = 48.0,
    stem_min_radius_mm: float = 7.0,
    infill_percent: float = 20.0,
    material_density_g_cm3: float = 1.24,
    material_yield_mpa: float = 55.0,
) -> dict:
    import trimesh
    from picogk_mp.physics.checks import TippingCheck, StemBendingCheck

    try:
        mesh = trimesh.load(str(stl_path), force="mesh")
        volume_mm3 = float(mesh.volume)
    except Exception as exc:
        return {"status": "error", "message": f"Cannot load STL: {exc}"}

    ctx = {
        "head_mass_g":    load_mass_g,
        "base_r_mm":      base_radius_mm,
        "arm_reach_mm":   arm_reach_mm,
        "volume_mm3":     volume_mm3,
        "infill_pct":     infill_percent,
        "density_g_cm3":  material_density_g_cm3,
        "stem_r_min_mm":  stem_min_radius_mm,
        "yield_mpa":      material_yield_mpa,
    }

    checks_out: dict = {}
    all_passed = True
    for check in [TippingCheck(), StemBendingCheck()]:
        r = check.evaluate(ctx)
        checks_out[check.name] = {
            "passed":      r.passed,
            "sf":          round(r.sf, 2),
            "sf_required": r.sf_required,
            "detail":      r.detail,
        }
        if not r.passed:
            all_passed = False

    return {
        "status":      "ok",
        "all_passed":  all_passed,
        "volume_mm3":  round(volume_mm3),
        "stand_mass_g": round(volume_mm3 * (infill_percent / 100)
                              * material_density_g_cm3 / 1000, 1),
        "checks":      checks_out,
    }


def _render_to_file(stl_path: Path, png_path: Path) -> None:
    """Render STL to PNG via vedo offscreen renderer."""
    try:
        import vedo
        mesh_v = vedo.load(str(stl_path))
        mesh_v.color([100, 140, 120]).lighting("metallic")
        plt = vedo.Plotter(
            offscreen=True, size=(1280, 960),
            bg=(35, 35, 46), bg2=(10, 10, 18),
        )
        plt.add(mesh_v)
        plt.show()
        cam = plt.camera
        cam.SetPosition(65, -480, 295)
        cam.SetFocalPoint(-21, 0, 138)
        cam.SetViewUp(0, 0, 1)
        plt.renderer.ResetCameraClippingRange()
        plt.render()
        png_path.parent.mkdir(parents=True, exist_ok=True)
        plt.screenshot(str(png_path))
        plt.close()
    except Exception:
        # Render is optional; if it fails the tool still returns data.
        pass


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
