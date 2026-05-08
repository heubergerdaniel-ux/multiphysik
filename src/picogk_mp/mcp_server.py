"""MCP server -- multiphysik design pipeline.

General physics-driven design workflow for any 3D-printed structural part.
Claude acts as the natural-language → physics translator; these tools handle
the compute.

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
generate_shape   Build any geometry from SDF primitives (sphere/box/capsule/cylinder)
run_topopt       BESO topology optimisation -- works on ANY STL with flexible BCs
check_physics    Structural safety-factor checks (tipping + bending) on disc-base parts
render_stl       STL -> inline PNG preview
list_stls        Discover STL files in the project

Workflow (Claude's role)
------------------------
1. Understand the design intent from natural language.
2. Decompose the shape into SDF primitives -> call generate_shape.
3. Identify load point, load direction, and fixture from the geometry
   -> call run_topopt with the correct fixture_type / load_direction.
4. Render the result and report safety factors.
5. If constraints not met, adjust and iterate.

Claude chooses primitives and boundary conditions; the tools handle physics.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastmcp import FastMCP
from fastmcp.utilities.types import Image

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "multiphysik",
    instructions=(
        "Physics-driven generative design pipeline for ANY 3D-printed structural part. "
        "YOU are the natural-language -> physics translator. "
        "\n\nWORKFLOW:\n"
        "1. Decompose geometry into SDF primitives -> generate_shape "
        "(sphere, box, capsule, cylinder) OR generate_shapek "
        "(adds cone, torus, pipe). "
        "2. Identify load point + fixture -> run_topopt "
        "(fixture_type='disc_base'|'face', fixed_face='x0'/'z0'/etc, "
        "load_direction='gravity'|'-x'|'+y'/etc). "
        "3. render_stl to show the result inline. "
        "4. check_physics for disc-base standing parts. "
        "5. measure_shape for volume, mass, CoG, and inertia tensor. "
        "6. generate_lattice for beam-based infill (requires picogk.go). "
        "7. run_cfd_flow for aerodynamic drag (Cd, Re, velocity field). "
        "8. run_cfd_thermal for convective cooling (h_conv, temperature field). "
        "\n\nExamples of what you can build: headphone stand, wall bracket, "
        "shelf arm, hook, drone frame, furniture joint, structural node, "
        "ring/torus structures, tapered pipes, any custom mechanical part. "
        "\n\nAlways render after generate or optimise. "
        "list_stls finds existing files."
    ),
)

# Project root: three levels up from this file
# src/picogk_mp/mcp_server.py -> src/picogk_mp -> src -> project root
_THIS = Path(__file__).resolve()
ROOT  = _THIS.parent.parent.parent
DOCS  = ROOT / "docs"
FIXTURES = ROOT / "tests" / "fixtures"


# ===========================================================================
# Tool 1 -- general shape generator
# ===========================================================================

@mcp.tool()
def generate_shape(
    primitives:    list[dict[str, Any]],
    resolution_mm: float = 1.0,
    out_stl:       Optional[str] = None,
) -> dict:
    """Generate any 3D shape from a list of SDF primitives.

    Builds watertight geometry by taking the SDF union of all primitives
    and extracting a surface with marching cubes.  Works for ANY part:
    headphone stands, wall brackets, hooks, drone frames, furniture joints,
    structural nodes, custom tools -- anything describable with primitives.

    Primitive types (all coordinates in mm)
    ----------------------------------------
    Sphere   : {"type":"sphere",   "center":[x,y,z], "radius":r}
    Box      : {"type":"box",      "min":[x0,y0,z0], "max":[x1,y1,z1]}
    Capsule  : {"type":"capsule",  "from":[x,y,z], "to":[x,y,z],
                                   "radius_from":r0, "radius_to":r1}
               Use radius_from==radius_to for a uniform beam/rod.
    Cylinder : {"type":"cylinder", "center_xy":[cx,cy], "z_range":[z0,z1],
                                   "radius":r}
               Always z-axis aligned (upright).

    Composition tips
    ----------------
    - Overlap primitives generously: SDF union fuses them smoothly at
      any intersection without seams.
    - Add a sphere at every joint/junction for a smooth, organic transition.
    - Coordinate system is yours to choose -- just be consistent when you
      later specify load_point and fixture in run_topopt.

    Parameters
    ----------
    primitives    : List of primitive dicts (see types above).
    resolution_mm : Grid pitch [mm].  1 = good quality (~1-5 s for typical
                    parts).  2 = fast preview (~0.1-0.5 s).  0.5 = fine
                    detail (~10-40 s).
    out_stl       : Output path.  Default: docs/generated_shape.stl.

    Returns
    -------
    dict: status, stl_path, volume_mm3, bounds_min/max ([x,y,z]), elapsed_s.
    bounds_* describe the mesh bounding box -- use them to identify
    sensible load_point coordinates for run_topopt.
    """
    from picogk_mp.generators.shape import generate_shape_stl

    if out_stl:
        out_path = Path(out_stl) if Path(out_stl).is_absolute() else ROOT / out_stl
    else:
        out_path = DOCS / "generated_shape.stl"

    try:
        result = generate_shape_stl(
            primitives    = primitives,
            resolution_mm = resolution_mm,
            out_stl       = str(out_path),
        )
        if result.get("status") != "ok":
            return result

        # Auto-render preview
        png_path = out_path.with_suffix(".png")
        _render_to_file(out_path, png_path)
        result["png_path"] = str(png_path)
        return result

    except Exception:
        return {"status": "error", "message": traceback.format_exc()}


# ===========================================================================
# Tool 2 -- topology optimisation (general)
# ===========================================================================

@mcp.tool()
def run_topopt(
    stl_path: str,
    load_mass_g: float,
    load_point_x_mm: float,
    load_point_y_mm: float,
    load_point_z_mm: float,
    fixture_type: str = "disc_base",
    fixed_face: str = "z0",
    base_radius_mm: float = 48.0,
    load_direction: str = "gravity",
    vol_frac: float = 0.75,
    safety_factor: float = 2.0,
    resolution_mm: float = 3.0,
    max_iter: int = 40,
    out_stl: Optional[str] = None,
) -> dict:
    """Run BESO topology optimisation on any 3D-printed structural part.

    Works with any STL geometry -- not just headphone stands.  Iteratively
    removes low-stress material while preserving structural integrity.

    Parameters
    ----------
    stl_path         : Input STL (absolute or project-relative path).
    load_mass_g      : Operating load [g].  Design load = mass * safety_factor.
    load_point_x_mm  : X of load application point [mm].
    load_point_y_mm  : Y of load application point [mm].
    load_point_z_mm  : Z of load application point [mm].
    fixture_type     : How the part is supported:
                       "disc_base" -- circular disc clamped at z=0.
                                      Use for standing parts (stands, posts).
                       "face"      -- entire face named by fixed_face clamped.
                                      Use for wall brackets, bolted flanges,
                                      glued surfaces, etc.
    fixed_face       : Which face when fixture_type="face":
                       x0=left  x1=right  y0=front  y1=back  z0=floor  z1=ceiling
    base_radius_mm   : Disc radius [mm] -- only for fixture_type="disc_base".
    load_direction   : Direction the load force acts:
                       "gravity"/"-z"=down, "+z"=up,
                       "-x"/"+x"=lateral X, "-y"/"+y"=lateral Y.
    vol_frac         : Fraction of material to keep (0.5=aggressive,
                       0.85=conservative).  Default 0.75.
    safety_factor    : Design-load multiplier applied to load_mass_g.
    resolution_mm    : Voxel pitch for FEM [mm].  3 mm = ~10 s/iter.
    max_iter         : Maximum BESO iterations.
    out_stl          : Output path.  Default: docs/optimised_<stem>.stl.
    """
    from picogk_mp.topopt import TopoptPipeline, BoundaryConditions
    from picogk_mp.topopt.boundary import (
        fixed_face_dofs, fixed_cylinder_base_dofs, point_load_dof,
    )

    # --- resolve paths ---
    stl_in = Path(stl_path) if Path(stl_path).is_absolute() else ROOT / stl_path
    if not stl_in.exists():
        return {"status": "error", "message": f"STL not found: {stl_in}"}

    stem = stl_in.stem
    out_stl_path = Path(out_stl) if out_stl else DOCS / f"optimised_{stem}.stl"
    out_png_path = out_stl_path.with_suffix(".png")
    out_stl_path.parent.mkdir(parents=True, exist_ok=True)

    # --- load direction vector ---
    _dir_map = {
        "gravity": (0.0, 0.0, -1.0), "-z": (0.0, 0.0, -1.0),
        "+z": (0.0, 0.0, 1.0),
        "-x": (-1.0, 0.0, 0.0), "+x": (1.0, 0.0, 0.0),
        "-y": (0.0, -1.0, 0.0), "+y": (0.0, 1.0, 0.0),
    }
    dx, dy, dz = _dir_map.get(load_direction.lower(), (0.0, 0.0, -1.0))

    design_mass_g  = load_mass_g * safety_factor
    design_force_N = design_mass_g * 1e-3 * 9.81
    force_N = (dx * design_force_N, dy * design_force_N, dz * design_force_N)

    try:
        t0 = time.time()

        pipeline = TopoptPipeline(
            stl_in,
            topopt_h_mm=resolution_mm,
            vol_frac=vol_frac,
            max_iter=max_iter,
        )

        Nx, Ny, Nz = pipeline.grid_shape
        h, offset  = pipeline.h, pipeline.offset

        # --- boundary conditions dispatch ---
        ft = fixture_type.lower()
        if ft == "disc_base":
            fixed_dofs = fixed_cylinder_base_dofs(Nx, Ny, Nz, h, offset, base_radius_mm)
        elif ft == "face":
            face_key = fixed_face.lower()
            if face_key not in ("x0", "x1", "y0", "y1", "z0", "z1"):
                return {"status": "error",
                        "message": f"fixed_face must be x0/x1/y0/y1/z0/z1, got '{fixed_face}'"}
            fixed_dofs = fixed_face_dofs(Nx, Ny, Nz, face_key)
        else:
            return {"status": "error",
                    "message": f"fixture_type must be 'disc_base' or 'face', got '{fixture_type}'"}

        force_vec = point_load_dof(
            Nx, Ny, Nz, h, offset,
            position_mm=(load_point_x_mm, load_point_y_mm, load_point_z_mm),
            force_N=force_N,
        )
        bc = BoundaryConditions(fixed_dofs=fixed_dofs, force_vec=force_vec)

        pipeline.run(bc, out_stl=out_stl_path)
        elapsed = time.time() - t0

        final_state = getattr(pipeline, "_final_state", None)
        compliance_history = (
            [round(c, 4) for c in final_state.compliance_history]
            if final_state is not None else []
        )
        elements_initial = int(pipeline.mask.sum())
        elements_final   = int(pipeline._final_mask.sum())
        vol_achieved     = round(elements_final / elements_initial, 3)

        _render_to_file(out_stl_path, out_png_path)

        # physics checks: disc_base only (tipping + stem bending)
        if ft == "disc_base":
            reach = float(np.sqrt(load_point_x_mm**2 + load_point_y_mm**2))
            physics = _physics_checks(
                stl_path=str(out_stl_path),
                load_mass_g=load_mass_g,
                arm_reach_mm=reach,
                base_radius_mm=base_radius_mm,
            )
        else:
            physics = {"note": "generic fixture -- disc-base physics checks not applicable"}

        return {
            "status":          "ok",
            "stl_path":        str(out_stl_path),
            "png_path":        str(out_png_path),
            "elements_initial": elements_initial,
            "elements_final":   elements_final,
            "vol_frac_achieved": vol_achieved,
            "design_force_N":   round(design_force_N, 2),
            "elapsed_s":        round(elapsed, 1),
            "compliance_nm":    compliance_history,
            "physics":          physics,
        }

    except Exception:
        return {"status": "error", "message": traceback.format_exc()}


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
# Tool 3 -- parametric base-design generator
# ===========================================================================

@mcp.tool()
def generate_holder(
    base_radius_mm:      float = 48.0,
    base_height_mm:      float = 14.0,
    stem_height_mm:      float = 234.0,
    stem_radius_base_mm: float = 9.0,
    stem_radius_top_mm:  float = 7.0,
    arm_reach_mm:        float = 82.0,
    arm_tip_z_mm:        float = 244.0,
    arm_radius_mm:       float = 8.5,
    end_cap_radius_mm:   float = 9.0,
    resolution_mm:       float = 1.0,
    out_stl: Optional[str] = None,
) -> dict:
    """Generate a parametric headphone-holder STL from design parameters.

    Builds the full holder geometry (base disc + tapered stem + S-curve arm
    + end cap) via a signed-distance-field union and scikit-image marching
    cubes.  No picogk / external CAD kernel required.

    The returned dict contains arm_tip_x/y/z_mm and base_radius_mm ready
    to pass directly into run_topopt or check_physics.

    Parameters
    ----------
    base_radius_mm      : Base disc radius [mm].  Larger = more stable.
    base_height_mm      : Base disc thickness [mm].
    stem_height_mm      : Height where the arm leaves the stem [mm].
                          Total stand height is roughly stem_height_mm + 20.
    stem_radius_base_mm : Stem radius at the base junction [mm].
    stem_radius_top_mm  : Stem radius at the top junction [mm] (taper).
    arm_reach_mm        : Horizontal reach of the hook tip [mm].
    arm_tip_z_mm        : Z-height of the hook tip [mm].
    arm_radius_mm       : Arm beam radius [mm].
    end_cap_radius_mm   : End-cap sphere radius [mm].
    resolution_mm       : SDF grid pitch [mm].  1 mm is a good balance
                          (~5 s); use 0.5 for a finer mesh (~40 s).
    out_stl             : Output path.  Default: docs/generated_holder.stl
    """
    from picogk_mp.generators.holder import generate_holder_stl

    if out_stl is not None:
        out_path = Path(out_stl) if Path(out_stl).is_absolute() else ROOT / out_stl
    else:
        out_path = DOCS / "generated_holder.stl"

    try:
        result = generate_holder_stl(
            base_radius_mm      = base_radius_mm,
            base_height_mm      = base_height_mm,
            stem_height_mm      = stem_height_mm,
            stem_radius_base_mm = stem_radius_base_mm,
            stem_radius_top_mm  = stem_radius_top_mm,
            arm_reach_mm        = arm_reach_mm,
            arm_tip_z_mm        = arm_tip_z_mm,
            arm_radius_mm       = arm_radius_mm,
            end_cap_radius_mm   = end_cap_radius_mm,
            resolution_mm       = resolution_mm,
            out_stl             = str(out_path),
        )

        # Render preview
        png_path = out_path.with_suffix(".png")
        _render_to_file(out_path, png_path)
        result["png_path"] = str(png_path)

        return result

    except Exception:
        import traceback
        return {"status": "error", "message": traceback.format_exc()}


# ===========================================================================
# Tool 5 -- render STL -> PNG (returned inline)
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
# Tool 6 -- list available STL files
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
# Tool 7 -- shapek extended shape generator
# ===========================================================================

@mcp.tool()
def generate_shapek(
    shapes:        list[dict[str, Any]],
    resolution_mm: float = 1.0,
    out_stl:       Optional[str] = None,
) -> dict:
    """Generate geometry using ShapeKernel-extended primitives (superset of generate_shape).

    Supports all generate_shape primitive types plus:

    Cone   : {"type":"cone",  "apex":[x,y,z], "base":[x,y,z], "r_base":r}
             Solid cone from apex point to circular base.
    Torus  : {"type":"torus", "center":[x,y,z], "major_r":R, "minor_r":r}
             major_r = ring radius, minor_r = tube cross-section radius.
    Pipe   : {"type":"pipe",
              "spine":[[x0,y0,z0],[x1,y1,z1],...],
              "radius": r}             (constant radius)
           OR {"type":"pipe", "spine":[...], "radii":[r0,r1,...]}
             (one radius per spine waypoint, cubic spline interpolated)
             Pipe follows a polyline spine with smoothly modulated radius.

    All coordinates in mm. Builds a watertight STL via marching cubes.

    Parameters
    ----------
    shapes        : List of primitive dicts (see types above).
    resolution_mm : Grid pitch [mm].  1=good quality, 2=fast preview, 0.5=fine.
    out_stl       : Output path.  Default: docs/generated_shape.stl.

    Returns
    -------
    dict: status, stl_path, volume_mm3, bounds_min/max, elapsed_s, png_path.
    """
    import picogk_mp.shapek  # triggers _DISPATCH extension
    from picogk_mp.shapek.base_shape import build_compound_from_spec

    if out_stl:
        out_path = Path(out_stl) if Path(out_stl).is_absolute() else ROOT / out_stl
    else:
        out_path = DOCS / "generated_shape.stl"

    try:
        compound = build_compound_from_spec(shapes)
        result = compound.mesh_stl(resolution_mm=resolution_mm, out_stl=str(out_path))
        if result.get("status") != "ok":
            return result
        png_path = out_path.with_suffix(".png")
        _render_to_file(out_path, png_path)
        result["png_path"] = str(png_path)
        return result
    except Exception:
        return {"status": "error", "message": traceback.format_exc()}


# ===========================================================================
# Tool 8 -- measure_shape
# ===========================================================================

@mcp.tool()
def measure_shape(
    stl_path:             str,
    density_g_cm3:        float = 1.24,
    infill_pct:           float = 20.0,
    report_principal_axes: bool = False,
) -> dict:
    """Measure physical properties of an STL file.

    Computes volume, surface area, centre of gravity, mass, and the
    inertia tensor.  No picogk context required -- works headlessly.

    Parameters
    ----------
    stl_path              : STL to measure (absolute or project-relative).
    density_g_cm3         : Raw filament density [g/cm3].  PLA default: 1.24.
    infill_pct            : Print infill percentage 0-100.  Default: 20.
    report_principal_axes : If True, include eigendecomposition of inertia
                            tensor (principal moments + axes).

    Returns
    -------
    dict: status, volume_mm3, surface_area_mm2, mass_g, cog_mm [x,y,z],
          inertia_tensor_g_mm2 (3x3 as nested list). Optionally:
          principal_moments [I1,I2,I3], principal_axes (3x3 nested list).
    """
    from picogk_mp.shapek.measure import Measure

    stl_p = Path(stl_path) if Path(stl_path).is_absolute() else ROOT / stl_path
    if not stl_p.exists():
        return {"status": "error", "message": f"STL not found: {stl_p}"}

    try:
        m = Measure.from_stl(stl_p, density_g_cm3=density_g_cm3, infill_pct=infill_pct)
        result: dict[str, Any] = {
            "status":              "ok",
            "volume_mm3":          m.volume_mm3,
            "surface_area_mm2":    m.surface_area_mm2,
            "mass_g":              m.mass_g,
            "cog_mm":              [round(float(v), 2) for v in m.center_of_gravity_mm],
            "inertia_tensor_g_mm2": [
                [round(float(v), 4) for v in row]
                for row in m.inertia_tensor_g_mm2
            ],
        }
        if report_principal_axes:
            vals, vecs = Measure.principal_axes(m)
            result["principal_moments"] = [round(float(v), 4) for v in vals]
            result["principal_axes"]    = [
                [round(float(v), 6) for v in col]
                for col in vecs.T.tolist()
            ]
        return result
    except Exception:
        return {"status": "error", "message": traceback.format_exc()}


# ===========================================================================
# Tool 9 -- generate_lattice
# ===========================================================================

@mcp.tool()
def generate_lattice(
    lattice_type:   str,
    bounds_min:     list[float],
    bounds_max:     list[float],
    cell_size_mm:   float,
    strut_radius_mm: float,
    clip_stl:       Optional[str] = None,
    out_stl:        Optional[str] = None,
) -> dict:
    """Generate a beam-based engineering lattice infill (requires picogk.go context).

    Uses picogk.Lattice natively (C++, sub-second for typical densities).
    Much faster than TPMS for filling a bounding box.

    Parameters
    ----------
    lattice_type    : "cubic" (12-edge unit cell). More types planned.
    bounds_min      : [x,y,z] minimum corner of the fill region [mm].
    bounds_max      : [x,y,z] maximum corner of the fill region [mm].
    cell_size_mm    : Unit cell edge length [mm].
    strut_radius_mm : Strut / beam radius [mm].
    clip_stl        : Optional STL path. If supplied, the lattice is boolean-
                      intersected with the shape from that STL (useful for
                      filling an irregular solid with lattice infill).
    out_stl         : Output path. Default: docs/generated_lattice.stl.

    Returns
    -------
    dict: status, stl_path, volume_mm3, node_count, beam_count,
          elapsed_s, png_path.
    """
    from picogk_mp.shapek.lattice import EngineeringLattice

    out_path = (
        Path(out_stl) if out_stl and Path(out_stl).is_absolute()
        else ROOT / out_stl if out_stl
        else DOCS / "generated_lattice.stl"
    )

    try:
        if lattice_type.lower() != "cubic":
            return {
                "status": "error",
                "message": f"lattice_type '{lattice_type}' not yet supported. Use 'cubic'.",
            }

        lat = EngineeringLattice().fill_box(
            bounds_min, bounds_max, cell_size_mm, strut_radius_mm
        )
        result = lat.mesh_stl(out_path)
        if result.get("status") != "ok":
            return result

        # Optional: intersect with clip geometry
        if clip_stl is not None:
            from picogk_mp.csg import intersection
            from picogk import Mesh, Voxels  # type: ignore[import]

            clip_p = Path(clip_stl) if Path(clip_stl).is_absolute() else ROOT / clip_stl
            if not clip_p.exists():
                return {"status": "error", "message": f"clip_stl not found: {clip_p}"}

            lat_vox  = lat.voxelize()
            clip_vox = Voxels.from_mesh(Mesh.mshFromStlFile(str(clip_p)))
            combined = intersection(lat_vox, clip_vox)

            mesh = Mesh.from_voxels(combined)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.SaveToStlFile(str(out_path))
            vol, _ = combined.calculate_properties()
            result["volume_mm3"] = round(float(vol))

        png_path = out_path.with_suffix(".png")
        _render_to_file(out_path, png_path)
        result["png_path"] = str(png_path)
        return result

    except Exception:
        return {"status": "error", "message": traceback.format_exc()}


# ===========================================================================
# Tool 10 -- CFD-A: external flow (drag coefficient)
# ===========================================================================

@mcp.tool()
def run_cfd_flow(
    stl_path:        str,
    velocity_m_s:    float = 0.5,
    flow_direction:  str   = "x",
    resolution_mm:   float = 3.0,
    max_steps:       int   = 3000,
    out_png:         Optional[str] = None,
) -> dict:
    """Compute 2D external flow around an STL part (D2Q9 Lattice-Boltzmann).

    Simulates a 2D cross-section of the part in a free-stream flow and
    returns the drag coefficient Cd, Reynolds number Re, pressure drop,
    and a velocity-field PNG showing streamlines around the shape.

    Parameters
    ----------
    stl_path       : Input STL (absolute or project-relative).
    velocity_m_s   : Free-stream velocity [m/s].  Default 0.5 m/s (gentle desk
                     air current).
    flow_direction : "x", "y", or "z" -- axis aligned with the flow.
    resolution_mm  : Lattice cell size [mm].  3 mm is a good balance of speed
                     and accuracy.  Use 2 mm for finer detail.
    max_steps      : Maximum LBM time steps.  Simulation converges earlier if
                     the velocity field stabilises.
    out_png        : Path for the velocity PNG.  Default: docs/<stem>_flow.png.

    Returns
    -------
    dict: status, Cd, Re, pressure_drop_Pa, velocity_png, Ny, Nx, elapsed_s.
    """
    from picogk_mp.cfd import run_flow
    from picogk_mp.cfd.postprocess import save_velocity_png

    stl_p = Path(stl_path) if Path(stl_path).is_absolute() else ROOT / stl_path
    if not stl_p.exists():
        return {"status": "error", "message": f"STL not found: {stl_p}"}

    png_p = (
        Path(out_png) if out_png
        else DOCS / f"{stl_p.stem}_flow.png"
    )

    try:
        result = run_flow(
            stl_path=stl_p,
            velocity_m_s=velocity_m_s,
            flow_direction=flow_direction,
            resolution_mm=resolution_mm,
            max_steps=max_steps,
        )

        png_path = save_velocity_png(result, png_p)

        # Pressure drop: mean inlet pressure - mean outlet pressure [Pa]
        rho_air = 1.2   # kg/m3
        cs2_phys = (result.domain.U_lb * result.domain.velocity_m_s
                    / result.domain.U_lb) ** 2 / 3.0
        rho_lb = result.rho_lb
        # Simplified: pressure in lattice units ~ rho * cs2
        p_in  = float(rho_lb[:, 1].mean()) / 3.0
        p_out = float(rho_lb[:, -2].mean()) / 3.0
        # Convert lattice pressure to Pascals (approximate)
        U_phys = result.domain.velocity_m_s
        dp_Pa  = (p_in - p_out) * rho_air * U_phys**2

        return {
            "status":           "ok",
            "Cd":               round(result.Cd, 4),
            "Re":               round(result.Re, 1),
            "pressure_drop_Pa": round(dp_Pa, 4),
            "Ny":               result.domain.Ny,
            "Nx":               result.domain.Nx,
            "velocity_png":     str(png_path),
            "elapsed_s":        result.elapsed_s,
        }

    except Exception:
        return {"status": "error", "message": traceback.format_exc()}


# ===========================================================================
# Tool 11 -- CFD-B: convective thermal analysis
# ===========================================================================

@mcp.tool()
def run_cfd_thermal(
    stl_path:       str,
    velocity_m_s:   float = 0.5,
    heat_flux_W_m2: float = 1000.0,
    T_inlet_C:      float = 20.0,
    flow_direction: str   = "x",
    resolution_mm:  float = 3.0,
    max_steps:      int   = 3000,
    out_png:        Optional[str] = None,
) -> dict:
    """Compute convective cooling of an STL part (D2Q9 + D2Q5 LBM).

    First runs an external-flow simulation to obtain the velocity field,
    then solves the heat equation (advection-diffusion) to find the
    temperature distribution and convective heat transfer coefficient.

    Parameters
    ----------
    stl_path       : Input STL.
    velocity_m_s   : Free-stream velocity [m/s].
    heat_flux_W_m2 : Heat flux emitted from the solid surface [W/m2].
    T_inlet_C      : Inlet / ambient air temperature [C].
    flow_direction : "x", "y", or "z".
    resolution_mm  : Lattice cell size [mm].
    max_steps      : Max LBM steps for each solver phase.
    out_png        : Path for the temperature PNG.  Default: docs/<stem>_thermal.png.

    Returns
    -------
    dict: status, h_conv_W_m2K, T_max_C, T_surface_avg_C, Cd, Re,
          temperature_png, elapsed_s.
    """
    from picogk_mp.cfd import run_flow, run_thermal
    from picogk_mp.cfd.postprocess import save_temperature_png

    stl_p = Path(stl_path) if Path(stl_path).is_absolute() else ROOT / stl_path
    if not stl_p.exists():
        return {"status": "error", "message": f"STL not found: {stl_p}"}

    png_p = (
        Path(out_png) if out_png
        else DOCS / f"{stl_p.stem}_thermal.png"
    )

    try:
        flow = run_flow(
            stl_path=stl_p,
            velocity_m_s=velocity_m_s,
            flow_direction=flow_direction,
            resolution_mm=resolution_mm,
            max_steps=max_steps,
        )

        thermal = run_thermal(
            flow_result=flow,
            heat_flux_W_m2=heat_flux_W_m2,
            T_inlet_C=T_inlet_C,
            max_steps=max_steps,
        )

        png_path = save_temperature_png(thermal, png_p)

        return {
            "status":           "ok",
            "h_conv_W_m2K":     round(thermal.h_conv, 2),
            "T_max_C":          round(thermal.T_max, 2),
            "T_surface_avg_C":  round(thermal.T_surface_avg, 2),
            "Cd":               round(flow.Cd, 4),
            "Re":               round(flow.Re, 1),
            "temperature_png":  str(png_path),
            "elapsed_s":        round(flow.elapsed_s + thermal.elapsed_s, 1),
        }

    except Exception:
        return {"status": "error", "message": traceback.format_exc()}


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
