"""Physics Brief Mapper — uebersetzt PhysicsBrief in Laufzeitobjekte.

Funktionen:
    brief_to_requirements(brief)   -> list[PhysicsRequirement]
    brief_to_sim_engine(brief)     -> SimEngine
    brief_to_boundary_conditions(brief, Nx, Ny, Nz, h, offset) -> BoundaryConditions
    brief_to_topopt_kwargs(brief)  -> dict
    suggest_geometry(brief)        -> dict
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Sequence

from picogk_mp.physics.brief import (
    ComponentType, ConstraintType, DesignIntent,
    LoadType, PhysicsBrief,
)
from picogk_mp.physics.requirement import (
    BendingRequirement,
    BucklingRequirement,
    DragRequirement,
    PhysicsRequirement,
    TensionRequirement,
    TippingRequirement,
    TorsionRequirement,
)
from picogk_mp.physics.params import Param
from picogk_mp.physics.engine import SimEngine

if TYPE_CHECKING:
    import numpy as np


# ======================================================================
# brief_to_requirements
# ======================================================================

def brief_to_requirements(brief: PhysicsBrief) -> List[PhysicsRequirement]:
    """Waehlt Requirements anhand von Lastfall und Constraints.

    Auswahllogik:
      BendingRequirement  -- wenn FORCE oder GRAVITY vorhanden
      TippingRequirement  -- nur wenn STAND + FIXED_DISC
      TorsionRequirement  -- wenn TORSION-Lastfall
      BucklingRequirement -- wenn schlanke Stutze (heuristisch L/r > 20)
      DragRequirement     -- wenn FLOW-Lastfall
      TensionRequirement  -- wenn axiale Kraft ohne Biegung (kein Hebelarm)
    """
    reqs: List[PhysicsRequirement] = []

    load_types = {lc.load_type for lc in brief.load_cases}
    constraint_types = {c.constraint_type for c in brief.constraints}
    has_force   = bool(load_types & {LoadType.FORCE, LoadType.GRAVITY})
    has_torsion = LoadType.TORSION in load_types
    has_flow    = LoadType.FLOW in load_types

    # Biegeanforderung: immer wenn Querkraft vorhanden
    if has_force:
        req = BendingRequirement()
        req.sf_required = brief.failure.sf_bending
        reqs.append(req)

    # Kippstabilitaet: nur bei STAND + FIXED_DISC
    if (
        brief.intent.component_type == ComponentType.STAND
        and ConstraintType.FIXED_DISC in constraint_types
    ):
        req = TippingRequirement()
        req.sf_required = brief.failure.sf_tipping
        reqs.append(req)

    # Torsionsanforderung
    if has_torsion:
        req = TorsionRequirement()
        req.sf_required = brief.failure.sf_torsion
        reqs.append(req)

    # Knickstabilitaet: heuristisch bei schlanken Traegern (L/r_est > 20)
    if has_force and _is_slender(brief):
        req = BucklingRequirement()
        req.sf_required = brief.failure.sf_buckling
        reqs.append(req)

    # Stroemungsanforderung (informational)
    if has_flow:
        req = DragRequirement(Cd_warn=brief.failure.max_Cd)
        reqs.append(req)

    # Zuganforderung: wenn Kraft vorhanden aber kein Hebelarm (reine Zugstange)
    if has_force and not _has_bending_arm(brief) and not any(
        isinstance(r, BendingRequirement) for r in reqs
    ):
        req = TensionRequirement()
        req.sf_required = brief.failure.sf_tension
        reqs.append(req)

    return reqs


def _is_slender(brief: PhysicsBrief) -> bool:
    """Heuristik: L/r_est > 20 → Knicken moeglich."""
    reach = _arm_reach_mm(brief)
    if reach <= 0:
        return False
    r_est = 10.0   # konservativer Schaetzwert 10mm Radius
    return (reach / r_est) > 20.0


def _has_bending_arm(brief: PhysicsBrief) -> bool:
    """True wenn mindestens ein FORCE/GRAVITY Lastfall einen Hebelarm hat."""
    return _arm_reach_mm(brief) > 0.0


def _arm_reach_mm(brief: PhysicsBrief) -> float:
    """Horizontaler Abstand des Angriffspunkts vom Ursprung [mm]."""
    for lc in brief.load_cases:
        if lc.load_type in (LoadType.FORCE, LoadType.GRAVITY) and lc.application_point:
            pt = lc.application_point
            return math.sqrt(pt[0]**2 + pt[1]**2)
    return 0.0


# ======================================================================
# brief_to_sim_engine
# ======================================================================

def brief_to_sim_engine(brief: PhysicsBrief) -> SimEngine:
    """Baut eine SimEngine aus dem Brief.

    Resolver-Params (aus Brief bekannt):
        load_mass_g, yield_mpa, density_g_cm3, infill_pct

    Geometrie-Params (unaufgeloest, via inject() zu befuellen):
        base_r_mm, load_reach_mm, volume_mm3, section_r_mm
    """
    # Primaere Kraft als Betriebslast
    primary_force_N = 0.0
    for lc in brief.load_cases:
        if lc.load_type in (LoadType.FORCE, LoadType.GRAVITY):
            primary_force_N = max(primary_force_N, lc.magnitude)

    load_mass_g = primary_force_N / 9.81 * 1000 if primary_force_N > 0 else 0.0

    resolver = {
        "load_mass_g":    load_mass_g,
        "yield_mpa":      brief.material.resolved("yield_mpa"),
        "density_g_cm3":  brief.material.resolved("density_g_cm3"),
        "infill_pct":     brief.material.infill_pct,
        "E_mpa":          brief.material.resolved("E_mpa"),
    }

    engine = SimEngine(resolver=resolver)
    engine.register(
        Param("load_mass_g",        "Betriebslast",      unit="g",   lo=0.0),
        Param("yield_mpa",          "Streckgrenze",      unit="MPa", lo=1.0),
        Param("density_g_cm3",      "Dichte",            unit="g/cm3", lo=0.01),
        Param("infill_pct",         "Infill",            unit="%",   lo=1.0, hi=100.0),
        Param("E_mpa",              "E-Modul",           unit="MPa", lo=1.0),
        # Geometrie -- via inject() befuellen
        Param("base_r_mm",          "Basisradius",       unit="mm"),
        Param("load_reach_mm",      "Lastreichweite",    unit="mm"),
        Param("volume_mm3",         "Volumen",           unit="mm3"),
        Param("section_r_mm",       "Querschnitt r",     unit="mm"),
        Param("buckling_length_mm", "Knicklange",        unit="mm"),
    )

    for req in brief_to_requirements(brief):
        engine.add_check(req)

    return engine


# ======================================================================
# brief_to_boundary_conditions
# ======================================================================

def brief_to_boundary_conditions(
    brief: PhysicsBrief,
    Nx: int, Ny: int, Nz: int,
    h: float,
    offset: Sequence[float],
) -> "np.ndarray":
    """Gibt ein numpy-Array mit festen DOF-Indizes aus den Constraints zurueck.

    Fuer TopoptPipeline: fixed_dofs, force_vec = brief_to_boundary_conditions(...)
    """
    import numpy as np
    from picogk_mp.topopt.boundary import (
        fixed_cylinder_base_dofs,
        fixed_face_dofs,
        point_load_dof,
    )

    fixed_parts = []
    for c in brief.constraints:
        if c.constraint_type == ConstraintType.FIXED_FACE and c.face:
            fixed_parts.append(fixed_face_dofs(Nx, Ny, Nz, face=c.face))
        elif c.constraint_type == ConstraintType.FIXED_DISC and c.disc_radius_mm:
            fixed_parts.append(
                fixed_cylinder_base_dofs(Nx, Ny, Nz, h, offset, radius=c.disc_radius_mm)
            )

    fixed_dofs = np.unique(np.concatenate(fixed_parts)) if fixed_parts else np.array([], dtype=np.int64)

    # Primaeren Lastfall als Punktlast
    total_dofs = 3 * (Nx + 1) * (Ny + 1) * (Nz + 1)
    force_vec = np.zeros(total_dofs)

    for lc in brief.load_cases:
        if lc.load_type in (LoadType.FORCE, LoadType.GRAVITY) and lc.application_point:
            fv = lc.force_vector_N
            force_vec += point_load_dof(
                Nx, Ny, Nz, h, offset,
                position_mm=lc.application_point,
                force_N=fv,
            )
            break   # primaeren Lastfall nehmen

    return fixed_dofs, force_vec


# ======================================================================
# brief_to_topopt_kwargs
# ======================================================================

def brief_to_topopt_kwargs(brief: PhysicsBrief) -> dict:
    """Gibt kwargs fuer TopoptPipeline zurueck: E0, nu, vol_frac."""
    E0  = brief.material.resolved("E_mpa")
    nu  = brief.material.resolved("nu")

    keywords_lower = [kw.lower() for kw in brief.intent.keywords]
    if any(k in kw for kw in keywords_lower for k in ("leichtgewichtig", "lightweight", "leicht")):
        vol_frac = 0.40
    elif any(k in kw for kw in keywords_lower for k in ("steifigkeit", "stiff", "steif", "rigid")):
        vol_frac = 0.85
    else:
        vol_frac = 0.65

    return {"E0": E0, "nu": nu, "vol_frac": vol_frac}


# ======================================================================
# suggest_geometry
# ======================================================================

def suggest_geometry(brief: PhysicsBrief) -> dict:
    """Empfiehlt eine Geometriesprache anhand von Lastfall und Design-Intent.

    Gibt zurueck:
        geometry_class   -- Python-Klasse / Muster (String)
        cross_section    -- Querschnittstyp
        rationale        -- Begruendung
        vol_frac         -- empfohlener Volumenanteil fuer Topopt
        min_wall_mm      -- Mindest-Wandstaerke
        max_overhang_deg -- Ueberhang-Limit fuer Druck
    """
    load_types = {lc.load_type for lc in brief.load_cases}
    kws = [kw.lower() for kw in brief.intent.keywords]
    topopt_kw = brief_to_topopt_kwargs(brief)
    vol_frac  = topopt_kw["vol_frac"]

    # Querschnitt / Geometrieempfehlung nach dominantem Lastfall
    if LoadType.MOMENT in load_types:
        geometry_class = "BoxShape (Gurt + Steg) oder I-Profil"
        cross_section  = "I-Profil oder Kastenprofil"
        rationale      = "Biegemoment -> maximales Widerstandsmoment bei minimaler Masse"
    elif LoadType.TORSION in load_types:
        geometry_class = "CylinderShape (geschlossener Hohlquerschnitt)"
        cross_section  = "Kreishohlquerschnitt"
        rationale      = "Torsion -> geschlossener Querschnitt minimiert Schubspannung"
    elif LoadType.PRESSURE in load_types:
        geometry_class = "RevolveShape (Gewolbe / Kuppel)"
        cross_section  = "Gewoelbte Schale"
        rationale      = "Drucklast -> Gewoelbe leitet Kraft in Druckkraefte um (keine Biegung)"
    elif LoadType.FLOW in load_types:
        geometry_class = "SDF-Streamlined (Tropfenform)"
        cross_section  = "Tropfen / NACA-Profil"
        rationale      = "Stroemungslast -> minimaler Cd durch Stromlinienform"
    else:
        # FORCE oder GRAVITY: Kragarm oder Zugstange
        if _arm_reach_mm(brief) > 0:
            geometry_class = "CapsuleShape oder BoxShape (Kragarm)"
            cross_section  = "Kreisquerschnitt oder Kastenprofil"
            rationale      = "Querkraft mit Hebelarm -> Biege-optimierter Kragarm"
        else:
            geometry_class = "CylinderShape (Zugstange)"
            cross_section  = "Kreisquerschnitt"
            rationale      = "Axiale Kraft ohne Hebelarm -> minimaler Querschnitt A_min"

    # TPMS-Hinweis bei "leichtgewichtig"
    tpms_hint = ""
    if any(k in kw for kw in kws for k in ("leichtgewichtig", "lightweight", "leicht")):
        tpms_hint = " + Gyroid TPMS-Infill empfohlen"

    return {
        "geometry_class":    geometry_class + tpms_hint,
        "cross_section":     cross_section,
        "rationale":         rationale,
        "vol_frac":          vol_frac,
        "min_wall_mm":       brief.failure.min_wall_thickness_mm,
        "max_overhang_deg":  brief.failure.max_overhang_deg,
    }
