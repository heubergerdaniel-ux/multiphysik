"""Physics Brief Mapper — uebersetzt PhysicsBrief in Laufzeitobjekte.

Funktionen:
    brief_to_requirements(brief)              -> list[PhysicsRequirement]
    brief_to_interface_primitives(brief)      -> list[dict]
    brief_to_body_shapes(brief, derived)      -> list[BaseShape]
    brief_to_sim_engine(brief)                -> SimEngine
    brief_to_boundary_conditions(brief, ...) -> BoundaryConditions
    brief_to_topopt_kwargs(brief)             -> dict
    suggest_geometry(brief)                   -> dict
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from picogk_mp.physics.brief import (
    ComponentType, ConstraintType, DesignIntent,
    LoadType, PhysicsBrief,
)
from picogk_mp.physics.requirement import (
    BendingRequirement,
    BucklingRequirement,
    DragRequirement,
    PhysicsRequirement,
    PressFitRetentionRequirement,
    ScrewBearingRequirement,
    TensionRequirement,
    TippingRequirement,
    TorsionRequirement,
)
from picogk_mp.physics.interface import (
    InterfaceType as _IT,
    interface_to_cut_primitives,
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

    # Lochleibung: wenn Schrauben-Interfaces vorhanden
    screw_types = (_IT.SCREW_THROUGH, _IT.THREADED_INSERT)
    if any(f.feature_type in screw_types for f in getattr(brief, "interfaces", [])):
        reqs.append(ScrewBearingRequirement())

    # Passfugendruck: wenn Presspassungs-Interfaces vorhanden
    if any(f.feature_type == _IT.PRESS_FIT for f in getattr(brief, "interfaces", [])):
        reqs.append(PressFitRetentionRequirement())

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
# brief_to_interface_primitives
# ======================================================================

def brief_to_interface_primitives(brief: PhysicsBrief) -> List[Dict[str, Any]]:
    """Erzeugt Cut-Primitive-Dicts fuer generate_shape_stl aus allen InterfaceFeatures.

    Gibt eine Liste von Primitive-Dicts zurueck, jedes mit ``"mode": "cut"``.
    Uebergabe direkt an generate_shape_stl:

        solid_prims = [...]          # Koerperprimitiven
        cuts = brief_to_interface_primitives(brief)
        result = generate_shape_stl(solid_prims + cuts, resolution_mm=1.5)
    """
    cuts: List[Dict[str, Any]] = []
    for f in getattr(brief, "interfaces", []):
        cuts.extend(interface_to_cut_primitives(f))
    return cuts


# ======================================================================
# brief_to_body_shapes
# ======================================================================

def brief_to_body_shapes(
    brief: PhysicsBrief,
    derived: Dict[str, Any],
) -> list:
    """Erzeugt mathematisch beschriebene Koerper-Shapes aus Brief + derive()-Ergebnis.

    Gibt eine Liste von BaseShape-Objekten zurueck (PipeShape, RevolveShape, ...).
    Kann direkt in CompoundShape / DifferenceShape weiterverarbeitet werden.

    Entscheidungslogik nach dominantem LoadType
    -------------------------------------------
    FORCE / GRAVITY mit Hebelarm  -> PipeShape mit Momentenverjuengung
                                     r(t) = r_min*(1-t)^(1/3)   (Euler-Bernoulli)
    FORCE / GRAVITY axial         -> CylinderShape (Zugstange)
    TORSION                       -> RevolveShape Hohlzylinder
    PRESSURE                      -> RevolveShape Gewoelbe/Kuppel
    Fallback                      -> PipeShape konstanter Radius

    Parameter
    ---------
    derived : dict aus PhysicsRequirement.derive(brief), mindestens:
        section_r_min_mm  -- Mindestquerschnittsradius [mm]
        (weitere Schlussel werden ignoriert)

    Hinweis
    -------
    ShapeKernel (picogk_mp.shapek) ist optional.  Fehlt das Paket, gibt die
    Funktion eine leere Liste zurueck und schreibt eine Warnung.
    """
    try:
        from picogk_mp.shapek.base_shape import (  # type: ignore
            PipeShape, RevolveShape, CylinderShape,
        )
        from picogk_mp.shapek.modulation import LineModulation  # type: ignore
    except ImportError:
        import warnings
        warnings.warn(
            "picogk_mp.shapek nicht verfuegbar -- brief_to_body_shapes gibt [] zurueck.",
            stacklevel=2,
        )
        return []

    r_min: float = float(derived.get("section_r_min_mm", 10.0))
    r_tip: float = max(r_min * 0.4, 4.0)   # Spitze: 40% von r_min, mind. 4 mm

    load_types = {lc.load_type for lc in brief.load_cases}
    shapes: list = []

    if LoadType.PRESSURE in load_types:
        # Gewoelbe: Rotation eines Kreisbogens um die z-Achse
        profile = [(0.0, r_min), (r_min * 0.7, r_min * 0.7), (r_min, 0.0)]
        shapes.append(RevolveShape(profile=profile, axis="z"))

    elif LoadType.TORSION in load_types:
        # Hohlzylinder: minimiert Schubspannung
        reach = _arm_reach_mm(brief) or (r_min * 10)
        shapes.append(CylinderShape(
            radius_outer=r_min,
            radius_inner=r_min * 0.6,
            length=reach,
        ))

    elif _arm_reach_mm(brief) > 0:
        # Biegearm mit Momentenverjuengung (Euler-Bernoulli)
        # M(t) = F*(1-t)*L  =>  r(t) proportional M^(1/3)
        # r(t=0) = r_min  (volle Einspannung), r(t=1) = r_tip (freies Ende)
        def _radius_fn(t: float) -> float:
            # (1-t)^(1/3) von 1 nach 0; interpoliert zwischen r_min und r_tip
            alpha = (1.0 - t) ** (1.0 / 3.0)
            return r_min * alpha + r_tip * (1.0 - alpha)

        radius_mod = LineModulation(_radius_fn)

        # Polyline aus dem primaeren Lastfall ableiten
        for lc in brief.load_cases:
            if lc.load_type in (LoadType.FORCE, LoadType.GRAVITY) and lc.application_point:
                pt = lc.application_point
                # Arm laeuft vom Ursprung zum Angriffspunkt
                polyline = [(0.0, 0.0, 0.0), (pt[0], pt[1], pt[2])]
                shapes.append(PipeShape(polyline=polyline, radius_mod=radius_mod))
                break

    else:
        # Axiale Zugstange: konstanter Kreisquerschnitt
        for lc in brief.load_cases:
            if lc.load_type in (LoadType.FORCE, LoadType.GRAVITY) and lc.direction:
                d = lc.direction
                length = r_min * 10   # Schaetzlaenge
                shapes.append(PipeShape(
                    polyline=[(0.0, 0.0, 0.0),
                               (d[0]*length, d[1]*length, d[2]*length)],
                    radius_mod=LineModulation(r_min),
                ))
                break

    return shapes


# ======================================================================
# brief_to_sim_engine
# ======================================================================

def brief_to_sim_engine(brief: PhysicsBrief) -> SimEngine:
    """Baut eine SimEngine aus dem Brief.

    Resolver-Params (aus Brief bekannt):
        load_mass_g, yield_mpa, density_g_cm3, infill_pct

    Geometrie-Params (unaufgeloest, via inject() zu befuellen):
        base_r_mm, load_reach_mm, volume_mm3, section_r_mm

    Interface-Params (aus InterfaceFeatures, falls vorhanden):
        n_screws, screw_d_mm, plate_t_mm            -- bei SCREW_THROUGH / THREADED_INSERT
        interference_mm, shaft_d_mm, hub_outer_d_mm,
        engagement_l_mm, axial_force_N, mu_friction  -- bei PRESS_FIT
    """
    from picogk_mp.physics.interface import screw_bearing_params, press_fit_params

    # Primaere Kraft als Betriebslast
    primary_force_N = 0.0
    for lc in brief.load_cases:
        if lc.load_type in (LoadType.FORCE, LoadType.GRAVITY):
            primary_force_N = max(primary_force_N, lc.magnitude)

    load_mass_g = primary_force_N / 9.81 * 1000 if primary_force_N > 0 else 0.0

    resolver: Dict[str, Any] = {
        "load_mass_g":    load_mass_g,
        "yield_mpa":      brief.material.resolved("yield_mpa"),
        "density_g_cm3":  brief.material.resolved("density_g_cm3"),
        "infill_pct":     brief.material.infill_pct,
        "E_mpa":          brief.material.resolved("E_mpa"),
    }

    # Pre-populate screw-bearing params from InterfaceFeatures so they don't
    # need to be injected manually by the caller.
    ifaces = getattr(brief, "interfaces", [])
    screw_types = (_IT.SCREW_THROUGH, _IT.THREADED_INSERT)
    has_screws = any(f.feature_type in screw_types for f in ifaces)
    has_press  = any(f.feature_type == _IT.PRESS_FIT for f in ifaces)

    if has_screws:
        sbp = screw_bearing_params(ifaces)
        if sbp:
            resolver.update(sbp)   # n_screws, screw_d_mm, plate_t_mm

    if has_press:
        pf = next(f for f in ifaces if f.feature_type == _IT.PRESS_FIT)
        pfp = press_fit_params(pf)
        resolver.update(pfp)       # interference_mm, shaft_d_mm, hub_outer_d_mm, ...
        # axial_force_N: use design load if not specified
        if "axial_force_N" not in resolver:
            resolver["axial_force_N"] = primary_force_N

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

    # Schrauben-Lochleibungsparameter registrieren
    if has_screws:
        engine.register(
            Param("n_screws",   "Schraubenanzahl",    unit=""),
            Param("screw_d_mm", "Schraubendurchm.",   unit="mm", lo=1.0),
            Param("plate_t_mm", "Plattendicke",       unit="mm", lo=0.1),
        )

    # Presspassungsparameter registrieren
    if has_press:
        engine.register(
            Param("interference_mm",  "Uebermas",          unit="mm"),
            Param("shaft_d_mm",       "Wellendurchm.",      unit="mm", lo=1.0),
            Param("hub_outer_d_mm",   "Nabenaussend.",      unit="mm", lo=1.0),
            Param("engagement_l_mm",  "Eingriffslaenge",    unit="mm", lo=0.1),
            Param("axial_force_N",    "Axiallast",          unit="N",  lo=0.0),
            Param("mu_friction",      "Haftreibungszahl",   unit=""),
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
