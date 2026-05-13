"""Fertigungsschnittstellen (Interface Features).

Trennung von Constraint (FEM-Randbedingung) und InterfaceFeature (Geometrie der Schnittstelle):

  Constraint       = kinematische BC (eingespannte Flaeche, Scheibenlager)
  InterfaceFeature = physische Schnittstelle (Bohrung, Passung, Gewindebuchse)

InterfaceFeature liefert:
  - Cut-Primitive-Dicts fuer generate_shape_stl ("mode": "cut")
  - Eingabeparameter fuer ScrewBearingRequirement / PressFitRetentionRequirement

Alle Masse in mm, Kraefte in N, Spannungen in MPa.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ======================================================================
# Enum
# ======================================================================

class InterfaceType(str, Enum):
    SCREW_THROUGH   = "screw_through"   # Durchgangsbohrung (Schraube durch PLA → Untergrund)
    THREADED_INSERT = "threaded_insert" # Gewindebuchse / Helicoil in PLA
    PRESS_FIT       = "press_fit"       # Presspassung (Welle in Nabe)
    DOWEL_PIN       = "dowel_pin"       # Zentrierstift (Formschluss)


# ======================================================================
# Dataclass
# ======================================================================

@dataclass
class InterfaceFeature:
    """Geometrie und Toleranzangaben einer Fertigungsschnittstelle.

    Felder
    ------
    feature_type         : Art der Schnittstelle
    position             : [x, y, z] Mittelpunkt Bohrungsachse [mm]
    diameter_mm          : Nenndurchmesser (Schraubenschaft / Wellendurchmesser)
    depth_mm             : Bohrungstiefe / Eingriffslange [mm]
    axis                 : Bohrungsachse "x" | "y" | "z"
    clearance_mm         : Fertigungszuschlag (Durchgangsbohrung, default 0.3 mm)
    interference_mm      : Uebermas (nur PRESS_FIT, positiv = Presspassung, default 0)
    counterbore_d_mm     : Senkungsdurchmesser fuer Schraubenkopf [mm]
    counterbore_depth_mm : Senkungstiefe [mm]
    hub_outer_d_mm       : Nabenaussendurchmesser (fuer PRESS_FIT Berechnung)
    mu_friction          : Haftreibungszahl (PLA-PLA ca. 0.3)
    thread_spec          : Gewindebezeichnung, z.B. "M6" (informational)
    n_count              : Anzahl gleicher Bohrungen dieses Typs (Lochkreis)
    description          : Freitextbeschreibung
    """

    feature_type:         InterfaceType
    position:             List[float]          # [x, y, z]
    diameter_mm:          float
    depth_mm:             float
    axis:                 str   = "x"          # "x" | "y" | "z"
    clearance_mm:         float = 0.3
    interference_mm:      float = 0.0
    counterbore_d_mm:     Optional[float] = None
    counterbore_depth_mm: Optional[float] = None
    hub_outer_d_mm:       Optional[float] = None
    mu_friction:          float = 0.30
    thread_spec:          str   = ""
    n_count:              int   = 1
    description:          str   = ""

    # ------------------------------------------------------------------
    # Serialisierung
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_type":         str(self.feature_type.value),
            "position":             list(self.position),
            "diameter_mm":          self.diameter_mm,
            "depth_mm":             self.depth_mm,
            "axis":                 self.axis,
            "clearance_mm":         self.clearance_mm,
            "interference_mm":      self.interference_mm,
            "counterbore_d_mm":     self.counterbore_d_mm,
            "counterbore_depth_mm": self.counterbore_depth_mm,
            "hub_outer_d_mm":       self.hub_outer_d_mm,
            "mu_friction":          self.mu_friction,
            "thread_spec":          self.thread_spec,
            "n_count":              self.n_count,
            "description":          self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InterfaceFeature":
        return cls(
            feature_type=InterfaceType(data["feature_type"]),
            position=list(data["position"]),
            diameter_mm=float(data["diameter_mm"]),
            depth_mm=float(data["depth_mm"]),
            axis=str(data.get("axis", "x")),
            clearance_mm=float(data.get("clearance_mm", 0.3)),
            interference_mm=float(data.get("interference_mm", 0.0)),
            counterbore_d_mm=data.get("counterbore_d_mm"),
            counterbore_depth_mm=data.get("counterbore_depth_mm"),
            hub_outer_d_mm=data.get("hub_outer_d_mm"),
            mu_friction=float(data.get("mu_friction", 0.30)),
            thread_spec=str(data.get("thread_spec", "")),
            n_count=int(data.get("n_count", 1)),
            description=str(data.get("description", "")),
        )


# ======================================================================
# Cut-Primitive-Erzeugung
# ======================================================================

def _cylinder_cut_dict(position: List[float], axis: str,
                        diameter_mm: float, depth_mm: float) -> Dict[str, Any]:
    """Erzeugt einen Zylinder-Cut-Primitive-Dict fuer generate_shape_stl."""
    x, y, z = float(position[0]), float(position[1]), float(position[2])
    r = diameter_mm / 2.0
    half = depth_mm / 2.0

    if axis == "x":
        return {
            "type": "cylinder_x",   # spezielle orientierte Variante -- siehe shape.py
            "center_yz": [y, z],
            "x_range":   [x - half, x + half],
            "radius":    r,
            "mode":      "cut",
        }
    if axis == "y":
        return {
            "type": "cylinder_y",
            "center_xz": [x, z],
            "y_range":   [y - half, y + half],
            "radius":    r,
            "mode":      "cut",
        }
    # default: z
    return {
        "type": "cylinder",
        "center_xy": [x, y],
        "z_range":   [z - half, z + half],
        "radius":    r,
        "mode":      "cut",
    }


def interface_to_shapek_cuts(f: InterfaceFeature) -> List[Any]:
    """Wandelt ein InterfaceFeature in ShapeKernel-BaseShape-Schnitte um.

    Gibt eine Liste von BaseShape-Objekten zurueck (CylinderXShape, CylinderYShape
    oder CylinderShape), die als DifferenceShape vom Koerper subtrahiert werden.
    Jedes Objekt hat positives SDF innen (Bohrungszylinder).

    Verwendung::

        from picogk_mp.shapek.base_shape import CompoundShape, DifferenceShape
        body = CompoundShape(plate, arm, ...)
        for cut in interface_to_shapek_cuts(feature):
            body = DifferenceShape(body, cut)
        result = body.mesh_stl(resolution_mm=1.5)
    """
    try:
        from picogk_mp.shapek.base_shape import (  # type: ignore
            CylinderShape, CylinderXShape, CylinderYShape,
        )
    except ImportError as exc:
        raise ImportError(
            "picogk_mp.shapek ist nicht verfuegbar -- "
            "interface_to_shapek_cuts() benoetigt ShapeKernel"
        ) from exc

    cuts: List[Any] = []
    x, y, z = float(f.position[0]), float(f.position[1]), float(f.position[2])

    if f.feature_type == InterfaceType.PRESS_FIT:
        d_bore = f.diameter_mm - f.interference_mm
    else:
        d_bore = f.diameter_mm + f.clearance_mm
    r_bore = d_bore / 2.0
    half   = f.depth_mm / 2.0

    if f.axis == "x":
        cuts.append(CylinderXShape([y, z], [x - half, x + half], r_bore))
    elif f.axis == "y":
        cuts.append(CylinderYShape([x, z], [y - half, y + half], r_bore))
    else:
        cuts.append(CylinderShape([x, y], [z - half, z + half], r_bore))

    # Counterbore (Senkung)
    if f.counterbore_d_mm is not None and f.counterbore_depth_mm is not None:
        r_cb    = float(f.counterbore_d_mm) / 2.0
        half_cb = float(f.counterbore_depth_mm) / 2.0
        if f.axis == "x":
            cb_x0 = x - half + half_cb
            cuts.append(CylinderXShape([y, z], [cb_x0 - half_cb, cb_x0 + half_cb], r_cb))
        elif f.axis == "y":
            cb_y0 = y - half + half_cb
            cuts.append(CylinderYShape([x, z], [cb_y0 - half_cb, cb_y0 + half_cb], r_cb))
        else:
            cb_z0 = z - half + half_cb
            cuts.append(CylinderShape([x, y], [cb_z0 - half_cb, cb_z0 + half_cb], r_cb))

    return cuts


def interface_to_cut_primitives(f: InterfaceFeature) -> List[Dict[str, Any]]:
    """Wandelt ein InterfaceFeature in Cut-Primitive-Dicts fuer generate_shape_stl um.

    Gibt eine Liste von Primitive-Dicts zurueck, jedes mit "mode": "cut".
    Fuer SCREW_THROUGH / THREADED_INSERT / DOWEL_PIN: Bohrungszylinder + optional Senkung.
    Fuer PRESS_FIT: Zylinder mit d - interference (enger als Nennmas).
    """
    cuts: List[Dict[str, Any]] = []

    # Bohrungsdurchmesser bestimmen
    if f.feature_type == InterfaceType.PRESS_FIT:
        d_bore = f.diameter_mm - f.interference_mm   # enger als Welle
    else:
        d_bore = f.diameter_mm + f.clearance_mm      # weiter als Schraube/Stift

    # Hauptbohrung
    cuts.append(_cylinder_cut_dict(f.position, f.axis, d_bore, f.depth_mm))

    # Senkung (Counterbore) fuer Schraubenkopf
    if f.counterbore_d_mm is not None and f.counterbore_depth_mm is not None:
        # Senkung liegt an der Eingangsseite der Bohrung
        x, y, z = float(f.position[0]), float(f.position[1]), float(f.position[2])
        half_main = f.depth_mm / 2.0
        half_cb   = float(f.counterbore_depth_mm) / 2.0

        # Verschiebung zur Eingangsseite: je nach Achse
        if f.axis == "x":
            cb_pos = [x - half_main + half_cb, y, z]
        elif f.axis == "y":
            cb_pos = [x, y - half_main + half_cb, z]
        else:  # z
            cb_pos = [x, y, z - half_main + half_cb]

        cuts.append(_cylinder_cut_dict(
            cb_pos, f.axis,
            float(f.counterbore_d_mm), float(f.counterbore_depth_mm),
        ))

    return cuts


# ======================================================================
# Physikalische Kenngroessen fuer Requirements
# ======================================================================

def screw_bearing_params(features: List[InterfaceFeature]) -> Dict[str, Any]:
    """Extrahiert Lochleibungsparameter aus einer Liste von SCREW-Interfaces.

    Gibt dict mit:
        n_screws    : Gesamtanzahl Schrauben
        screw_d_mm  : kleinster Schraubendurchmesser (massgebend)
        plate_t_mm  : schaetzweise Plattendicke = Bohrungstiefe
    """
    screw_ifaces = [
        f for f in features
        if f.feature_type in (InterfaceType.SCREW_THROUGH, InterfaceType.THREADED_INSERT)
    ]
    if not screw_ifaces:
        return {}

    n_screws   = sum(f.n_count for f in screw_ifaces)
    screw_d_mm = min(f.diameter_mm for f in screw_ifaces)
    plate_t_mm = min(f.depth_mm    for f in screw_ifaces)

    return {
        "n_screws":   n_screws,
        "screw_d_mm": screw_d_mm,
        "plate_t_mm": plate_t_mm,
    }


def press_fit_params(feature: InterfaceFeature) -> Dict[str, Any]:
    """Extrahiert Passfugenparameter aus einem PRESS_FIT-Interface."""
    if feature.feature_type != InterfaceType.PRESS_FIT:
        raise ValueError("Kein PRESS_FIT Interface")
    return {
        "interference_mm":  feature.interference_mm,
        "shaft_d_mm":       feature.diameter_mm,
        "hub_outer_d_mm":   feature.hub_outer_d_mm or (feature.diameter_mm * 2.0),
        "engagement_l_mm":  feature.depth_mm,
        "mu_friction":      feature.mu_friction,
    }
