"""Physics Brief — strukturiertes Anforderungsdokument fuer den Physics-First Workflow.

Workflow-Ablauf:
  1. Claude fuellt PhysicsBrief aus natuerlichsprachlichem Prompt
  2. brief_to_requirements(brief) leitet Mindestgeometrie analytisch her
  3. Geometrie-Tools bauen das Mesh
  4. SimEngine.inject(geometry_params).run() verifiziert das fertige Mesh

Keine externen Abhaengigkeiten ausser stdlib.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ======================================================================
# Enums (str, Enum → JSON-sicher ohne .value)
# ======================================================================

class MaterialPreset(str, Enum):
    PLA    = "PLA"
    PETG   = "PETG"
    ABS    = "ABS"
    PA12   = "PA12"
    TPU    = "TPU"
    CUSTOM = "CUSTOM"


class LoadType(str, Enum):
    FORCE       = "force"
    MOMENT      = "moment"
    TORSION     = "torsion"      # Torsionsmoment [N*mm]
    PRESSURE    = "pressure"
    TEMPERATURE = "temperature"
    GRAVITY     = "gravity"
    FLOW        = "flow"         # Stroemungslast → aktiviert DragRequirement


class LoadCombination(str, Enum):
    AND = "AND"
    OR  = "OR"


class ConstraintType(str, Enum):
    FIXED_FACE     = "fixed_face"
    FIXED_DISC     = "fixed_disc"
    HINGE          = "hinge"
    SYMMETRY_PLANE = "symmetry_plane"
    SLIDING_GUIDE  = "sliding_guide"
    BOLT_PATTERN   = "bolt_pattern"
    PRESS_FIT      = "press_fit"
    ADHESIVE       = "adhesive"


class GeometryLanguage(str, Enum):
    PRIMITIVES   = "primitives"
    SDF_ORGANIC  = "sdf_organic"
    TPMS_INFILL  = "tpms_infill"
    LATTICE      = "lattice"
    HYBRID       = "hybrid"


class ComponentType(str, Enum):
    BRACKET   = "bracket"
    HOUSING   = "housing"
    BEAM      = "beam"
    CLAMP     = "clamp"
    CONNECTOR = "connector"
    PIPE      = "pipe"
    STAND     = "stand"
    FRAME     = "frame"
    CUSTOM    = "custom"


# ======================================================================
# Material-Presets
# ======================================================================

_MATERIAL_PRESETS: Dict[str, Dict[str, float]] = {
    "PLA":    {"E_mpa": 3500, "nu": 0.36, "density_g_cm3": 1.24, "yield_mpa": 55,  "thermal_k_W_mK": 0.13},
    "PETG":   {"E_mpa": 2100, "nu": 0.38, "density_g_cm3": 1.27, "yield_mpa": 50,  "thermal_k_W_mK": 0.20},
    "ABS":    {"E_mpa": 2300, "nu": 0.35, "density_g_cm3": 1.05, "yield_mpa": 45,  "thermal_k_W_mK": 0.17},
    "PA12":   {"E_mpa": 1600, "nu": 0.40, "density_g_cm3": 1.01, "yield_mpa": 48,  "thermal_k_W_mK": 0.23},
    "TPU":    {"E_mpa":   50, "nu": 0.49, "density_g_cm3": 1.21, "yield_mpa": 30,  "thermal_k_W_mK": 0.25},
    "CUSTOM": {"E_mpa": 3500, "nu": 0.36, "density_g_cm3": 1.24, "yield_mpa": 55,  "thermal_k_W_mK": 0.13},
}


# ======================================================================
# Material
# ======================================================================

@dataclass
class Material:
    preset: MaterialPreset = MaterialPreset.PLA
    infill_pct: float = 20.0

    # Optionale Overrides (None = Preset-Wert verwenden)
    E_mpa:           Optional[float] = None
    nu:              Optional[float] = None
    density_g_cm3:   Optional[float] = None
    yield_mpa:       Optional[float] = None
    thermal_k_W_mK:  Optional[float] = None

    def resolved(self, key: str) -> float:
        """Gibt den expliziten Override-Wert oder den Preset-Wert zurueck."""
        explicit = getattr(self, key, None)
        if explicit is not None:
            return float(explicit)
        preset_data = _MATERIAL_PRESETS.get(self.preset.value if hasattr(self.preset, "value") else str(self.preset), {})
        if key not in preset_data:
            raise KeyError(f"Material.resolved: unbekannter Schlussel '{key}'")
        return float(preset_data[key])

    def effective_density_g_cm3(self) -> float:
        """Effektive Dichte unter Beruecksichtigung des Infill-Anteils."""
        return self.resolved("density_g_cm3") * self.infill_pct / 100.0

    def to_dict(self) -> dict:
        return {
            "preset":         str(self.preset.value if hasattr(self.preset, "value") else self.preset),
            "infill_pct":     self.infill_pct,
            "E_mpa":          self.E_mpa,
            "nu":             self.nu,
            "density_g_cm3":  self.density_g_cm3,
            "yield_mpa":      self.yield_mpa,
            "thermal_k_W_mK": self.thermal_k_W_mK,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Material":
        return cls(
            preset=MaterialPreset(data.get("preset", "PLA")),
            infill_pct=float(data.get("infill_pct", 20.0)),
            E_mpa=data.get("E_mpa"),
            nu=data.get("nu"),
            density_g_cm3=data.get("density_g_cm3"),
            yield_mpa=data.get("yield_mpa"),
            thermal_k_W_mK=data.get("thermal_k_W_mK"),
        )


# ======================================================================
# LoadCase
# ======================================================================

@dataclass
class LoadCase:
    load_type: LoadType
    magnitude: float                             # [N | N*mm | MPa | K | g]
    direction: Optional[List[float]] = None      # [dx, dy, dz]
    application_point: Optional[List[float]] = None  # [x, y, z] mm
    sf_static:   float = 2.0
    sf_dynamic:  float = 2.0
    description: str = ""

    @property
    def design_magnitude(self) -> float:
        """Bemessungslast = Betriebslast * Sicherheitsfaktor (statisch)."""
        return self.magnitude * self.sf_static

    @property
    def force_vector_N(self) -> List[float]:
        """Normierter Kraftvektor * Bemessungslast [N].
        Nur sinnvoll fuer FORCE und GRAVITY.
        """
        if self.direction is None:
            return [0.0, 0.0, -self.design_magnitude]
        norm = math.sqrt(sum(d**2 for d in self.direction))
        if norm == 0:
            return [0.0, 0.0, -self.design_magnitude]
        return [d / norm * self.design_magnitude for d in self.direction]

    def to_dict(self) -> dict:
        return {
            "load_type":         str(self.load_type.value if hasattr(self.load_type, "value") else self.load_type),
            "magnitude":         self.magnitude,
            "direction":         self.direction,
            "application_point": self.application_point,
            "sf_static":         self.sf_static,
            "sf_dynamic":        self.sf_dynamic,
            "description":       self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LoadCase":
        return cls(
            load_type=LoadType(data["load_type"]),
            magnitude=float(data["magnitude"]),
            direction=data.get("direction"),
            application_point=data.get("application_point"),
            sf_static=float(data.get("sf_static", 2.0)),
            sf_dynamic=float(data.get("sf_dynamic", 2.0)),
            description=data.get("description", ""),
        )


# ======================================================================
# Constraint
# ======================================================================

@dataclass
class Constraint:
    constraint_type: ConstraintType
    face:              Optional[str] = None         # "x0"|"x1"|"y0"|"y1"|"z0"|"z1"
    disc_radius_mm:    Optional[float] = None       # fuer FIXED_DISC
    bolt_positions:    Optional[List[List[float]]] = None
    bounding_box_min_mm: Optional[List[float]] = None
    bounding_box_max_mm: Optional[List[float]] = None
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "constraint_type":    str(self.constraint_type.value if hasattr(self.constraint_type, "value") else self.constraint_type),
            "face":               self.face,
            "disc_radius_mm":     self.disc_radius_mm,
            "bolt_positions":     self.bolt_positions,
            "bounding_box_min_mm": self.bounding_box_min_mm,
            "bounding_box_max_mm": self.bounding_box_max_mm,
            "description":        self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Constraint":
        return cls(
            constraint_type=ConstraintType(data["constraint_type"]),
            face=data.get("face"),
            disc_radius_mm=data.get("disc_radius_mm"),
            bolt_positions=data.get("bolt_positions"),
            bounding_box_min_mm=data.get("bounding_box_min_mm"),
            bounding_box_max_mm=data.get("bounding_box_max_mm"),
            description=data.get("description", ""),
        )


# ======================================================================
# FailureCriteria
# ======================================================================

@dataclass
class FailureCriteria:
    sf_fracture:          float = 5.0
    sf_tipping:           float = 1.5
    sf_bending:           float = 3.0
    sf_torsion:           float = 3.0
    sf_buckling:          float = 3.0
    sf_tension:           float = 2.0
    max_Cd:               float = 2.0   # Drag-Warnschwelle (informational)
    max_deformation_mm:   Optional[float] = None
    min_wall_thickness_mm: float = 1.2
    max_overhang_deg:     float = 45.0
    min_eigenfreq_hz:     Optional[float] = None
    max_temperature_c:    Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "sf_fracture":          self.sf_fracture,
            "sf_tipping":           self.sf_tipping,
            "sf_bending":           self.sf_bending,
            "sf_torsion":           self.sf_torsion,
            "sf_buckling":          self.sf_buckling,
            "sf_tension":           self.sf_tension,
            "max_Cd":               self.max_Cd,
            "max_deformation_mm":   self.max_deformation_mm,
            "min_wall_thickness_mm": self.min_wall_thickness_mm,
            "max_overhang_deg":     self.max_overhang_deg,
            "min_eigenfreq_hz":     self.min_eigenfreq_hz,
            "max_temperature_c":    self.max_temperature_c,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FailureCriteria":
        return cls(
            sf_fracture=float(data.get("sf_fracture", 5.0)),
            sf_tipping=float(data.get("sf_tipping", 1.5)),
            sf_bending=float(data.get("sf_bending", 3.0)),
            sf_torsion=float(data.get("sf_torsion", 3.0)),
            sf_buckling=float(data.get("sf_buckling", 3.0)),
            sf_tension=float(data.get("sf_tension", 2.0)),
            max_Cd=float(data.get("max_Cd", 2.0)),
            max_deformation_mm=data.get("max_deformation_mm"),
            min_wall_thickness_mm=float(data.get("min_wall_thickness_mm", 1.2)),
            max_overhang_deg=float(data.get("max_overhang_deg", 45.0)),
            min_eigenfreq_hz=data.get("min_eigenfreq_hz"),
            max_temperature_c=data.get("max_temperature_c"),
        )


# ======================================================================
# DesignIntent
# ======================================================================

@dataclass
class DesignIntent:
    component_type:    ComponentType    = ComponentType.CUSTOM
    keywords:          List[str]        = field(default_factory=list)
    geometry_language: GeometryLanguage = GeometryLanguage.PRIMITIVES
    notes:             str              = ""

    def to_dict(self) -> dict:
        return {
            "component_type":    str(self.component_type.value if hasattr(self.component_type, "value") else self.component_type),
            "keywords":          list(self.keywords),
            "geometry_language": str(self.geometry_language.value if hasattr(self.geometry_language, "value") else self.geometry_language),
            "notes":             self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DesignIntent":
        return cls(
            component_type=ComponentType(data.get("component_type", "custom")),
            keywords=list(data.get("keywords", [])),
            geometry_language=GeometryLanguage(data.get("geometry_language", "primitives")),
            notes=data.get("notes", ""),
        )


# ======================================================================
# PhysicsBrief
# ======================================================================

@dataclass
class PhysicsBrief:
    """Strukturiertes Anforderungsdokument — Wurzel des Physics-First Workflows.

    Wird von Claude aus dem natuerlichsprachlichen Prompt ausgefuellt,
    bevor irgendeine Geometrie erzeugt wird.
    """

    source_prompt:    str
    material:         Material
    load_cases:       List[LoadCase]
    constraints:      List[Constraint]
    load_combination: LoadCombination = LoadCombination.AND
    failure:          FailureCriteria = field(default_factory=FailureCriteria)
    intent:           DesignIntent    = field(default_factory=DesignIntent)
    interfaces:       "List[Any]"     = field(default_factory=list)  # List[InterfaceFeature]
    brief_id:         str             = field(default_factory=lambda: uuid.uuid4().hex[:8])

    # ------------------------------------------------------------------
    # Validierung
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Gibt eine Liste von Fehlern zurueck. Leer = gueltig."""
        errors: List[str] = []

        # LoadCases
        if not self.load_cases:
            errors.append("load_cases darf nicht leer sein")
        for i, lc in enumerate(self.load_cases):
            if lc.magnitude <= 0:
                errors.append(f"load_cases[{i}].magnitude muss > 0 sein (ist {lc.magnitude})")
            if lc.load_type == LoadType.FORCE and lc.direction is None:
                errors.append(f"load_cases[{i}]: LoadType.FORCE benoetigt direction")

        # Constraints
        if not self.constraints:
            errors.append("constraints darf nicht leer sein")
        for i, c in enumerate(self.constraints):
            if c.constraint_type == ConstraintType.FIXED_DISC and c.disc_radius_mm is None:
                errors.append(f"constraints[{i}]: FIXED_DISC benoetigt disc_radius_mm")
            if c.constraint_type == ConstraintType.FIXED_FACE and c.face is None:
                errors.append(f"constraints[{i}]: FIXED_FACE benoetigt face")

        # Material
        try:
            self.material.resolved("E_mpa")
        except (KeyError, TypeError) as exc:
            errors.append(f"Material: {exc}")

        # FailureCriteria
        if self.failure.sf_tipping < 1.0:
            errors.append(f"failure.sf_tipping muss >= 1.0 sein (ist {self.failure.sf_tipping})")
        if self.failure.sf_bending < 1.0:
            errors.append(f"failure.sf_bending muss >= 1.0 sein (ist {self.failure.sf_bending})")

        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0

    # ------------------------------------------------------------------
    # Serialisierung
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        from picogk_mp.physics.interface import InterfaceFeature as _IF
        return {
            "brief_id":        self.brief_id,
            "source_prompt":   self.source_prompt,
            "load_combination": str(self.load_combination.value if hasattr(self.load_combination, "value") else self.load_combination),
            "material":        self.material.to_dict(),
            "load_cases":      [lc.to_dict() for lc in self.load_cases],
            "constraints":     [c.to_dict() for c in self.constraints],
            "failure":         self.failure.to_dict(),
            "intent":          self.intent.to_dict(),
            "interfaces":      [f.to_dict() if isinstance(f, _IF) else f for f in self.interfaces],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhysicsBrief":
        from picogk_mp.physics.interface import InterfaceFeature as _IF
        ifaces = [_IF.from_dict(f) for f in data.get("interfaces", [])]
        obj = cls(
            source_prompt=data.get("source_prompt", ""),
            material=Material.from_dict(data.get("material", {})),
            load_cases=[LoadCase.from_dict(lc) for lc in data.get("load_cases", [])],
            constraints=[Constraint.from_dict(c) for c in data.get("constraints", [])],
            load_combination=LoadCombination(data.get("load_combination", "AND")),
            failure=FailureCriteria.from_dict(data.get("failure", {})),
            intent=DesignIntent.from_dict(data.get("intent", {})),
            interfaces=ifaces,
        )
        if "brief_id" in data:
            obj.brief_id = data["brief_id"]
        return obj
