"""Physics Requirements — bidirektionale Anforderungsklassen.

Eine Anforderung (PhysicsRequirement) arbeitet in zwei Richtungen:

  derive(brief) -> dict
      VOR der Geometrie: leitet Mindestgeometrieparameter analytisch her.
      Beispiel: BendingRequirement.derive() gibt {"section_r_min_mm": 6.8}.
      Claude nutzt diese Werte beim Aufbau der Geometrie.

  verify(ctx) -> RequirementResult
      NACH der Geometrie: prueft ob das konkrete Mesh die Anforderung erfuellt.
      ctx ist ein dict mit aufgeloesten Geometrieparametern (volume_mm3, etc.).

Keine externen Abhaengigkeiten ausser math.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from picogk_mp.physics.brief import PhysicsBrief, LoadType


# ======================================================================
# Ergebnis-Container
# ======================================================================

@dataclass
class RequirementResult:
    """Ergebnis einer einzelnen Anforderungspruefung."""

    name: str
    passed: bool
    sf: float           # aktueller Sicherheitsfaktor
    sf_required: float  # Mindest-SF (0.0 = informational)
    detail: str         # menschenlesbare Zusammenfassung

    def __str__(self) -> str:
        status = "OK  " if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.detail}"


# ======================================================================
# Basisklasse
# ======================================================================

class PhysicsRequirement:
    """Abstrakte Basisklasse fuer alle physikalischen Anforderungen.

    Unterklassen implementieren:
      derive(brief)  -- analytische Geometriegrenze VOR der Geometrie
      verify(ctx)    -- Nachpruefung AM fertigen Mesh
    """

    name: str = "PhysicsRequirement"
    sf_required: float = 1.0
    required_params: Tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Vorwaerts-Richtung (Physics → Geometrie)
    # ------------------------------------------------------------------

    def derive(self, brief: "PhysicsBrief") -> dict:
        """Leitet Mindestgeometrieparameter aus dem Brief analytisch her.

        Gibt ein dict mit Schluesseln wie:
          section_r_min_mm   -- Mindest-Querschnittsradius [mm]
          base_r_min_mm      -- Mindest-Basisradius [mm]
          section_area_min_mm2 -- Mindest-Querschnittsflaeche [mm2]

        Gibt leeres dict zurueck wenn keine analytische Ableitung moeglich.
        """
        return {}

    # ------------------------------------------------------------------
    # Rueckwaerts-Richtung (Geometrie → Physics)
    # ------------------------------------------------------------------

    def verify(self, ctx: Dict[str, Any]) -> RequirementResult:
        """Prueft ob das Mesh die Anforderung erfuellt.

        ctx ist ein dict mit aufgeloesten Parametern (z.B. aus SimEngine).
        Wirft KeyError wenn ein required_param fehlt.
        """
        raise NotImplementedError

    # evaluate() is a backward-compat alias for verify() used by SimEngine and older tests.
    def evaluate(self, ctx: Dict[str, Any]) -> RequirementResult:
        return self.verify(ctx)

    def _verify_ctx(self, ctx: Dict[str, Any]) -> None:
        missing = [k for k in self.required_params if k not in ctx]
        if missing:
            raise KeyError(
                f"Requirement '{self.name}' benoetigt fehlende Parameter: {missing}"
            )

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    @staticmethod
    def _primary_force_N(brief: "PhysicsBrief") -> float:
        """Groesste Bemessungskraft [N] aus allen FORCE-Lastfaellen."""
        from picogk_mp.physics.brief import LoadType as LT
        forces = [
            lc.design_magnitude
            for lc in brief.load_cases
            if lc.load_type in (LT.FORCE, LT.GRAVITY)
        ]
        if not forces:
            return 0.0
        return max(forces)

    @staticmethod
    def _primary_moment_Nmm(brief: "PhysicsBrief") -> float:
        """Groesstes Bemessungsmoment [N*mm] aus allen MOMENT/TORSION-Lastfaellen."""
        from picogk_mp.physics.brief import LoadType as LT
        moments = [
            lc.design_magnitude
            for lc in brief.load_cases
            if lc.load_type in (LT.MOMENT, LT.TORSION)
        ]
        if not moments:
            return 0.0
        return max(moments)

    @staticmethod
    def _arm_reach_mm(brief: "PhysicsBrief") -> float:
        """Hebelarm: Abstand Angriffspunkt → Einspannung [mm].
        Approximation: horizontaler Abstand des Angriffspunkts vom Ursprung.
        """
        from picogk_mp.physics.brief import LoadType as LT
        for lc in brief.load_cases:
            if lc.load_type in (LT.FORCE, LT.GRAVITY) and lc.application_point:
                pt = lc.application_point
                return math.sqrt(pt[0]**2 + pt[1]**2)
        return 0.0


# ======================================================================
# BendingRequirement
# ======================================================================

class BendingRequirement(PhysicsRequirement):
    """Biegespannungs-Anforderung an einem runden Querschnitt.

    derive: r_min = cuberoot(4 * M_design / (pi * SF * sigma_yield))
    verify: SF = sigma_yield / sigma, sigma = M * r / I
    """

    name = "Biegeanforderung"
    sf_required = 3.0
    required_params = ("load_mass_g", "load_reach_mm", "section_r_mm", "yield_mpa")

    def derive(self, brief: "PhysicsBrief") -> dict:
        F_N = self._primary_force_N(brief)
        if F_N <= 0:
            return {}
        arm_mm = self._arm_reach_mm(brief)
        if arm_mm <= 0:
            return {}
        sig_y = brief.material.resolved("yield_mpa")
        M_Nmm = F_N * arm_mm                     # [N*mm]
        # sigma = M * r / I = M * r / (pi*r^4/4) = 4*M / (pi*r^3)
        # r_min^3 = 4 * M * SF / (pi * sigma_yield)
        r_min = (4.0 * M_Nmm * self.sf_required / (math.pi * sig_y)) ** (1.0 / 3.0)
        return {"section_r_min_mm": round(r_min, 2)}

    def verify(self, ctx: Dict[str, Any]) -> RequirementResult:
        self._verify_ctx(ctx)
        F_N   = ctx["load_mass_g"] * 9.81e-3       # N
        arm_m = ctx["load_reach_mm"] * 1e-3        # m
        r_m   = ctx["section_r_mm"] * 1e-3         # m
        sig_y = ctx["yield_mpa"]

        M_Nm  = F_N * arm_m
        I_m4  = math.pi * r_m**4 / 4.0
        sigma = M_Nm * r_m / I_m4 / 1e6 if I_m4 > 0 else float("inf")
        sf    = sig_y / sigma if sigma > 0 else float("inf")

        detail = (
            f"SF={sf:.1f} (Mindest {self.sf_required})  |  "
            f"Moment {M_Nm*1000:.1f} N*mm  "
            f"Spannung {sigma:.2f} MPa  "
            f"Streckgrenze {sig_y} MPa"
        )
        return RequirementResult(self.name, sf >= self.sf_required, sf, self.sf_required, detail)


# ======================================================================
# TippingRequirement  (nur fuer STAND + FIXED_DISC)
# ======================================================================

class TippingRequirement(PhysicsRequirement):
    """Kippstabilitaets-Anforderung fuer stehende Teile mit Scheibenbasis.

    Nur aktivieren wenn ComponentType.STAND und ConstraintType.FIXED_DISC.

    derive: base_r_min = load_mass * SF * reach / (part_mass + load_mass * SF)
            (part_mass wird konservativ aus Volumen + Material geschaetzt)
    verify: SF = restoring_moment / tipping_moment
    """

    name = "Kippstabilitaet"
    sf_required = 1.5
    required_params = (
        "load_mass_g", "base_r_mm", "load_reach_mm",
        "volume_mm3", "infill_pct", "density_g_cm3",
    )

    def derive(self, brief: "PhysicsBrief") -> dict:
        F_N = self._primary_force_N(brief)
        if F_N <= 0:
            return {}
        load_g   = F_N / 9.81 * 1000            # N → g
        reach_mm = self._arm_reach_mm(brief)
        if reach_mm <= 0:
            return {}
        rho_eff  = brief.material.effective_density_g_cm3()  # g/cm3
        # Konservative Schaetzung: Volumen ~ Zylinder h=300, r=40mm
        vol_est_mm3 = math.pi * 40**2 * 300
        part_g      = vol_est_mm3 * rho_eff / 1000
        SF          = self.sf_required
        base_r_min  = (load_g * SF * reach_mm) / (part_g + load_g * SF)
        return {"base_r_min_mm": round(base_r_min, 1)}

    def verify(self, ctx: Dict[str, Any]) -> RequirementResult:
        self._verify_ctx(ctx)
        load_g  = ctx["load_mass_g"]
        base_r  = ctx["base_r_mm"]
        reach   = ctx["load_reach_mm"]
        vol     = ctx["volume_mm3"]
        infill  = ctx["infill_pct"] / 100.0
        rho     = ctx["density_g_cm3"] * infill
        part_g  = vol * rho / 1000.0

        lever_load  = reach - base_r
        lever_stand = base_r

        if lever_load <= 0:
            return RequirementResult(
                self.name, True, 999.0, self.sf_required,
                f"Last innerhalb Basis (reach={reach:.0f} <= base_r={base_r:.0f}) -- kein Kipprisiko",
            )

        tip_moment     = load_g * lever_load
        restore_moment = part_g * lever_stand
        sf = restore_moment / tip_moment if tip_moment > 0 else float("inf")

        min_r = (load_g * self.sf_required * reach) / (part_g + load_g * self.sf_required)
        detail = (
            f"SF={sf:.2f} (Mindest {self.sf_required})  |  "
            f"Kippmoment {tip_moment:.0f} g*mm  "
            f"Gegenmoment {restore_moment:.0f} g*mm  "
            f"Koerpermasse {part_g:.1f} g"
        )
        if not sf >= self.sf_required:
            detail += f"  |  --> min. Basisradius: {min_r:.0f} mm"

        return RequirementResult(self.name, sf >= self.sf_required, sf, self.sf_required, detail)


# ======================================================================
# TorsionRequirement
# ======================================================================

class TorsionRequirement(PhysicsRequirement):
    """Torsionsspannungs-Anforderung an einem runden Vollquerschnitt.

    Ip = pi * r^4 / 2  (polares Flaechentraegheitsmoment)
    tau = T * r / Ip
    SF  = tau_allow / tau = (yield_mpa / sqrt(3)) / tau  (von Mises)

    derive: r_min = (2 * T * SF / (pi * tau_allow))^(1/4)
    """

    name = "Torsionsanforderung"
    sf_required = 3.0
    required_params = ("torsion_moment_Nmm", "section_r_mm", "yield_mpa")

    def derive(self, brief: "PhysicsBrief") -> dict:
        T_Nmm = self._primary_moment_Nmm(brief)
        if T_Nmm <= 0:
            return {}
        sig_y   = brief.material.resolved("yield_mpa")
        tau_all = sig_y / math.sqrt(3.0)        # von Mises Scherfestigkeit
        # tau = T * r / Ip = T * r / (pi*r^4/2) = 2*T / (pi*r^3)
        # r_min^3 = 2 * T * SF / (pi * tau_allow)
        r_min = (2.0 * T_Nmm * self.sf_required / (math.pi * tau_all)) ** (1.0 / 3.0)
        return {"section_r_min_mm": round(r_min, 2)}

    def verify(self, ctx: Dict[str, Any]) -> RequirementResult:
        self._verify_ctx(ctx)
        T_Nmm   = ctx["torsion_moment_Nmm"]
        r_mm    = ctx["section_r_mm"]
        sig_y   = ctx["yield_mpa"]
        tau_all = sig_y / math.sqrt(3.0)

        r_m  = r_mm * 1e-3
        T_Nm = T_Nmm * 1e-3
        Ip   = math.pi * r_m**4 / 2.0
        tau  = T_Nm * r_m / Ip / 1e6 if Ip > 0 else float("inf")  # MPa
        sf   = tau_all / tau if tau > 0 else float("inf")

        detail = (
            f"SF={sf:.1f} (Mindest {self.sf_required})  |  "
            f"Torsionsmoment {T_Nmm:.1f} N*mm  "
            f"Schubspannung {tau:.2f} MPa  "
            f"Schubfestigkeit {tau_all:.1f} MPa"
        )
        return RequirementResult(self.name, sf >= self.sf_required, sf, self.sf_required, detail)


# ======================================================================
# BucklingRequirement  (Euler Knickfall II: beidseitig eingespannt)
# ======================================================================

class BucklingRequirement(PhysicsRequirement):
    """Knickstabilitaets-Anforderung nach Euler (Knickfall II: ein Ende fest, eines frei).

    F_krit = pi^2 * E * I / (beta * L)^2   (beta=2 fuer Kragtraeger)
    I      = pi * r^4 / 4

    derive: r_min sodass F_krit >= SF * F_design
    verify: SF_buckling = F_krit / F_design
    """

    name = "Knickstabilitaet (Euler)"
    sf_required = 3.0
    required_params = (
        "load_mass_g", "buckling_length_mm", "section_r_mm",
        "E_mpa",
    )
    # Kragtraeger (Euler Fall I: ein Ende eingespannt, eines frei): beta=2
    _BETA = 2.0

    def derive(self, brief: "PhysicsBrief") -> dict:
        F_N = self._primary_force_N(brief)
        if F_N <= 0:
            return {}
        E_mpa  = brief.material.resolved("E_mpa")
        reach  = self._arm_reach_mm(brief)
        if reach <= 0:
            return {}
        L_mm   = reach                           # Knicklange = Hebelarm
        # F_krit = pi^2 * E * I / (beta*L)^2 >= SF * F
        # I = pi * r^4 / 4
        # r^4 >= SF * F * (beta*L)^2 / (pi^2 * E * pi/4)
        #      = 4 * SF * F * (beta*L)^2 / (pi^3 * E)
        F_N_design = F_N * self.sf_required      # schon SF eingerechnet
        beta_L_mm  = self._BETA * L_mm
        r_min4 = (
            4.0 * F_N_design * (beta_L_mm * 1e-3)**2
            / (math.pi**3 * E_mpa * 1e6)   # SI: F[N], L[m], E[Pa] → r^4[m^4]
        )
        r_min  = r_min4 ** 0.25 * 1000     # m → mm
        return {"section_r_min_mm": round(r_min, 2)}

    def verify(self, ctx: Dict[str, Any]) -> RequirementResult:
        self._verify_ctx(ctx)
        F_design_N = ctx["load_mass_g"] * 9.81e-3   # N
        L_m        = ctx["buckling_length_mm"] * 1e-3
        r_m        = ctx["section_r_mm"] * 1e-3
        E_Pa       = ctx["E_mpa"] * 1e6

        I_m4   = math.pi * r_m**4 / 4.0
        F_krit = math.pi**2 * E_Pa * I_m4 / (self._BETA * L_m)**2  # N
        sf     = F_krit / F_design_N if F_design_N > 0 else float("inf")

        detail = (
            f"SF={sf:.1f} (Mindest {self.sf_required})  |  "
            f"F_krit {F_krit:.1f} N  "
            f"F_design {F_design_N:.1f} N  "
            f"L={ctx['buckling_length_mm']:.0f} mm  r={ctx['section_r_mm']:.1f} mm"
        )
        return RequirementResult(self.name, sf >= self.sf_required, sf, self.sf_required, detail)


# ======================================================================
# TensionRequirement
# ======================================================================

class TensionRequirement(PhysicsRequirement):
    """Zug-/Druckspannungs-Anforderung.

    sigma = F / A,  SF = sigma_yield / sigma
    derive: A_min = F * SF / sigma_yield  →  r_min = sqrt(A_min / pi)
    """

    name = "Zug-/Druckanforderung"
    sf_required = 2.0
    required_params = ("load_mass_g", "section_r_mm", "yield_mpa")

    def derive(self, brief: "PhysicsBrief") -> dict:
        F_N = self._primary_force_N(brief)
        if F_N <= 0:
            return {}
        sig_y  = brief.material.resolved("yield_mpa")
        A_min  = F_N * self.sf_required / sig_y   # mm^2 (N / MPa = mm^2)
        r_min  = math.sqrt(A_min / math.pi)
        return {
            "section_r_min_mm":     round(r_min, 2),
            "section_area_min_mm2": round(A_min, 1),
        }

    def verify(self, ctx: Dict[str, Any]) -> RequirementResult:
        self._verify_ctx(ctx)
        F_N    = ctx["load_mass_g"] * 9.81e-3      # N
        r_mm   = ctx["section_r_mm"]
        sig_y  = ctx["yield_mpa"]

        A_mm2  = math.pi * r_mm**2
        sigma  = F_N / A_mm2 * 1e-3 if A_mm2 > 0 else float("inf")  # N/mm^2 = MPa? No: N/mm^2
        # sigma [MPa] = F[N] / A[mm^2] * 1 (N/mm^2 = MPa)
        sigma  = F_N / A_mm2 if A_mm2 > 0 else float("inf")
        sf     = sig_y / sigma if sigma > 0 else float("inf")

        detail = (
            f"SF={sf:.1f} (Mindest {self.sf_required})  |  "
            f"Kraft {F_N:.1f} N  "
            f"Querschnitt {A_mm2:.1f} mm2  "
            f"Spannung {sigma:.2f} MPa  "
            f"Streckgrenze {sig_y} MPa"
        )
        return RequirementResult(self.name, sf >= self.sf_required, sf, self.sf_required, detail)


# ======================================================================
# DragRequirement  (informational, kein Pflicht-SF)
# ======================================================================

class DragRequirement(PhysicsRequirement):
    """Aerodynamischer Widerstandskoeffizient (informational).

    Kein analytisches derive() -- Cd ist keine direkte Geometriefunktion.
    verify() gibt Warnung wenn Cd > max_Cd (kein harter Abbruch).
    """

    name = "Aerodynamischer Widerstand (Cd)"
    sf_required = 0.0   # informational: kein Pflicht-SF
    required_params = ("Cd", "Re")

    def __init__(self, Cd_warn: float = 2.0) -> None:
        self.Cd_warn = Cd_warn

    def derive(self, brief: "PhysicsBrief") -> dict:
        # Cd ist nicht analytisch aus Geometrieparametern ableitbar
        return {}

    def verify(self, ctx: Dict[str, Any]) -> RequirementResult:
        self._verify_ctx(ctx)
        Cd      = float(ctx["Cd"])
        Re      = float(ctx["Re"])
        Cd_warn = float(ctx.get("Cd_warn", self.Cd_warn))

        passed = Cd <= Cd_warn
        sf     = Cd_warn / Cd if Cd > 0 else float("inf")

        detail = (
            f"Cd={Cd:.3f}  Re={Re:.0f}  "
            f"(Grenzwert Cd_warn={Cd_warn:.1f})"
        )
        if not passed:
            detail += "  -- hoher Widerstand, aerodynamische Optimierung empfohlen"

        return RequirementResult(self.name, passed, round(sf, 3), self.sf_required, detail)


# ======================================================================
# Aliase fuer Rueckwaertskompatibilitaet
# ======================================================================

# Bestehende Tests und Code verwenden diese Namen -- nicht entfernen.
BaseCheck    = PhysicsRequirement
CheckResult  = RequirementResult

# Konkrete Aliases (alter Name → neue Klasse)
SectionBendingCheck = BendingRequirement
StemBendingCheck    = BendingRequirement   # legacy alias
TippingCheck        = TippingRequirement
DragCheck           = DragRequirement
