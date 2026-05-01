"""BESO (Bi-directional Evolutionary Structural Optimization) update rule.

Algorithm (per iteration)
-------------------------
1. Compute strain-energy sensitivity number alpha_e for every element.
2. Apply a moving average filter over the sensitivity history to reduce
   checkerboard patterns and improve mesh-independence.
3. Rank all elements by alpha_e.
4. Remove the lowest-alpha solid elements until the target volume is met.
5. Re-add the highest-alpha void elements if they exceed the admission
   threshold (bidirectional step).
6. Check convergence: relative change in total compliance < tol for
   *patience* consecutive iterations.

References
----------
Huang & Xie, "Evolutionary topology optimization of continuum structures",
Wiley, 2010.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Sensitivity filter (element neighbourhood averaging)
# ---------------------------------------------------------------------------

def _neighbour_filter(
    alpha: np.ndarray,
    mask: np.ndarray,
    r_filter: float = 1.5,
) -> np.ndarray:
    """Weight sensitivity by distance to neighbours within radius r_filter.

    Uses a simple box-average over elements within r_filter element lengths.
    Suppresses checkerboard artefacts without requiring a full FE remesh.

    Parameters
    ----------
    alpha    : (Nx, Ny, Nz) raw sensitivity
    mask     : (Nx, Ny, Nz) bool -- solid elements
    r_filter : radius in element lengths (0 = no filter)

    Notes
    -----
    The filter is skipped if its footprint would cover any entire grid
    dimension -- on small grids that would equalise all sensitivities and
    destroy the ranking information needed for BESO.
    """
    from scipy.ndimage import uniform_filter
    size = int(np.ceil(r_filter)) * 2 + 1
    if size <= 1 or any(size >= d for d in alpha.shape):
        return alpha.astype(float)
    return uniform_filter(alpha.astype(float), size=size)


# ---------------------------------------------------------------------------
# Single BESO update step
# ---------------------------------------------------------------------------

@dataclass
class BESOState:
    """Mutable state carried across BESO iterations."""

    mask: np.ndarray                           # current solid/void assignment
    alpha_history: List[np.ndarray] = field(default_factory=list)
    compliance_history: List[float] = field(default_factory=list)

    @property
    def volume_fraction(self) -> float:
        return float(self.mask.mean())

    @property
    def n_solid(self) -> int:
        return int(self.mask.sum())


def beso_step(
    state: BESOState,
    alpha_e: np.ndarray,         # (Nel,) strain energy per element
    vol_target: float,           # target solid fraction (e.g. 0.40)
    er: float = 0.02,            # evolutionary removal ratio per step
    r_filter: float = 1.5,       # sensitivity filter radius [elements]
    add_ratio: float = 0.01,     # fraction of removed elements that can be re-added
) -> BESOState:
    """Perform one BESO iteration.

    Parameters
    ----------
    state       : current BESOState (mask + history)
    alpha_e     : (Nel,) raw sensitivity from FEM solve
    vol_target  : target solid volume fraction to reach eventually
    er          : maximum volume change per iteration (fraction of total)
    r_filter    : spatial smoothing radius
    add_ratio   : bidirectional addition threshold (fraction of alpha_max)

    Returns
    -------
    Updated BESOState (new mask + updated histories).
    """
    Nx, Ny, Nz = state.mask.shape

    # 1. Reshape sensitivity to 3D
    alpha_3d = alpha_e.reshape(Nx, Ny, Nz)

    # 2. Apply spatial filter
    alpha_f = _neighbour_filter(alpha_3d, state.mask, r_filter)

    # 3. Accumulate sensitivity history (moving average over last 2 iters)
    hist = state.alpha_history + [alpha_f]
    if len(hist) > 2:
        hist = hist[-2:]
    alpha_avg = np.mean(hist, axis=0)           # (Nx, Ny, Nz)

    # 4. ESO removal step: rank ONLY current solid elements by sensitivity,
    #    remove the weakest ones until the target count is reached.
    #
    #    We do NOT add void elements back (no bidirectional addition) because:
    #    a) void-element virtual sensitivities near force/BC nodes are
    #       inflated by the E_min regularisation, causing non-physical additions;
    #    b) we start from the full holder geometry which already has the right
    #       topology -- we only need to remove underloaded material.
    n_total   = Nx * Ny * Nz
    n_goal    = max(1, int(round(vol_target * n_total)))

    alpha_flat   = alpha_avg.ravel()
    solid_flat   = state.mask.ravel()
    solid_idx    = np.where(solid_flat)[0]       # indices of current solid els
    n_current    = len(solid_idx)

    # Rate-limit: change at most er * n_total elements per step
    max_remove = max(1, int(round(er * n_total)))
    n_remove   = int(np.clip(n_current - n_goal, 0, max_remove))

    # 5. Rank solid elements only; keep the top (n_current - n_remove)
    solid_alpha = alpha_flat[solid_idx]
    rank_solid  = np.argsort(solid_alpha)[::-1]   # highest sensitivity first
    n_keep      = n_current - n_remove
    keep_idx    = solid_idx[rank_solid[:n_keep]]   # top-n_keep solid elements

    # 6. Build new mask
    new_mask_flat = np.zeros(n_total, dtype=bool)
    new_mask_flat[keep_idx] = True
    new_mask = new_mask_flat.reshape(Nx, Ny, Nz)

    # 8. Update state
    # Compliance = total strain energy of SOLID elements only.
    # alpha_e has shape (Nel,) covering all elements (solid + void).
    # Void boundary elements share nodes with solids and accumulate
    # artificial strain energy at full stiffness E0 -- summing them
    # inflates compliance by 3-4 orders of magnitude and breaks
    # convergence detection.  Use only solid contributions.
    solid_mask_flat = state.mask.ravel()
    compliance = float(alpha_e[solid_mask_flat].sum())
    return BESOState(
        mask=new_mask,
        alpha_history=hist,
        compliance_history=state.compliance_history + [compliance],
    )


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------

def is_converged(
    state: BESOState,
    tol: float = 1e-3,
    patience: int = 5,
) -> bool:
    """True when relative compliance change is below *tol* for *patience* steps."""
    hist = state.compliance_history
    if len(hist) < patience + 1:
        return False
    recent   = np.array(hist[-patience:])
    baseline = abs(hist[-(patience + 1)])
    if baseline < 1e-30:
        return True
    max_change = np.max(np.abs(np.diff(recent))) / baseline
    return bool(max_change < tol)
