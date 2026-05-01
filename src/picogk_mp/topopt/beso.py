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
    r_filter : radius in element lengths
    """
    from scipy.ndimage import uniform_filter
    w = uniform_filter(alpha.astype(float), size=int(np.ceil(r_filter)) * 2 + 1)
    return w


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

    # 4. Determine target volume for THIS iteration (step toward vol_target)
    vol_current = state.volume_fraction
    vol_step    = max(vol_target, vol_current - er)  # remove at most er per step

    n_total  = Nx * Ny * Nz
    n_target = max(1, int(round(vol_step * n_total)))

    # 5. Rank all elements by averaged sensitivity
    alpha_flat = alpha_avg.ravel()
    rank       = np.argsort(alpha_flat)[::-1]      # highest first

    # 6. Build new mask
    new_mask_flat = np.zeros(n_total, dtype=bool)
    new_mask_flat[rank[:n_target]] = True          # top-n_target elements are solid

    # 7. Bidirectional addition: re-add void elements with very high sensitivity
    #    (prevents permanently removing elements that become load-critical)
    alpha_max = alpha_flat.max()
    admissible = alpha_flat > add_ratio * alpha_max
    new_mask_flat[admissible] = True               # always include high-sensitivity

    new_mask = new_mask_flat.reshape(Nx, Ny, Nz)

    # 8. Update state
    compliance = float(alpha_e.sum())             # total strain energy ~ compliance
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
