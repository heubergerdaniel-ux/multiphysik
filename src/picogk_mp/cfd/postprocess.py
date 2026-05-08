"""Post-processing for CFD results: Cd, velocity PNG, temperature PNG."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .solver import FlowResult, ThermalResult


def velocity_magnitude(result: FlowResult) -> np.ndarray:
    """Velocity magnitude normalised by U_lb."""
    return np.sqrt(result.ux_lb**2 + result.uy_lb**2) / result.domain.U_lb


def save_velocity_png(result: FlowResult, out_path: str | Path) -> Path:
    """Save a 2D velocity-magnitude map with streamlines to PNG.

    Uses matplotlib; the file is written to *out_path*.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    mag = velocity_magnitude(result)
    mask = result.domain.mask

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    # Velocity magnitude heatmap (mask solid cells as NaN)
    display = np.where(mask, np.nan, mag)
    im = ax.imshow(
        display,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=2.0,
        interpolation="bilinear",
    )
    plt.colorbar(im, ax=ax, label="Velocity / U_in")

    # Streamlines (subsample for clarity)
    Ny, Nx = mag.shape
    step = max(Ny // 20, 1)
    Y, X = np.mgrid[0:Ny:step, 0:Nx:step]
    ux_s = result.ux_lb[::step, ::step] / result.domain.U_lb
    uy_s = result.uy_lb[::step, ::step] / result.domain.U_lb
    ax.streamplot(
        X[0].astype(float), np.arange(0, Ny, step, dtype=float),
        ux_s, uy_s,
        density=0.8, color="white", linewidth=0.5, arrowsize=0.5,
    )

    # Solid overlay
    solid_display = np.where(mask, 0.5, np.nan)
    ax.imshow(solid_display, origin="lower", cmap="Greys", alpha=0.7,
              vmin=0, vmax=1)

    ax.set_title(
        f"Flow velocity  |  Cd = {result.Cd:.3f}  |  Re = {result.Re:.0f}  "
        f"|  elapsed = {result.elapsed_s:.1f} s"
    )
    ax.set_xlabel("x (cells, flow direction)")
    ax.set_ylabel("y (cells, transverse)")

    fig.tight_layout()
    fig.savefig(str(out), dpi=120)
    plt.close(fig)
    return out


def save_temperature_png(result: ThermalResult, out_path: str | Path) -> Path:
    """Save a temperature field heatmap to PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    mask = result.domain.mask
    T = np.where(mask, np.nan, result.T_field)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    im = ax.imshow(T, origin="lower", cmap="hot", interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="Temperature (same units as T_inlet)")

    solid_display = np.where(mask, 0.5, np.nan)
    ax.imshow(solid_display, origin="lower", cmap="Blues", alpha=0.6,
              vmin=0, vmax=1)

    ax.set_title(
        f"Temperature field  |  h_conv = {result.h_conv:.1f} W/m2K  "
        f"|  T_max = {result.T_max:.1f}  |  T_surface = {result.T_surface_avg:.1f}"
    )
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")

    fig.tight_layout()
    fig.savefig(str(out), dpi=120)
    plt.close(fig)
    return out
