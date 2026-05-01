"""Render headphone_holder.stl to PNG using vedo (vtk-backed, offscreen)."""
from pathlib import Path
import vedo

STL = Path(__file__).parent.parent / "tests" / "fixtures" / "headphone_holder.stl"
OUT = Path(__file__).parent.parent / "docs" / "headphone_holder_preview.png"

mesh = vedo.load(str(STL))
mesh.color([130, 132, 145]).lighting("metallic")

plt = vedo.Plotter(offscreen=True, size=(1280, 960), bg=(35, 35, 46), bg2=(10, 10, 18))
plt.add(mesh)
plt.show(
    camera={
        # right-front, elevated so both arm and base are visible
        "pos":       (130, -230, 200),
        "focalPoint": (-20, 0, 85),
        "viewup":    (0, 0, 1),
    }
)
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.screenshot(str(OUT))
plt.close()
print(f"saved: {OUT}  ({OUT.stat().st_size // 1024} KB)")
