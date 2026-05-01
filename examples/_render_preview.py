"""Render headphone holder STL to PNG using vedo (vtk-backed, offscreen).

Camera is set via direct VTK calls after show() because the camera= dict
argument to show() is silently ignored in some vedo offscreen builds.

Model bbox:  x [-91, 48]  y [-48, 48]  z [0, 259]
Bbox centre: (-21, 0, 129)
Arm tip:     (-91, 0, ~244)  -- extends in -x direction from stem top
"""
from pathlib import Path
import vedo

STL = Path(__file__).parent.parent / "tests" / "fixtures" / "headphone_holder_v2.stl"
OUT = Path(__file__).parent.parent / "docs" / "headphone_holder_v2_preview.png"

mesh = vedo.load(str(STL))
mesh.color([130, 132, 145]).lighting("metallic")

plt = vedo.Plotter(offscreen=True, size=(1280, 960), bg=(35, 35, 46), bg2=(10, 10, 18))
plt.add(mesh)

# show() without camera arg just initialises the renderer
plt.show()

# Set camera manually via VTK so it always takes effect in offscreen mode.
# Position: front-right, elevated -- arm sweeps LEFT (-x) into view.
# x=+80 puts camera right of stem; arm at x=-91 extends to left of frame.
# y=-380 far enough in front to capture full 96mm base depth.
# z=200  below arm tip (z=244) so arm fills top portion of frame.
cam = plt.camera
# Camera: front-right, elevated.
# Bbox: x[-91,48] y[-48,48] z[0,259].  Centre (-21, 0, 129).
# Arm tip at x=-91, z=244.  Base disc at z=0, r=48.
# y=-600 pushes far enough back to fit full 259mm height in 4:3 frame.
cam.SetPosition(65, -480, 295)
cam.SetFocalPoint(-21, 0, 138)
cam.SetViewUp(0, 0, 1)
plt.renderer.ResetCameraClippingRange()
plt.render()

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.screenshot(str(OUT))
plt.close()
print(f"saved: {OUT}  ({OUT.stat().st_size // 1024} KB)")
