"""Headphone holder -- 'Arc Slim' design, pure Lattice. Opens in VedoViewer."""
import time
from pathlib import Path

import picogk
from picogk import Lattice, Mesh, VedoViewer, Voxels

OUT     = Path(__file__).parent.parent / "tests" / "fixtures" / "headphone_holder.stl"
PREVIEW = Path(__file__).parent.parent / "docs" / "headphone_holder_preview.png"
VOXEL   = 0.5   # mm


def _build(viewer: VedoViewer) -> None:
    lat = Lattice()

    # ------------------------------------------------------------------
    # BASE -- 4 corner pads + perimeter rails + center hub + spokes
    #         72 x 36 mm footprint, fully organic (no flat faces)
    # ------------------------------------------------------------------
    corners = [(-36, -18, 0), (36, -18, 0), (36, 18, 0), (-36, 18, 0)]

    for c in corners:
        lat.add_sphere(c, 7.0)                          # corner pads

    for i in range(4):
        lat.add_beam(corners[i], corners[(i + 1) % 4], 4.5, 4.5)   # rails

    lat.add_sphere([0, 0, 0], 10.0)                     # center hub
    for c in corners:
        lat.add_beam([0, 0, 0], c, 6.0, 4.5)           # spokes

    # ------------------------------------------------------------------
    # STEM -- 160 mm round post
    # ------------------------------------------------------------------
    lat.add_beam([0, 0, 0], [0, 0, 160], 10.0, 10.0)

    # ------------------------------------------------------------------
    # HOOK ARM -- smooth 3-segment arc, tip curves back down
    # ------------------------------------------------------------------
    lat.add_sphere([0,   0, 160], 11.5)                 # junction
    lat.add_beam([0,   0, 160], [-30,  0, 168], 10.0, 8.5)
    lat.add_beam([-30, 0, 168], [-58,  0, 170],  8.5, 7.0)
    lat.add_beam([-58, 0, 170], [-76,  0, 162],  7.0, 5.5)   # hook
    lat.add_sphere([-76, 0, 162], 6.5)                  # end cap

    # ------------------------------------------------------------------
    # VOXELISE + STL EXPORT
    # ------------------------------------------------------------------
    holder = Voxels.from_lattice(lat)
    mesh   = Mesh.from_voxels(holder)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mesh.SaveToStlFile(str(OUT))
    vol, _ = holder.calculate_properties()
    print(f"headphone_holder.stl: {mesh.triangle_count()} tri | "
          f"{vol:.0f} mm3 | {OUT.stat().st_size // 1024} KB")

    # ------------------------------------------------------------------
    # VIEWER  (matte anthracite, 3/4 front-left view)
    # ------------------------------------------------------------------
    viewer.add_mesh(mesh, color=[0.22, 0.22, 0.26])
    viewer.SetGroupMaterial(0, [0.22, 0.22, 0.26], fMetallic=0.05, fRoughness=0.65)
    viewer.SetViewAngles(fOrbit=-38, fElevation=22)
    viewer.SetZoom(1.5)
    viewer.request_render()

    # Give the renderer a moment, then save a PNG preview to docs/
    time.sleep(3)
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    viewer.RequestScreenShot(str(PREVIEW))
    time.sleep(1)
    print(f"preview saved -> {PREVIEW}")


if __name__ == "__main__":
    import sys
    # Pass --interactive to open the live viewer window instead of offscreen PNG
    interactive = "--interactive" in sys.argv
    viewer = VedoViewer(title="Kopfhoererhalter -- Arc Slim", offscreen=not interactive)
    eotc   = not interactive   # offscreen: close immediately; interactive: stay open
    picogk.go(VOXEL, lambda: _build(viewer), viewer=viewer, end_on_task_completion=eotc)
