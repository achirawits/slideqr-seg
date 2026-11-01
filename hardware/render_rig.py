"""Preview images of the rig for slides and the README (PNG, not committed):
  rig_iso.png      shaded 3D view (VTK, off-screen)
  rig_drawing.png  dimensioned top and side views, with the camera's field of view

  ~/cadenv/bin/python hardware/render_rig.py --out hardware/out
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import vtk  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rig_cad  # noqa: E402

BG = (0.957, 0.969, 0.980)
INK, MUTED, ACCENT, WARN = "#12304a", "#617181", "#2f6f9f", "#a34d3f"


def polydata(wp, tol=0.2):
    verts, tris = wp.val().tessellate(tol, 0.2)
    pts = vtk.vtkPoints()
    for p in verts:
        pts.InsertNextPoint(p.x, p.y, p.z)
    cells = vtk.vtkCellArray()
    for t in tris:
        cells.InsertNextCell(3, t)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    n = vtk.vtkPolyDataNormals()
    n.SetInputData(pd)
    n.SetFeatureAngle(30)
    n.Update()
    return n.GetOutput()


def actor(pd, rgb):
    m = vtk.vtkPolyDataMapper()
    m.SetInputData(pd)
    a = vtk.vtkActor()
    a.SetMapper(m)
    p = a.GetProperty()
    p.SetColor(*rgb)
    p.SetAmbient(0.25)
    p.SetDiffuse(0.75)
    p.SetSpecular(0.15)
    return a


def edges(pd):
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(pd)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOn()
    fe.SetFeatureAngle(40)
    fe.ManifoldEdgesOff()
    fe.NonManifoldEdgesOff()
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(fe.GetOutputPort())
    m.ScalarVisibilityOff()
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(0.15, 0.18, 0.22)
    a.GetProperty().SetLineWidth(1.2)
    return a


def render_iso(parts, R, path, size=(1920, 1440)):
    ren = vtk.vtkRenderer()
    ren.SetBackground(*BG)
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetSize(*size)
    win.SetMultiSamples(8)
    win.AddRenderer(ren)
    items = dict(parts)
    items["camera"] = rig_cad.camera_dummy(R)
    for name, wp in items.items():
        pd = polydata(wp)
        ren.AddActor(actor(pd, rig_cad.COLORS[name]))
        ren.AddActor(edges(pd))
    cam = ren.GetActiveCamera()
    cam.SetFocalPoint(0, 20, 150)
    cam.SetPosition(-620, -820, 620)
    cam.SetViewUp(0, 0, 1)
    ren.ResetCamera()
    cam.Zoom(1.05)
    light = vtk.vtkLight()
    light.SetPosition(-400, -600, 900)
    light.SetFocalPoint(0, 0, 0)
    light.SetIntensity(0.6)
    ren.AddLight(light)
    win.Render()
    f = vtk.vtkWindowToImageFilter()
    f.SetInput(win)
    f.Update()
    w = vtk.vtkPNGWriter()
    w.SetFileName(str(path))
    w.SetInputConnection(f.GetOutputPort())
    w.Write()


def dim(ax, p0, p1, text, off, horizontal, color=INK, fs=11):
    (x0, y0), (x1, y1) = p0, p1
    if horizontal:
        y = y0 + off
        ax.annotate("", (x0, y), (x1, y), arrowprops=dict(arrowstyle="<->", color=color, lw=1))
        ax.plot([x0, x0], [y0, y], color=color, lw=0.6)
        ax.plot([x1, x1], [y1, y], color=color, lw=0.6)
        ax.text((x0 + x1) / 2, y + (4 if off > 0 else -4), text, ha="center",
                va="bottom" if off > 0 else "top", fontsize=fs, color=color)
    else:
        x = x0 + off
        ax.annotate("", (x, y0), (x, y1), arrowprops=dict(arrowstyle="<->", color=color, lw=1))
        ax.plot([x0, x], [y0, y0], color=color, lw=0.6)
        ax.plot([x1, x], [y1, y1], color=color, lw=0.6)
        ax.text(x + (4 if off > 0 else -4), (y0 + y1) / 2, text, ha="left" if off > 0 else "right",
                va="center", fontsize=fs, color=color, rotation=0)


def render_drawing(base, R, path, font):
    from matplotlib import font_manager
    font_manager.fontManager.addfont(font)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    (PW, PD), (FW, FH) = base["plate"], base["frame"]
    n, pitch, L = base["n_ticks"], base["tick_pitch"], base["tick_len"]
    t = R["plate_t"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.2), dpi=150,
                                   gridspec_kw=dict(width_ratios=[1.15, 1]))
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.set_aspect("equal")
        ax.axis("off")

    # top view
    ax1.add_patch(Rectangle((-PW / 2, -PD / 2), PW, PD, fc="#d4d7da", ec=INK, lw=1.2))
    ax1.plot([0, 0], [-PD / 2, PD / 2], color=MUTED, lw=0.8, ls="--")
    ax1.add_patch(Rectangle((-FW / 2, -FH / 2), FW, FH, fc="none", ec=INK, lw=1.0))
    for i in range(n):
        x = (i - (n - 1) / 2) * pitch
        ax1.plot([x, x], [FH / 2 - L / 2, FH / 2 + L / 2], color=INK, lw=1.2)
    bw, bd = R["bracket_w"], R["bracket_d"]
    ax1.add_patch(Rectangle((-bw / 2, PD / 2), bw, bd, fc="#2f6f9f", ec=INK, lw=1, alpha=0.85))
    ax1.add_patch(Rectangle((-10, PD / 2 + bd / 2 - 10), 20, 20, fc="#b0b5bb", ec=INK, lw=1))
    ax1.text(0, PD / 2 + bd + 8, "เสา 2020", ha="center", fontsize=11, color=INK)
    ax1.text(4, PD / 2 - 14, "รอยต่อ 2 ชิ้น", fontsize=10, color=MUTED)
    dim(ax1, (-PW / 2, -PD / 2), (PW / 2, -PD / 2), f"{PW:.0f}", -22, True)
    dim(ax1, (-FW / 2, -FH / 2), (FW / 2, -FH / 2), f"กรอบ {FW:.0f}", -10, True, ACCENT, 10)
    dim(ax1, (-PW / 2, -PD / 2), (-PW / 2, PD / 2), f"{PD:.0f}", -14, False)
    dim(ax1, (FW / 2, -FH / 2), (FW / 2, FH / 2), f"{FH:.0f}", 12, False, ACCENT, 10)
    x0 = (0 - (n - 1) / 2) * pitch
    dim(ax1, (x0, FH / 2), (x0 + pitch, FH / 2), f"{pitch:.0f}", 16, True, ACCENT, 10)
    ax1.set_xlim(-PW / 2 - 45, PW / 2 + 30)
    ax1.set_ylim(-PD / 2 - 45, PD / 2 + bd + 25)
    ax1.set_title("มุมบน (มม.)", fontsize=14, color=INK, loc="left")

    # side view along x: field of view over the frame
    fov_w = FW + 20.0
    rows = []
    for hfov in (50, 65, 72):
        h = (fov_w / 2) / math.tan(math.radians(hfov / 2))
        rows.append((hfov, h))
    ax2.add_patch(Rectangle((-PW / 2, 0), PW, t, fc="#d4d7da", ec=INK, lw=1.2))
    ax2.plot([-FW / 2, FW / 2], [t + 1, t + 1], color=ACCENT, lw=3)
    hfov, h = rows[1]
    lens = t + h
    ax2.add_patch(Polygon([(0, lens), (-fov_w / 2, t), (fov_w / 2, t)], fc=ACCENT, alpha=0.12, ec=ACCENT, lw=1, ls="--"))
    ax2.add_patch(Rectangle((-30, lens), 60, 45, fc="#222", ec=INK))
    ax2.add_patch(Rectangle((-9, lens - 10), 18, 10, fc="#222", ec=INK))
    ax2.text(36, lens + 18, "กล้อง", fontsize=11, color=INK, va="center")
    dim(ax2, (fov_w / 2, t), (fov_w / 2, lens), f"{h:.0f}", 40, False)
    dim(ax2, (-fov_w / 2, t), (fov_w / 2, t), f"ภาพต้องครอบ ≥ {fov_w:.0f}", -22, True, ACCENT, 11)
    table = "\n".join(f"HFOV {a}°  →  เลนส์สูง {hh:.0f} มม." for a, hh in rows)
    ax2.text(-PW / 2, lens + 70, table, fontsize=11, color=INK, va="bottom", linespacing=1.6)
    ax2.set_xlim(-PW / 2 - 10, PW / 2 + 110)
    ax2.set_ylim(-45, lens + 175)
    ax2.set_title(f"มุมข้าง: ตัวอย่างเลนส์ HFOV {hfov}°", fontsize=14, color=INK, loc="left")
    fig.text(0.01, 0.02, "ขนาดฐาน กรอบ และขีด ตรงกับภาพจำลอง (configs/synth.yaml) • ความสูงคำนวณจากความกว้างภาพที่ต้องครอบ",
             fontsize=10, color=MUTED)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.93, bottom=0.07, wspace=0.08)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out"))
    ap.add_argument("--font", default="/mnt/c/Windows/Fonts/tahoma.ttf", help="font with Thai glyphs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parts, base, R = rig_cad.build()
    render_iso(parts, R, out / "rig_iso.png")
    render_drawing(base, R, out / "rig_drawing.png", args.font)
    print(out / "rig_iso.png", out / "rig_drawing.png")


if __name__ == "__main__":
    main()
