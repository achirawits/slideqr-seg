"""Parametric CAD for the slide-reading rig: a flat PLA base printed in two halves with the
engraved frame and ticks, a rear bracket that joins the halves and holds a 2020 aluminium
column, and a sliding carriage with an arm that hangs the camera straight over the middle
of the frame.

Plate, frame and tick sizes are read from configs/synth.yaml (scene.pla_base) so the
printed base matches the synthetic renders. Everything else is in RIG below.

  ~/cadenv/bin/python hardware/rig_cad.py --out hardware/out

Coordinates: x along the slide row, y towards the column (back), z up; origin at the centre
of the plate's bottom face. Units: mm.
"""
import argparse
import math
from pathlib import Path

import cadquery as cq
import yaml

ROOT = Path(__file__).resolve().parent.parent

RIG = dict(
    plate_t=10.0,            # plate thickness
    lap_len=20.0,            # half-lap overlap of the two halves
    groove_w=0.8,            # engraved line width
    groove_d=0.6,            # engraved line depth
    m4_clear=4.5, m4_nut_af=7.2, m4_nut_h=3.4, m4_head_d=8.0, m4_head_h=4.0,
    m5_clear=5.5, m5_head_d=9.5, m5_head_h=5.0, m5_insert_d=6.4,
    foot_recess_d=12.0, foot_recess_h=1.0,
    bracket_w=120.0,         # along x
    bracket_d=40.0,          # behind the plate, along y
    bracket_h=40.0,
    bracket_lip=15.0,        # how far the lip reaches over the plate's rear strip
    bracket_lip_t=6.0,
    column=20.0,             # 2020 extrusion
    column_len=400.0,
    column_pocket=20.0,      # depth of the extrusion pocket in the bracket
    fit=0.3,                 # clearance for sliding/press fits
    carriage_h=50.0,
    carriage_wall=6.0,
    arm_w=26.0,
    arm_t=14.0,
    arm_past_centre=25.0,    # arm reaches this far beyond the frame centre
    cam_slot_len=40.0,       # 1/4"-20 slot, centred on the frame centre
    cam_screw_clear=6.6,     # 1/4"-20 clearance
    cam_dummy=(60.0, 40.0, 45.0),  # camera body used only in the preview assembly
    carriage_z=330.0,        # carriage bottom height in the preview assembly
)


def pla_base_cfg():
    cfg = yaml.safe_load(open(ROOT / "configs" / "synth.yaml", encoding="utf-8"))
    pb = cfg["scene"]["pla_base"]
    return dict(plate=tuple(pb["plate_mm"]), frame=tuple(pb["frame_mm"]),
                tick_pitch=float(pb["tick_pitch_mm"]), n_ticks=int(pb["n_ticks"]),
                tick_len=float(pb["tick_len_mm"]))


def _engrave(wp, base, R):
    """Cut the frame outline and the ticks into the top face."""
    (PW, PD), (FW, FH) = base["plate"], base["frame"]
    t, w, d = R["plate_t"], R["groove_w"], R["groove_d"]
    outer = cq.Workplane("XY").workplane(offset=t - d).rect(FW + w, FH + w).extrude(d + 1)
    inner = cq.Workplane("XY").workplane(offset=t - d).rect(FW - w, FH - w).extrude(d + 1)
    wp = wp.cut(outer.cut(inner))
    n, p, L = base["n_ticks"], base["tick_pitch"], base["tick_len"]
    for i in range(n):  # ticks straddle the back frame line (label end of the slides)
        x = (i - (n - 1) / 2) * p
        tick = cq.Workplane("XY").workplane(offset=t - d).center(x, FH / 2).rect(w, L).extrude(d + 1)
        wp = wp.cut(tick)
    return wp


def _m4_from_top(wp, pts, R, z_top):
    """Counterbored M4 clearance holes from the top face."""
    for x, y in pts:
        hole = cq.Workplane("XY").center(x, y).circle(R["m4_clear"] / 2).extrude(z_top + 1)
        cb = (cq.Workplane("XY").workplane(offset=z_top - R["m4_head_h"]).center(x, y)
              .circle(R["m4_head_d"] / 2).extrude(R["m4_head_h"] + 1))
        wp = wp.cut(hole).cut(cb)
    return wp


def _m4_nut_traps(wp, pts, R):
    """M4 clearance through the plate with a hex nut trap in the bottom face."""
    for x, y in pts:
        hole = cq.Workplane("XY").center(x, y).circle(R["m4_clear"] / 2).extrude(R["plate_t"] + 1)
        nut = cq.Workplane("XY").center(x, y).polygon(6, R["m4_nut_af"] / math.cos(math.pi / 6)).extrude(R["m4_nut_h"])
        wp = wp.cut(hole).cut(nut)
    return wp


def plate_half(side, base, R):
    """side = -1 (left, carries the lower lap) or +1 (right, carries the upper lap)."""
    PW, PD = base["plate"]
    t, lap = R["plate_t"], R["lap_len"]
    half = PW / 2
    if side < 0:
        body = cq.Workplane("XY").center(-half / 2, 0).rect(half, PD).extrude(t)
        tongue = cq.Workplane("XY").center(lap / 2, 0).rect(lap, PD).extrude(t / 2)
        body = body.union(tongue)
    else:
        body = cq.Workplane("XY").center(half / 2 + lap / 2, 0).rect(half - lap, PD).extrude(t)
        tongue = cq.Workplane("XY").workplane(offset=t / 2).center(lap / 2, 0).rect(lap, PD).extrude(t / 2)
        body = body.union(tongue)
    body = _engrave(body, base, R)
    # lap joint bolts: outside the frame and ticks, clear of the bracket lip; nut in the lower lap
    lap_y = PD / 2 - R["bracket_lip"] - R["m4_head_d"] / 2 - 3.0
    lap_pts = [(lap / 2, lap_y), (lap / 2, -lap_y)]
    for x, y in lap_pts:
        body = body.cut(cq.Workplane("XY").center(x, y).circle(R["m4_clear"] / 2).extrude(t + 1))
        if side < 0:
            body = body.cut(cq.Workplane("XY").center(x, y)
                            .polygon(6, R["m4_nut_af"] / math.cos(math.pi / 6)).extrude(R["m4_nut_h"]))
        else:
            body = body.cut(cq.Workplane("XY").workplane(offset=t - R["m4_head_h"]).center(x, y)
                            .circle(R["m4_head_d"] / 2).extrude(R["m4_head_h"] + 1))
    # bracket bolts on the rear strip, two per half
    bx = [s * v for s in (side,) for v in (25.0, 50.0)]
    yb = PD / 2 - R["bracket_lip"] / 2
    body = _m4_nut_traps(body, [(x, yb) for x in bx], R)
    # rubber-foot recesses
    for x in (side * (half - 20),):
        for y in (PD / 2 - 20, -(PD / 2 - 20)):
            body = body.cut(cq.Workplane("XY").center(x, y).circle(R["foot_recess_d"] / 2).extrude(R["foot_recess_h"]))
    return body


def bracket(base, R):
    PW, PD = base["plate"]
    t = R["plate_t"]
    W, D, H = R["bracket_w"], R["bracket_d"], R["bracket_h"]
    y0 = PD / 2 + R["fit"]
    block = cq.Workplane("XY").center(0, y0 + D / 2).rect(W, D).extrude(H).edges("|Z").fillet(3.0)
    lip = (cq.Workplane("XY").workplane(offset=t + R["fit"]).center(0, PD / 2 - R["bracket_lip"] / 2 + R["fit"] / 2)
           .rect(W, R["bracket_lip"] + R["fit"]).extrude(R["bracket_lip_t"]))
    body = block.union(lip)
    # gussets between block and column pocket
    cy = y0 + D / 2
    c = R["column"] + 2 * R["fit"]
    pocket = cq.Workplane("XY").workplane(offset=H - R["column_pocket"]).center(0, cy).rect(c, c).extrude(R["column_pocket"] + 1)
    body = body.cut(pocket)
    # M5 into the extrusion's end tap, from below, counterbored
    body = body.cut(cq.Workplane("XY").center(0, cy).circle(R["m5_clear"] / 2).extrude(H))
    body = body.cut(cq.Workplane("XY").center(0, cy).circle(R["m5_head_d"] / 2).extrude(R["m5_head_h"]))
    # bolts down through the lip into the plate's nut traps
    yb = PD / 2 - R["bracket_lip"] / 2
    top = t + R["fit"] + R["bracket_lip_t"]
    pts = [(s * v, yb) for s in (-1, 1) for v in (25.0, 50.0)]
    for x, y in pts:
        body = body.cut(cq.Workplane("XY").center(x, y).circle(R["m4_clear"] / 2).extrude(top + 1))
        body = body.cut(cq.Workplane("XY").workplane(offset=top - 2.0).center(x, y)
                        .circle(R["m4_head_d"] / 2).extrude(H))
    return body


def column(base, R):
    PW, PD = base["plate"]
    cy = PD / 2 + R["fit"] + R["bracket_d"] / 2
    z0 = R["bracket_h"] - R["column_pocket"]
    c = R["column"]
    prof = cq.Workplane("XY").workplane(offset=z0).center(0, cy).rect(c, c).extrude(R["column_len"])
    slot = cq.Workplane("XY").workplane(offset=z0).center(0, cy).rect(6.2, c + 1).extrude(R["column_len"])
    slot2 = cq.Workplane("XY").workplane(offset=z0).center(0, cy).rect(c + 1, 6.2).extrude(R["column_len"])
    core = cq.Workplane("XY").workplane(offset=z0).center(0, cy).rect(c - 3.6, c - 3.6).extrude(R["column_len"])
    body = prof.cut(slot.union(slot2).cut(core))
    body = body.cut(cq.Workplane("XY").workplane(offset=z0).center(0, cy).circle(2.1).extrude(R["column_len"]))
    return body


def carriage(base, R):
    """Slides on the column, locked by an M5 thumb screw in a heat-set insert; the arm
    reaches forward so the camera screw sits over the frame centre."""
    PW, PD = base["plate"]
    cy = PD / 2 + R["fit"] + R["bracket_d"] / 2
    c = R["column"] + 2 * R["fit"]
    wall, H = R["carriage_wall"], R["carriage_h"]
    outer = c + 2 * wall
    body = cq.Workplane("XY").center(0, cy).rect(outer, outer).extrude(H)
    body = body.cut(cq.Workplane("XY").center(0, cy).rect(c, c).extrude(H))
    # thumb screw boss on the back
    boss = (cq.Workplane("XZ", origin=(0, cy + outer / 2, H / 2)).circle(7.0).extrude(-6.0))
    body = body.union(boss)
    body = body.cut(cq.Workplane("XZ", origin=(0, cy + outer / 2 + 6.0, H / 2))
                    .circle(R["m5_insert_d"] / 2).extrude(wall + 6.0))
    # arm from the front face of the carriage to past the frame centre
    y_front = cy - outer / 2
    y_end = -R["arm_past_centre"]
    L = y_front - y_end
    arm = (cq.Workplane("XY").workplane(offset=H - R["arm_t"]).center(0, (y_front + y_end) / 2)
           .rect(R["arm_w"], L + 1).extrude(R["arm_t"]))
    arm = arm.edges("|Z and <Y").fillet(R["arm_w"] / 2 - 0.5)
    rib = (cq.Workplane("YZ").polyline([(y_front, H), (y_front, H + 25), (y_front - 0.6 * L, H)]).close()
           .extrude(6.0).translate((-3.0, 0, 0)))
    body = body.union(arm).union(rib)
    slot = (cq.Workplane("XY").workplane(offset=H - R["arm_t"] - 1).center(0, 0)
            .slot2D(R["cam_slot_len"], R["cam_screw_clear"], angle=90).extrude(R["arm_t"] + 2))
    body = body.cut(slot)
    return body


def camera_dummy(R):
    w, d, h = R["cam_dummy"]
    z_top = R["carriage_z"] + R["carriage_h"] - R["arm_t"]
    body = cq.Workplane("XY").workplane(offset=z_top - h).rect(w, d).extrude(h)
    lens = cq.Workplane("XY").workplane(offset=z_top - h - 12).circle(9).extrude(12)
    return body.union(lens)


def build(R=None):
    R = dict(RIG) if R is None else R
    base = pla_base_cfg()
    parts = dict(
        plate_left=plate_half(-1, base, R),
        plate_right=plate_half(+1, base, R),
        column_bracket=bracket(base, R),
        column_2020=column(base, R),
        camera_carriage=carriage(base, R).translate((0, 0, R["carriage_z"])),
    )
    return parts, base, R


COLORS = dict(plate_left=(0.82, 0.83, 0.84), plate_right=(0.78, 0.79, 0.80),
              column_bracket=(0.18, 0.43, 0.62), column_2020=(0.70, 0.72, 0.75),
              camera_carriage=(0.18, 0.43, 0.62), camera=(0.12, 0.12, 0.12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "hardware" / "out"))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "step").mkdir(parents=True, exist_ok=True)
    (out / "stl").mkdir(parents=True, exist_ok=True)
    parts, base, R = build()
    asm = cq.Assembly(name="slide_rig")
    for name, wp in parts.items():
        cq.exporters.export(wp, str(out / "step" / f"{name}.step"))
        if name != "column_2020":   # bought part, not printed
            shape = wp.translate((0, 0, -R["carriage_z"])) if name == "camera_carriage" else wp
            cq.exporters.export(shape, str(out / "stl" / f"{name}.stl"), tolerance=0.05, angularTolerance=0.1)
        asm.add(wp, name=name, color=cq.Color(*COLORS[name]))
    asm.add(camera_dummy(R), name="camera_dummy", color=cq.Color(*COLORS["camera"]))
    asm.save(str(out / "step" / "slide_rig_assembly.step"))
    lens_z = R["carriage_z"] + R["carriage_h"] - R["arm_t"] - R["cam_dummy"][2] - 12
    print("parts:", ", ".join(parts))
    print(f"plate {base['plate']}  frame {base['frame']}  lens height in preview {lens_z - R['plate_t']:.0f} mm above plate")
    for name, wp in parts.items():
        bb = wp.val().BoundingBox()
        print(f"{name:16s} {bb.xlen:6.1f} x {bb.ylen:6.1f} x {bb.zlen:6.1f}  vol {wp.val().Volume() / 1000:7.1f} cm3")


if __name__ == "__main__":
    main()
