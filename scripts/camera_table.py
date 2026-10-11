"""Camera selection table: pixels per module (ppm) vs code size, symbology and
number of slides per image. Pure geometry, no camera model is assumed.

  ppm      = image_width_px / field_width_mm * code_mm / modules
  width_px = target_ppm * modules / code_mm * field_width_mm
  field_width_mm = (n - 1) * pitch + slide_width + 2 * margin

  python scripts/camera_table.py --out docs/camera_table.md
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slideqr.config import load_config  # noqa: E402

CODE_MM = [5, 6, 6.5, 7, 8, 10]
SLIDES = [1, 5, 10]
CAMERAS = [("1080p", 1920), ("8MP", 3264), ("12MP", 4000), ("16MP", 4656)]  # landscape width px
TARGETS = [3, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--margin-mm", type=float, default=None,
                    help="free space left/right of the slide row (default 5 mm)")
    ap.add_argument("--out", default="docs/camera_table.md")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    pitch = float(cfg["scene"]["slide_pitch_mm"])
    slide_w = float(cfg["scene"]["slide_mm"][0])
    margin = 5.0 if args.margin_mm is None else args.margin_mm
    mods = cfg["code"]["CODE_MODULES"]
    cur_mm = float(cfg["code"]["CODE_SIZE_MM"])
    cur_sym = cfg["code"]["symbology"]
    symbols = [("DataMatrix 18", "datamatrix", int(mods["datamatrix"])), ("QR 21", "qr", int(mods["qr"]))]

    head = ["code mm", "type", "slides", "field mm"] + [f"width px for {t} ppm" for t in TARGETS] + \
           [f"{name} ({w} px)" for name, w in CAMERAS]
    rows = []
    for code_mm in CODE_MM:
        for label, key, m in symbols:
            for n in SLIDES:
                field = (n - 1) * pitch + slide_w + 2 * margin
                need = [t * m / code_mm * field for t in TARGETS]
                ppm = [w / field * code_mm / m for _, w in CAMERAS]
                cur = abs(code_mm - cur_mm) < 1e-9 and key == cur_sym
                rows.append((cur, [f"{code_mm:g}", label, n, f"{field:.0f}"] +
                             [f"{x:,.0f}" for x in need] + [f"{p:.1f}" for p in ppm]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Camera selection table (geometry only)",
        "",
        f"Slide pitch {pitch:g} mm, slide width {slide_w:g} mm, {margin:g} mm margin each side. "
        "Field = width of the slide row the image must cover. Camera widths are the landscape width "
        "of a 4:3 sensor (1080p = 16:9).",
        "**Bold ★ row = current provisional setting.** ppm < 3 is marked *(<3)*.",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["current"] + head)
        for cur, r in rows:
            w.writerow([int(cur)] + r)
            cells = [str(c) for c in r]
            cells[-len(CAMERAS):] = [c if float(c) >= 3 else f"{c} *(<3)*" for c in cells[-len(CAMERAS):]]
            if cur:
                cells = [f"**{c}**" for c in cells]
                cells[0] = "★ " + cells[0]
            lines.append("| " + " | ".join(cells) + " |")
    # the real rig: slides lie freely inside the engraved frame, so the camera must
    # see the whole frame (+ margin) no matter how many slides are on it
    fw = float(cfg["scene"]["pla_base"]["frame_mm"][0]) + 2 * margin
    cur_m = int(mods[cur_sym])
    pla = ", ".join(f"{name} {w / fw * cur_mm / cur_m:.1f}" for name, w in CAMERAS)
    need = ", ".join(f"{t} ppm: {t * cur_m / cur_mm * fw:,.0f} px" for t in TARGETS)
    lines += [
        "",
        f"**PLA base (real rig):** slides lie freely in the {fw - 2 * margin:g} mm frame, so the field is "
        f"{fw:g} mm whatever the slide count. Current code ({cur_mm:g} mm {cur_sym} {cur_m}): {pla} ppm; "
        f"width needed for {need}.",
        "",
        "Caveats: ideal geometric ppm. Real effective resolution is lower (Bayer demosaicing, lens MTF, "
        "JPEG, focus), so treat ~3 ppm as a floor, not a target. Tilt/perspective shrinks ppm on the far side. "
        "Very high ppm with few slides needs a short working distance; check the lens minimum focus distance.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
