"""Presentation line chart: read rate vs pixels per module, one line per decoder input,
with the camera-decision bins shaded. Reads a per_bin_<split>.csv from evaluate.py.

  python scripts/plot_read_rate.py --csv runs/eval/yolo11n/per_bin_test.csv \
      --model "YOLO11n-seg" --out runs/story/read_rate_yolo11n_rig.png
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# reference palette (validated: scripts/validate_palette.js, light surface #fcfcfb)
SERIES = [("full_image", "ภาพเต็ม", "#2a78d6", "o"),
          ("bbox_crop", "ตัดตามกรอบ", "#eb6834", "s"),
          ("quad_warp", "ดัดภาพตาม 4 มุม", "#1baf7a", "^"),
          ("warp_then_crop", "วิธีผสม", "#eda100", "D")]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BAND, BAND_TXT = "#e9e8e3", "#52514e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="YOLO11n-seg")
    ap.add_argument("--setting", default="แท่น rig มองตรงลง, ชุดทดสอบ 450 ภาพ")
    ap.add_argument("--min-codes", type=int, default=30, help="bins with fewer codes are left out")
    ap.add_argument("--font", default="C:/Windows/Fonts/tahoma.ttf", help="font with Thai glyphs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    font_manager.fontManager.addfont(args.font)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=args.font).get_name()

    rows = []
    for r in csv.DictReader(open(args.csv, encoding="utf-8")):
        m = re.fullmatch(r"\[([\d.]+),([\d.]+)\)", r["bin"])
        if m and int(r["n_codes"]) >= args.min_codes:
            lo, hi = float(m.group(1)), float(m.group(2))
            rows.append(dict(x=(lo + hi) / 2, lo=lo, hi=hi, n=int(r["n_codes"]),
                             **{k: 100 * float(r[f"{k}_read_rate"]) for k, *_ in SERIES}))
    dropped = [r["bin"] for r in csv.DictReader(open(args.csv, encoding="utf-8"))
               if re.fullmatch(r"\[[\d.]+,[\d.]+\)", r["bin"]) and int(r["n_codes"]) < args.min_codes]
    xs = [r["x"] for r in rows]

    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for lo, hi, lab in [(3.5, 4.0, "กล้อง 8MP\n(3.5–4.0)"), (4.5, 5.0, "กล้อง 12MP\n(4.5–5.0)")]:
        ax.axvspan(lo, hi, color=BAND, zorder=0, lw=0)
        ax.text((lo + hi) / 2, 3, lab, ha="center", va="bottom", fontsize=9, color=BAND_TXT, zorder=1)
    for k, label, col, mk in SERIES:
        ys = [r[k] for r in rows]
        ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=6, mec=SURFACE, mew=1.5,
                solid_capstyle="round", solid_joinstyle="round", label=label, zorder=3)
    # direct labels at the line ends, nudged apart with thin leaders where they converge
    # (stacked downwards from the top so labels never climb into the title)
    ends = sorted(((rows[-1][k], label, col) for k, label, col, _ in SERIES), key=lambda t: -t[0])
    placed = []
    for y, label, col in ends:
        ty = min(y, 98.0) if not placed else min(y, placed[-1] - 4.2)
        placed.append(ty)
        ax.annotate(label, xy=(xs[-1], y), xytext=(xs[-1] + 0.32, ty), fontsize=10, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=col, lw=1), annotation_clip=False)
    ax.set_xlim(rows[0]["lo"], rows[-1]["hi"])
    ax.set_ylim(0, 100)
    ax.set_xticks([r["lo"] for r in rows] + [rows[-1]["hi"]])
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("พิกเซลต่อ module (ช่วงละ 0.5)", color=INK2, fontsize=11)
    ax.set_ylabel("อัตราอ่านรหัสสำเร็จ (%)", color=INK2, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.08), frameon=False, fontsize=9, ncol=2,
                    labelcolor=INK)
    del leg
    fig.text(0.06, 0.95, f"อัตราอ่านรหัสตามความละเอียดภาพ: {args.model}", fontsize=15, color=INK,
             weight="bold", va="top")
    fig.text(0.06, 0.895, args.setting, fontsize=10, color=INK2, va="top")
    fig.text(0.97, 0.95, "ภาพจำลอง (SYNTHETIC)", fontsize=10, color="white", ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.45", fc="#c2410c", ec="none"))
    n_lo, n_hi = min(r["n"] for r in rows), max(r["n"] for r in rows)
    note = f"รหัสต่อช่วง {n_lo}–{n_hi} อัน"
    if dropped:
        note += f" · ตัดช่วง {', '.join(dropped)} ออกเพราะรหัสน้อยกว่า {args.min_codes} อัน"
    note += " · ตัวเลขจากภาพสังเคราะห์ ยังไม่ได้ทดสอบกับภาพจริง"
    fig.text(0.06, 0.025, note, fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.08, right=0.86, top=0.84, bottom=0.14)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()
