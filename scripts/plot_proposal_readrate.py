"""Read rate vs px/module for the proposal (print figure, Thai labels in TH Sarabun New).

Our decoder input (warp, then crop) is the only coloured line; the baselines are
greys told apart by dash and marker so the figure survives grayscale printing.

  python scripts/plot_proposal_readrate.py --model yolo11n --font-dir proposal \
      --out proposal/figures/read_rate_yolo11n_rig

Writes <out>.pdf and <out>.png. All numbers are from SYNTHETIC images.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INK, MUTED, GRID, BAND, ACCENT = "#1d1d1b", "#5f5e5a", "#e4e2dc", "#efedea", "#2a78d6"
# (column, label, colour, linestyle, marker, width, z)
SERIES = [
    ("full_image", "ป้อนทั้งภาพ", "#a3a19b", (0, (1.2, 2.2)), "o", 1.4, 2),
    ("bbox_crop", "ตัดตามกรอบ (งานเดิม)", "#6f6d67", "-", "s", 1.4, 3),
    ("quad_warp", "ดัดภาพจาก 4 มุมอย่างเดียว", "#6f6d67", (0, (5, 2.5)), "^", 1.4, 3),
    ("oracle", "เพดาน: ใช้มุมจริง (oracle)", INK, (0, (1, 1.6)), None, 1.2, 4),
    ("warp_then_crop", "ดัดภาพ + ตัดกรอบสำรอง (ที่เสนอ)", ACCENT, "-", "o", 2.4, 5),
]
BANDS = [(3.5, 4.0, "8MP"), (4.5, 5.0, "12MP")]


def load_bins(path):
    """{bin_low: row} for the regular 0.5-wide bins with enough codes."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            b = r["bin"]
            if not b.startswith("[") or int(r["n_codes"]) < 30:
                continue
            lo, hi = (float(v) for v in b.strip("[)").split(","))
            if hi - lo == 0.5:
                out[lo] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n")
    ap.add_argument("--split", default="test")
    ap.add_argument("--eval-dir", default=str(ROOT / "runs" / "eval"))
    ap.add_argument("--font-dir", default=str(ROOT / "proposal"))
    ap.add_argument("--out", default=str(ROOT / "proposal" / "figures" / "read_rate_yolo11n_rig"))
    args = ap.parse_args()

    for f in ["THSarabunNew.ttf", "THSarabunNew_Bold.ttf"]:
        font_manager.fontManager.addfont(str(Path(args.font_dir) / f))
    plt.rcParams.update({"font.family": "TH Sarabun New", "font.size": 15, "axes.unicode_minus": False,
                         "pdf.fonttype": 42})

    ev = Path(args.eval_dir)
    model = load_bins(ev / args.model / f"per_bin_{args.split}.csv")
    oracle = load_bins(ev / "oracle" / f"per_bin_{args.split}.csv")
    xs = sorted(model)
    centre = [x + 0.25 for x in xs]

    fig, ax = plt.subplots(figsize=(16 / 2.54, 8.6 / 2.54), dpi=300)
    for lo, hi, name in BANDS:
        ax.axvspan(lo, hi, color=BAND, lw=0, zorder=0)
        ax.text((lo + hi) / 2, 51.2, name, ha="center", va="bottom", color=MUTED, fontsize=14)
    ys = {}
    for col, label, colour, ls, mk, lw, z in SERIES:
        src, key = (oracle, "warp_then_crop") if col == "oracle" else (model, col)
        y = [100 * float(src[x][f"{key}_read_rate"]) for x in xs]
        ys[col] = y
        ax.plot(centre, y, color=colour, ls=ls, lw=lw, marker=mk, ms=3.6 if mk else 0,
                mfc=colour, mec="white", mew=0.6, label=label, zorder=z)

    # the 8MP story, labelled once
    i = xs.index(3.5)
    for col, y_text in [("warp_then_crop", 101.6), ("bbox_crop", 84.0), ("full_image", 59.6)]:
        ours = col == "warp_then_crop"
        ax.text(centre[i], y_text, f"{ys[col][i]:.1f}%", ha="center", va="center", fontsize=14,
                color=ACCENT if ours else MUTED, fontweight="bold" if ours else "normal", zorder=6,
                bbox=dict(boxstyle="square,pad=0.1", fc=BAND, ec="none"))

    ax.set_xlim(2, 8)
    ax.set_ylim(50, 104)
    ax.set_xticks(range(2, 9))
    ax.set_yticks(range(50, 101, 10))
    ax.set_xlabel("ความละเอียดของรหัส (พิกเซลต่อ module)", color=INK)
    ax.set_ylabel("อัตราอ่านได้ถูกต้อง (%)", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, width=0.6, length=3)
    h, l = ax.get_legend_handles_labels()
    order = [4, 3, 2, 1, 0]
    ax.legend([h[k] for k in order], [l[k] for k in order], loc="lower right", frameon=False,
              fontsize=13.5, handlelength=2.6, labelspacing=0.25, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(out.with_suffix(".pdf"), out.with_suffix(".png"))


if __name__ == "__main__":
    main()
