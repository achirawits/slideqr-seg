"""Combine runs/eval/*/metrics.json into one Markdown report.

  python scripts/report.py --out docs/results_synthetic.md
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slideqr.evaluation import METHODS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["maskrcnn", "yolo11n", "yolo26n", "rfdetr", "rfdetr_long", "oracle"]  # oracle = ground-truth corners
DISPLAY = {"rfdetr_long": "rfdetr_long (SUPPLEMENTARY: no time budget)"}
LABEL = {"full_image": "full image", "bbox_crop": "axis-aligned crop", "quad_warp": "4-corner warp",
         "warp_then_crop": "warp, then crop if no read"}


def pct(x):
    return "-" if x != x else f"{100 * x:.1f}%"


def num(x, f="{:.2f}"):
    return "-" if x is None or x != x else f.format(x)


def lat(d):
    if not d:
        return "-"
    return f"{d['model_mean_s'] * 1000:.0f} / {d['pipeline_mean_s'] * 1000:.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results_synthetic.md")
    args = ap.parse_args()
    runs = {}
    for name in ORDER:
        p = ROOT / "runs" / "eval" / name / "metrics.json"
        if p.exists():
            runs[name] = json.loads(p.read_text(encoding="utf-8"))
    if not runs:
        raise SystemExit("no metrics found under runs/eval")
    L = ["# Model comparison - SYNTHETIC test images only", "",
         "All numbers come from synthetic renders; they are not real-photo results.", "",
         "* Same split, input long side 1280 (RF-DETR 1272), 60 min wall-clock training budget each, seed 0.",
         "* RF-DETR-Seg Nano runs ~25 min per epoch at 1272 px, so it got only 2 epochs in the budget; "
         "its numbers reflect the equal-time rule, not a converged model.",
         "* **rfdetr_long** is a SUPPLEMENTARY run OUTSIDE the equal-time rule (up to 30 epochs, early "
         "stopping on validation); compare it with the others only as \"what RF-DETR reaches when trained longer\".",
         "* warp, then crop = 4-corner warp first, axis-aligned crop only when the warp gives no read.",
         "* Decoder for all three inputs: zxing-cpp, LocalAverage binarizer with GlobalHistogram fallback.",
         "* **oracle** = ground-truth corners pushed through the same scoring: the ceiling for each ppm bin "
         "(and, in the corner-error column, the floor of the mask -> quad conversion).", ""]

    L += ["## Summary", "",
          "| model | params (M) | license | rig AP50 | rig AP50:95 | handheld AP50 | handheld AP50:95 | "
          "rig corner err (px, median) | GPU ms/img model / pipeline | CPU ms/img model / pipeline |",
          "|" + "---|" * 10]
    for n, m in runs.items():
        if n == "oracle":
            continue
        s = m["splits"]
        rig, hh = s.get("test", {}), s.get("test_handheld", {})
        ce = next((r["corner_err_px_median"] for r in rig.get("per_bin", []) if r["bin"] == "all"), None)
        L.append("| " + " | ".join([
            DISPLAY.get(n, n), num(m["params"] / 1e6, "{:.1f}"), m["license"],
            num(rig.get("mask_ap_code", {}).get("AP50"), "{:.3f}"), num(rig.get("mask_ap_code", {}).get("AP50_95"), "{:.3f}"),
            num(hh.get("mask_ap_code", {}).get("AP50"), "{:.3f}"), num(hh.get("mask_ap_code", {}).get("AP50_95"), "{:.3f}"),
            num(ce), lat(m.get("latency_gpu")), lat(m.get("latency_cpu"))]) + " |")
    L += ["", "Latency, one image at a time: *model* = pre-processing + forward pass + the model's own "
          "post-processing at network resolution (comparable across models); *pipeline* = also full-resolution "
          "masks, RLE and contours (CPU-bound, the same work for every model).", ""]

    for split, mode in [("test", "rig"), ("test_handheld", "handheld")]:
        L += [f"## Camera-decision ppm bins - {mode}", "",
              "Read rate = share of codes decoded correctly; wrong = decoded text that is not the code's payload.", "",
              "| model | bin | codes | detected | corner err median px | " +
              " | ".join(f"{LABEL[k]} read / wrong" for k in METHODS) + " |", "|" + "---|" * (5 + len(METHODS))]
        for n, m in runs.items():
            for r in m["splits"].get(split, {}).get("per_bin", []):
                if r["bin"] == "all" or r["bin"].startswith("["):
                    continue  # named camera-decision bins only
                L.append("| " + " | ".join([DISPLAY.get(n, n), r["bin"], str(r["n_codes"]), pct(r["detected"]),
                                            num(r["corner_err_px_median"])] +
                                           [f"{pct(r[f'{k}_read_rate'])} / {r[f'{k}_wrong_reads']}" for k in METHODS]) + " |")
        L.append("")

    for split, mode in [("test", "rig"), ("test_handheld", "handheld")]:
        L += [f"## All ppm bins - {mode}", ""]
        for n, m in runs.items():
            rows = m["splits"].get(split, {}).get("per_bin", [])
            if not rows:
                continue
            L += [f"### {DISPLAY.get(n, n)}", "", "| bin | codes | detected | corner err mean / median px | " +
                  " | ".join(f"{LABEL[k]} read / wrong" for k in METHODS) + " |", "|" + "---|" * (4 + len(METHODS))]
            for r in rows:
                L.append("| " + " | ".join([r["bin"], str(r["n_codes"]), pct(r["detected"]),
                                            f"{num(r['corner_err_px_mean'])} / {num(r['corner_err_px_median'])}"] +
                                           [f"{pct(r[f'{k}_read_rate'])} / {r[f'{k}_wrong_reads']}" for k in METHODS]) + " |")
            L.append("")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
