"""Why does YOLO11n beat the oracle on bbox_crop / warp_then_crop (rig, all bins)?

CPU only, uses saved predictions (runs/eval/<model>/preds_test.pkl) and the per-code CSVs
from scripts/per_code.py. Nothing in the published results is changed.

1. codes read by YOLO11n but not by the oracle (and the reverse), per decoder input
2. for those codes: crop size and quiet zone (modules) of both crops
3. ppm, tilt and distance to the image border of those codes
4. oracle (and YOLO11n) bbox_crop / warp_then_crop with other crop margins
5. side-by-side crops of a few discordant codes

  python scripts/oracle_crop_check.py --margins 0.25 0.35 0.5

Writes runs/oracle_crop_check/{codes.csv, sweep.csv, pair_*.png}. All numbers are from
SYNTHETIC images.
"""
import argparse
import csv
import math
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from per_code import oracle_preds  # noqa: E402
from slideqr import evaluation as EV  # noqa: E402
from slideqr.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = "yolo11n"


def crop_box(poly, margin, W, H):
    """Same box as evaluation.decode_bbox_crop; also says whether the image border cut it."""
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    raw = (math.floor(x0 - mx), math.floor(y0 - my), math.ceil(x1 + mx), math.ceil(y1 + my))
    box = (max(0, raw[0]), max(0, raw[1]), min(W, raw[2]), min(H, raw[3]))
    return box, box != raw


def read_crop(img, box):
    xa, ya, xb, yb = box
    if xb - xa < 4 or yb - ya < 4:
        return None
    res = EV.zx_read(np.ascontiguousarray(img[ya:yb, xa:xb]), first_only=True)
    return res[0] if res else None


def tilt_deg(corners):
    q = np.asarray(corners, float).reshape(4, 2)
    d = q[1] - q[0]
    a = abs(math.degrees(math.atan2(d[1], d[0]))) % 90
    return min(a, 90 - a)


def per_owner(preds, gts, H, W):
    """gt id -> list of predictions whose owner (evaluation.owners) is that code."""
    own = EV.owners(preds, gts, H, W)
    out = {}
    for p, o in zip(preds, own):
        if o is not None:
            out.setdefault(gts[o]["id"], []).append(p)
    return out


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return {(int(r["image_id"]), int(r["code_id"])): r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "train.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--margins", nargs="+", type=float, default=None)
    ap.add_argument("--eval-dir", default=str(ROOT / "runs" / "eval"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "oracle_crop_check"))
    ap.add_argument("--pairs", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(args.config, [])
    ec = cfg["eval"]
    base = ec["crop_margin_frac"]
    margins = sorted(set([base] + (args.margins if args.margins is not None else [base])))
    root = Path(cfg["data_root"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from pycocotools.coco import COCO
    coco = COCO(str(root / "annotations" / f"instances_{args.split}.json"))
    with open(Path(args.eval_dir) / MODEL / f"preds_{args.split}.pkl", "rb") as f:
        ypreds = pickle.load(f)
    ycsv = load_csv(Path(args.eval_dir) / MODEL / f"per_code_{args.split}.csv")
    ocsv = load_csv(Path(args.eval_dir) / "oracle" / f"per_code_{args.split}.csv")

    rows, sweep = [], {(m, k, s): 0 for m in [MODEL, "oracle"] for k in margins for s in ["crop", "hybrid", "crop_wrong"]}
    n_codes = 0
    for iid in sorted(coco.getImgIds()):
        img = cv2.imread(str(root / "images" / args.split / coco.imgs[iid]["file_name"]))
        H, W = img.shape[:2]
        gts = EV.gt_instances(coco, iid, 1)
        scored = [g for g in gts if not g["iscrowd"]]
        src = {MODEL: [p for p in ypreds.get(iid, []) if p["score"] >= ec["score_thr"]], "oracle": oracle_preds(img, gts)}
        owned = {m: per_owner(p, gts, H, W) for m, p in src.items()}
        # wrong reads: a crop that decodes to anything but the code under it
        for m, preds in src.items():
            own = EV.owners(preds, gts, H, W)
            for p, o in zip(preds, own):
                for k in margins:
                    t = read_crop(img, crop_box(p["poly"], k, W, H)[0]) if p["poly"] is not None else None
                    if t is not None and (o is None or t != gts[o]["payload"]):
                        sweep[(m, k, "crop_wrong")] += 1
        for g in scored:
            n_codes += 1
            key = (iid, g["id"])
            gq = np.asarray(g["corners"], float).reshape(4, 2)
            gx0, gy0 = gq.min(0)
            gx1, gy1 = gq.max(0)
            ppm = float(g["ppm"])
            info = {}
            for m in [MODEL, "oracle"]:
                warp = int((ycsv if m == MODEL else ocsv)[key]["quad_warp"])
                for k in margins:
                    got, best = False, None
                    for p in owned[m].get(g["id"], []):
                        if p["poly"] is None:
                            continue
                        box, cut = crop_box(p["poly"], k, W, H)
                        ok = read_crop(img, box) == g["payload"]
                        if best is None or ok:
                            best = (box, cut, p["poly"])
                        got = got or ok
                        if got:
                            break
                    sweep[(m, k, "crop")] += int(got)
                    sweep[(m, k, "hybrid")] += int(got or warp)
                    if k == base:
                        info[m] = (got, best)
            yc, oc = int(ycsv[key]["bbox_crop"]), int(ocsv[key]["bbox_crop"])
            yh, oh = int(ycsv[key]["warp_then_crop"]), int(ocsv[key]["warp_then_crop"])
            if info[MODEL][0] != bool(yc) or info["oracle"][0] != bool(oc):
                print("WARNING: recomputed crop read differs from per_code CSV", key)
            if yc == oc and yh == oh:
                continue
            row = dict(image_id=iid, code_id=g["id"], ppm=round(ppm, 3), tilt_deg=round(tilt_deg(g["corners"]), 1),
                       border_modules=round(min(gx0, gy0, W - gx1, H - gy1) / ppm, 2),
                       y_crop=yc, o_crop=oc, y_hybrid=yh, o_hybrid=oh,
                       y_warp=int(ycsv[key]["quad_warp"]), o_warp=int(ocsv[key]["quad_warp"]))
            for m, tag in [(MODEL, "y"), ("oracle", "o")]:
                best = info[m][1]
                if best is None:
                    continue
                (xa, ya, xb, yb), cut, poly = best
                px0, py0 = poly.min(0)
                px1, py1 = poly.max(0)
                row[f"{tag}_mask_w_mod"] = round((px1 - px0) / ppm, 2)
                row[f"{tag}_crop_w_mod"] = round((xb - xa) / ppm, 2)
                row[f"{tag}_crop_h_mod"] = round((yb - ya) / ppm, 2)
                row[f"{tag}_quiet_min_mod"] = round(min(gx0 - xa, gy0 - ya, xb - gx1, yb - gy1) / ppm, 2)
                row[f"{tag}_cut_by_border"] = int(cut)
                row[f"_{tag}_box"] = (xa, ya, xb, yb)
            rows.append(row)

    # tables
    keys = [k for k in rows[0] if not k.startswith("_")] if rows else []
    with open(out / "codes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(out / "sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "margin", "crop_read_rate", "hybrid_read_rate", "crop_wrong_reads", "codes"])
        for m in [MODEL, "oracle"]:
            for k in margins:
                w.writerow([m, k, round(sweep[(m, k, "crop")] / n_codes, 4), round(sweep[(m, k, "hybrid")] / n_codes, 4),
                            sweep[(m, k, "crop_wrong")], n_codes])
    print("codes", n_codes, "discordant", len(rows))
    for m in [MODEL, "oracle"]:
        for k in margins:
            print(m, k, "crop", round(100 * sweep[(m, k, "crop")] / n_codes, 2), "hybrid",
                  round(100 * sweep[(m, k, "hybrid")] / n_codes, 2), "wrong", sweep[(m, k, "crop_wrong")])

    # pictures: codes YOLO11n reads by crop and the oracle does not
    pick = [r for r in rows if r["y_crop"] == 1 and r["o_crop"] == 0 and "_y_box" in r and "_o_box" in r][: args.pairs]
    for i, r in enumerate(pick, 1):
        img = cv2.imread(str(root / "images" / args.split / coco.imgs[r["image_id"]]["file_name"]))
        g = next(a for a in EV.gt_instances(coco, r["image_id"], 1) if a["id"] == r["code_id"])
        tiles = []
        for tag, name, ok in [("y", "YOLO11n crop: read", 1), ("o", "oracle crop: not read", 0)]:
            xa, ya, xb, yb = r[f"_{tag}_box"]
            t = img[ya:yb, xa:xb].copy()
            q = np.asarray(g["corners"], float).reshape(4, 2) - [xa, ya]
            s = 360 / max(t.shape[:2])
            t = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            cv2.polylines(t, [np.round(q * s).astype(np.int32)], True, (0, 200, 0), 1)
            canvas = np.full((420, 380, 3), 255, np.uint8)
            canvas[50:50 + t.shape[0], 10:10 + t.shape[1]] = t
            cv2.putText(canvas, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0) if ok else (0, 0, 200), 1)
            cv2.putText(canvas, f"{xb - xa}x{yb - ya}px quiet {r[f'{tag}_quiet_min_mod']} mod", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
            tiles.append(canvas)
        pair = np.hstack(tiles)
        cv2.putText(pair, f"SYNTHETIC  ppm {r['ppm']}  tilt {r['tilt_deg']} deg  green = true corners", (10, 412),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
        cv2.imwrite(str(out / f"pair_{i}.png"), pair)
    print("pairs", len(pick), "->", out)


if __name__ == "__main__":
    main()
