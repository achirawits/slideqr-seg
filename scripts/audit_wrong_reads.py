"""List every wrong read (decoded text that is not the matched code's payload) for every
model, split and decoder input, with evidence images. Patient-identity safety check.

  python scripts/audit_wrong_reads.py

Writes runs/wrong_reads/wrong_reads.json and, per case, the region the decoder saw
plus a context crop with the predicted (red) and true (green) outline.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slideqr import codes as C  # noqa: E402
from slideqr import evaluation as EV  # noqa: E402
from slideqr.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["maskrcnn", "yolo11n", "yolo26n", "rfdetr", "rfdetr_long", "oracle"]


def fp(pts):
    return np.round((np.asarray(pts) - 0.5) * 16).astype(np.int32)


def context(img, pred_poly, gt_corners, pad=1.5):
    q = np.asarray(gt_corners, float).reshape(4, 2)
    c = q.mean(0)
    half = np.abs(q - c).max() * (1 + pad)
    x0, y0 = int(max(0, c[0] - half)), int(max(0, c[1] - half))
    x1, y1 = int(min(img.shape[1], c[0] + half)), int(min(img.shape[0], c[1] + half))
    out = img.copy()
    cv2.polylines(out, [fp(q)], True, (0, 255, 0), 1, cv2.LINE_AA, 4)
    if pred_poly is not None:
        cv2.polylines(out, [fp(pred_poly)], True, (0, 0, 255), 1, cv2.LINE_AA, 4)
    crop = out[y0:y1, x0:x1]
    return cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "train.yaml"))
    ap.add_argument("--eval-dir", default=str(ROOT / "runs" / "eval"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "wrong_reads"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    ec = cfg["eval"]
    root = Path(cfg["data_root"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for split in ["test", "test_handheld"]:
        coco = COCO(str(root / "annotations" / f"instances_{split}.json"))
        full = json.loads((Path(args.eval_dir) / f"_full_image_{split}.json").read_text(encoding="utf-8"))
        preds_by_model = {}
        for m in MODELS:
            p = Path(args.eval_dir) / m / f"preds_{split}.pkl"
            if p.exists():
                with open(p, "rb") as f:
                    preds_by_model[m] = pickle.load(f)
        for iid in sorted(coco.getImgIds()):
            info = coco.imgs[iid]
            gts_all = EV.gt_instances(coco, iid, 1)
            payloads = {g["payload"]: g for g in gts_all}
            img = None
            for t in full[str(iid)]:
                if t not in payloads:
                    cases.append(dict(split=split, image=info["file_name"], model="(none)", method="full_image",
                                      read=t, truth=None, kind="text not on any label in the image"))
            for m, preds in preds_by_model.items():
                ps = [p for p in preds.get(iid, []) if p["score"] >= ec["score_thr"]]
                if not ps:
                    continue
                if img is None:
                    img = cv2.imread(str(root / "images" / split / info["file_name"]))
                H, W = img.shape[:2]
                scored = [g for g in gts_all if not g["iscrowd"]]
                match = EV.match(ps, scored, H, W, ec["iou_match"])
                own = EV.owners(ps, gts_all, H, W)   # same rule as evaluate_image
                for i, p in enumerate(ps):
                    quad = EV.poly_to_quad(p["poly"])
                    reads = EV.decode_all(img, p["poly"], quad, ec)
                    g = gts_all[own[i]] if own[i] is not None else None
                    for meth, text in reads.items():
                        if text is None or (g is not None and text == g["payload"]):
                            continue
                        other = payloads.get(text)
                        if g is None:
                            kind = "no code under the prediction"
                        elif other is not None:
                            kind = "payload of ANOTHER code in the image (wrong slide)"
                        else:
                            kind = "text not on any label in the image (misdecode)"
                        k = len(cases)
                        rec = dict(id=k, split=split, image=info["file_name"], model=m, method=meth, score=p["score"],
                                   read=text, truth=g["payload"] if g else None, kind=kind,
                                   iou_matched=i in match, ppm=g["ppm"] if g else None,
                                   corner_err_px=EV.corner_error(quad, g["corners"]) if (g and quad is not None) else None,
                                   visible_frac=g.get("visible_frac") if g else None)
                        ref = g or other
                        if ref is not None:
                            cv2.imwrite(str(out / f"case{k:02d}_context.png"), context(img, p["poly"], ref["corners"]))
                        if meth in ("quad_warp", "warp_then_crop") and quad is not None:
                            cv2.imwrite(str(out / f"case{k:02d}_warp.png"),
                                        C.rectify(img, quad, 20, ec["rectify_px_per_module"], 3))
                        if meth in ("bbox_crop", "warp_then_crop") and p["poly"] is not None:
                            x0, y0 = p["poly"].min(0)
                            x1, y1 = p["poly"].max(0)
                            mx, my = (x1 - x0) * ec["crop_margin_frac"], (y1 - y0) * ec["crop_margin_frac"]
                            crop = img[int(max(0, y0 - my)):int(y1 + my), int(max(0, x0 - mx)):int(x1 + mx)]
                            cv2.imwrite(str(out / f"case{k:02d}_crop.png"),
                                        cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
                        cases.append(rec)
                        print(json.dumps(rec))
    (out / "wrong_reads.json").write_text(json.dumps(cases, indent=1), encoding="utf-8")
    print(f"{len(cases)} wrong reads -> {out}")


if __name__ == "__main__":
    main()
