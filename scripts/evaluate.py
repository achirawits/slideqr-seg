"""Evaluate one trained model on the rig test set and the handheld test set.

  python scripts/evaluate.py --model yolo11n

Writes runs/eval/<model>/{metrics.json, per_bin_<split>.csv}. All numbers are
from SYNTHETIC images.
"""
import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slideqr import evaluation as EV  # noqa: E402
from slideqr import models as M  # noqa: E402
from slideqr.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPLITS = {"test": "rig", "test_handheld": "handheld"}
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs" / "ultralytics_config"))


def make_predictor(name, cfg, device, score_floor, train_dir):
    summary = json.loads((Path(train_dir) / name / "train_summary.json").read_text(encoding="utf-8"))
    w = summary["weights"]
    fam = cfg["models"][name]["family"]
    imgsz, max_det = cfg["train"]["imgsz"], cfg["eval"]["max_det"]
    if fam == "ultralytics":
        return M.YoloPredictor(name, w, imgsz, device, score_floor, max_det)
    if fam == "torchvision":
        return M.MaskRCNNPredictor_(w, imgsz, device, score_floor, max_det)
    if fam == "rfdetr":
        sys.path.insert(0, str(ROOT / "scripts"))
        from train import rfdetr_class, rfdetr_resolution
        Model = rfdetr_class()
        model = Model(pretrain_weights=w, resolution=rfdetr_resolution(Model, imgsz), device=device)
        return M.RFDETRPredictor(model, {1: "code", 2: "slide"}, device, score_floor, max_det)
    raise ValueError(fam)


def full_image_cache(root, split, coco, eval_dir):
    """Model-independent baseline (whole image -> decoder), computed once per split."""
    path = Path(eval_dir) / f"_full_image_{split}.json"
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    out = {}
    for iid in sorted(coco.getImgIds()):
        img = cv2.imread(str(Path(root) / "images" / split / coco.imgs[iid]["file_name"]))
        out[iid] = EV.decode_full(img)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    return out


def coco_ap(coco, results, cat_id):
    from pycocotools.cocoeval import COCOeval
    if not results:
        return dict(AP50=0.0, AP50_95=0.0)
    dt = coco.loadRes([dict(r) for r in results])  # loadRes adds keys in place: give it copies
    ev = COCOeval(coco, dt, "segm")
    ev.params.catIds = [cat_id]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return dict(AP50=float(ev.stats[1]), AP50_95=float(ev.stats[0]))


def main():
    from pycocotools.coco import COCO
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "train.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--skip-cpu-timing", action="store_true")
    ap.add_argument("--train-dir", default=str(ROOT / "runs" / "train"))
    ap.add_argument("--eval-dir", default=str(ROOT / "runs" / "eval"))
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    ec = cfg["eval"]
    root = Path(cfg["data_root"])
    out = Path(args.eval_dir) / args.model
    out.mkdir(parents=True, exist_ok=True)
    edges = np.asarray(ec["ppm_bins"], float)
    focus = {k: tuple(v) for k, v in ec["focus_bins"].items()}

    pred = make_predictor(args.model, cfg, "cuda", ec["ap_score_thr"], args.train_dir)
    metrics = dict(model=args.model, note="SYNTHETIC images only", license=cfg["models"][args.model]["license"],
                   params=int(pred.n_params()), splits={})
    for split, mode in SPLITS.items():
        ann = root / "annotations" / f"instances_{split}.json"
        if not ann.exists():
            continue
        coco = COCO(str(ann))
        full = full_image_cache(root, split, coco, args.eval_dir)
        tally = EV.Tally(edges, focus)
        results, preds_dump = [], {}
        for iid in sorted(coco.getImgIds()):
            info = coco.imgs[iid]
            img = cv2.imread(str(root / "images" / split / info["file_name"]))
            dets = pred.predict(img)
            for d in dets:
                results.append(dict(image_id=iid, category_id=1 if d["cls"] == "code" else 2,
                                    segmentation=d["rle"], score=d["score"]))
            codes = [d for d in dets if d["cls"] == "code"]
            preds_dump[iid] = codes
            gts = EV.gt_instances(coco, iid, 1)
            EV.evaluate_image(img, gts, codes, ec, tally, full_texts=full[iid])
        with open(out / f"preds_{split}.pkl", "wb") as f:
            pickle.dump(preds_dump, f)
        rows = tally.table()
        with open(out / f"per_bin_{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        metrics["splits"][split] = dict(mode=mode, n_images=len(coco.getImgIds()),
                                        mask_ap_code=coco_ap(coco, results, 1),
                                        mask_ap_slide=coco_ap(coco, results, 2), per_bin=rows)

    # latency: one image at a time, end-to-end incl. full-resolution mask post-processing
    ann = root / "annotations" / "instances_test.json"
    coco = COCO(str(ann))
    ids = sorted(coco.getImgIds())
    imgs = [cv2.imread(str(root / "images" / "test" / coco.imgs[i]["file_name"])) for i in ids[: ec["timing_images_gpu"]]]
    pred_t = make_predictor(args.model, cfg, "cuda", ec["score_thr"], args.train_dir)
    metrics["latency_gpu"] = M.time_predict(pred_t, imgs, device_sync=torch.cuda.synchronize)
    if not args.skip_cpu_timing:
        pred_c = make_predictor(args.model, cfg, "cpu", ec["score_thr"], args.train_dir)
        metrics["latency_cpu"] = M.time_predict(pred_c, imgs[: ec["timing_images_cpu"]], warmup=1)
    M.save_json(metrics, out / "metrics.json")
    print(json.dumps({k: v for k, v in metrics.items() if k != "splits"}, indent=1))


if __name__ == "__main__":
    main()
