"""Train one model under the shared conditions in configs/train.yaml.

  python scripts/train.py --model yolo11n
  python scripts/train.py --model maskrcnn --set train.time_budget_min=90

Same split, same input long side, same wall-clock budget (validation included),
same seed. The resolved config and environment are saved next to the weights.
"""
import argparse
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slideqr.config import dump_config, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CFG = ROOT / "configs" / "train.yaml"
# Ultralytics writes settings to ~/.config, which is not writable on some WSL setups
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs" / "ultralytics_config"))


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def env_info():
    import torch
    info = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    from importlib.metadata import PackageNotFoundError, version
    for pkg in ("torchvision", "ultralytics", "rfdetr", "pycocotools", "supervision", "transformers",
                "numpy", "opencv-python", "zxing-cpp"):
        try:
            info[pkg] = version(pkg)
        except PackageNotFoundError:
            info[pkg] = None
    return info


# ---------------------------------------------------------------- Ultralytics

def train_ultralytics(name, cfg, out_dir):
    from ultralytics import YOLO
    mc, tc = cfg["models"][name], cfg["train"]
    model = YOLO(mc["weights"])
    model.train(data=str(Path(cfg["data_root"]).resolve() / "data.yaml"), imgsz=tc["imgsz"],
                epochs=1000, time=tc["time_budget_min"] / 60.0,  # time overrides epochs
                batch=mc["batch"], workers=tc["workers"], seed=cfg["seed"], deterministic=False,
                project=str(out_dir.parent), name=out_dir.name, exist_ok=True, plots=False, verbose=True)
    best = out_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)
    return best


# ---------------------------------------------------------------- Mask R-CNN

class CocoSegDataset:
    """Resized to `imgsz` on the long side; iscrowd (ignored) instances are left out."""

    def __init__(self, root, split, imgsz, train):
        from pycocotools.coco import COCO
        self.coco = COCO(str(Path(root) / "annotations" / f"instances_{split}.json"))
        self.ids = sorted(self.coco.getImgIds())
        self.img_dir = Path(root) / "images" / split
        self.imgsz, self.train = imgsz, train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import cv2
        import torch
        info = self.coco.imgs[self.ids[i]]
        img = cv2.imread(str(self.img_dir / info["file_name"]), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        s = self.imgsz / max(W, H)
        w, h = round(W * s), round(H * s)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        flip = self.train and random.random() < 0.5
        masks, labels, boxes = [], [], []
        for a in self.coco.imgToAnns[info["id"]]:
            if a["iscrowd"]:
                continue
            m = np.zeros((h, w), np.uint8)
            for seg in a["segmentation"]:
                pts = np.asarray(seg, float).reshape(-1, 2) * s - 0.5
                cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
            if flip:
                m = m[:, ::-1]
            ys, xs = np.nonzero(m)
            if len(xs) < 4:
                continue
            masks.append(m)
            labels.append(a["category_id"])
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        if flip:
            img = img[:, ::-1]
        x = torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1).float().div(255)
        tgt = dict(boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                   labels=torch.tensor(labels, dtype=torch.int64),
                   masks=torch.from_numpy(np.stack(masks)) if masks else torch.zeros((0, h, w), dtype=torch.uint8))
        return x, tgt


def collate(batch):
    return tuple(zip(*batch))


def coco_segm_ap(pred_fn, root, split, max_images=None):
    """Mask AP (code class) on a split; pred_fn(img_bgr) -> det records."""
    import cv2
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco = COCO(str(Path(root) / "annotations" / f"instances_{split}.json"))
    ids = sorted(coco.getImgIds())[:max_images]
    res = []
    for iid in ids:
        info = coco.imgs[iid]
        img = cv2.imread(str(Path(root) / "images" / split / info["file_name"]))
        for d in pred_fn(img):
            res.append(dict(image_id=iid, category_id=1 if d["cls"] == "code" else 2,
                            segmentation=d["rle"], score=d["score"]))
    if not res:
        return 0.0
    dt = coco.loadRes(res)
    ev = COCOeval(coco, dt, "segm")
    ev.params.imgIds, ev.params.catIds = ids, [1]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[0])


def train_maskrcnn(name, cfg, out_dir):
    import torch
    from torch.utils.data import DataLoader
    from slideqr.models import MaskRCNNPredictor_, build_maskrcnn

    mc, tc = cfg["models"][name], cfg["train"]
    dev = torch.device("cuda")
    model = build_maskrcnn(len(tc["classes"]), tc["imgsz"]).to(dev)
    ds = CocoSegDataset(cfg["data_root"], "train", tc["imgsz"], train=True)
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    dl = DataLoader(ds, batch_size=mc["batch"], shuffle=True, num_workers=tc["workers"], collate_fn=collate,
                    generator=g, persistent_workers=True, drop_last=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=mc["lr"], momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    budget = tc["time_budget_min"] * 60.0
    t0 = time.time()
    warm = 500
    it, epoch, best_ap, log = 0, 0, -1.0, []
    w_best = out_dir / "weights" / "best.pth"
    w_best.parent.mkdir(parents=True, exist_ok=True)
    done = False
    while not done:
        model.train()
        it_epoch_start = it
        for imgs, tgts in dl:
            frac = (time.time() - t0) / budget
            if frac >= 1.0:
                done = True
                break
            # linear warm-up, then cosine decay over the wall-clock budget
            lr = mc["lr"] * min(1.0, (it + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))
            for pg in opt.param_groups:
                pg["lr"] = lr
            imgs = [i.to(dev, non_blocking=True) for i in imgs]
            tgts = [{k: v.to(dev, non_blocking=True) for k, v in t.items()} for t in tgts]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = sum(model(imgs, tgts).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if it % 50 == 0:
                print(f"it {it} ep {epoch} loss {loss.item():.4f} lr {lr:.5f} t {time.time() - t0:.0f}s", flush=True)
            it += 1
        if it == it_epoch_start:
            break  # budget ran out before this epoch trained anything: nothing new to validate
        # validation counts against the budget, as it does for the other frameworks
        epoch += 1
        model.eval()
        pred = MaskRCNNPredictor_.__new__(MaskRCNNPredictor_)
        pred.torch, pred.model, pred.imgsz, pred.device = torch, model, tc["imgsz"], dev
        pred.score_floor, pred.max_det = cfg["eval"]["ap_score_thr"], cfg["eval"]["max_det"]
        ap = coco_segm_ap(pred.predict, cfg["data_root"], "val")
        log.append(dict(epoch=epoch, iters=it, seconds=time.time() - t0, val_mask_ap_code=ap))
        print(log[-1], flush=True)
        if ap > best_ap:
            best_ap = ap
            torch.save(model.state_dict(), w_best)
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    return w_best


# ---------------------------------------------------------------- RF-DETR

def rfdetr_class():
    """Smallest RF-DETR segmentation model shipped by the installed rfdetr."""
    import rfdetr
    for cls_name in ("RFDETRSegNano", "RFDETRSegSmall", "RFDETRSegPreview"):
        if hasattr(rfdetr, cls_name):
            return getattr(rfdetr, cls_name)
    raise ImportError("installed rfdetr has no segmentation model")


def rfdetr_resolution(model_cls, imgsz):
    """Largest resolution <= imgsz that the backbone accepts (multiple of
    patch_size * num_windows; 12 for RF-DETR-Seg Nano, so 1280 -> 1272)."""
    import rfdetr.config as rc
    mcfg = getattr(rc, model_cls.__name__ + "Config")()
    block = int(mcfg.patch_size) * int(mcfg.num_windows)
    return (int(imgsz) // block) * block


def rfdetr_dataset(root, name="rfdetr", every=1):
    """Roboflow-style COCO layout (train/valid/test/*.json next to images) built from
    hard links, so no image is copied. every=k keeps every k-th image (pilot runs)."""
    import os
    root = Path(root)
    out = root / name
    for split, sub in [("train", "train"), ("val", "valid"), ("test", "test"), ("test_handheld", "test_handheld")]:
        src = root / "annotations" / f"instances_{split}.json"
        if not src.exists():
            continue
        d = out / sub
        d.mkdir(parents=True, exist_ok=True)
        data = json.loads(src.read_text(encoding="utf-8"))
        data["images"] = sorted(data["images"], key=lambda im: im["id"])[::every]
        keep = {im["id"] for im in data["images"]}
        data["annotations"] = [a for a in data["annotations"] if a["image_id"] in keep]
        for im in data["images"]:
            dst = d / im["file_name"]
            if not dst.exists():
                os.link(root / "images" / split / im["file_name"], dst)
        (d / "_annotations.coco.json").write_text(json.dumps(data), encoding="utf-8")
    return out


def _rfdetr_max_time(seconds):
    """Make rfdetr's Lightning Trainer stop at a wall-clock limit (rfdetr's train() does
    not expose Trainer kwargs, so wrap the builder it imports at call time)."""
    import datetime
    import rfdetr.training as rt
    orig = rt.build_trainer

    def build(tc, mc, **kw):
        kw.setdefault("max_time", datetime.timedelta(seconds=int(seconds)))
        return orig(tc, mc, **kw)
    rt.build_trainer = build
    return orig


def train_rfdetr(name, cfg, out_dir):
    import rfdetr.training as rt
    mc, tc = cfg["models"][name], cfg["train"]
    Model = rfdetr_class()
    res = rfdetr_resolution(Model, tc["imgsz"])
    print(f"rfdetr: {Model.__name__}, resolution {res}", flush=True)
    budget = tc["time_budget_min"] * 60.0
    kw = dict(batch_size=mc["batch"], grad_accum_steps=mc["grad_accum"], lr=mc["lr"], seed=cfg["seed"],
              num_workers=tc["workers"])
    if mc.get("epochs"):
        # Supplementary run OUTSIDE the equal-time rule: fixed epoch cap, stop when the
        # validation metric stops improving.
        resume = mc.get("resume")
        if resume and not Path(resume).exists():
            raise FileNotFoundError(f"resume checkpoint {resume} not found")
        print(f"rfdetr: supplementary run, up to {mc['epochs']} epochs, early stopping "
              f"patience {mc['early_stopping_patience']} (no time budget)"
              + (f", resuming from {resume}" if resume else ""), flush=True)
        Model(resolution=res).train(dataset_dir=str(rfdetr_dataset(cfg["data_root"])), epochs=int(mc["epochs"]),
                                    output_dir=str(out_dir), early_stopping=True,
                                    early_stopping_patience=int(mc["early_stopping_patience"]),
                                    early_stopping_min_delta=float(mc["early_stopping_min_delta"]),
                                    **({"resume": str(resume)} if resume else {}), **kw)
        return _rfdetr_checkpoint(out_dir)
    # The LR schedule follows `epochs`, so estimate how many epochs fit the budget, then let
    # Lightning's max_time stop the real run exactly at the budget. Two 1-epoch pilots on
    # 1/40 and 1/20 of train/val (not counted in the budget): time = startup + f * epoch,
    # so the difference removes rfdetr's fixed start-up cost from the estimate.
    times = {}
    for every in (40, 20):
        pilot = rfdetr_dataset(cfg["data_root"], f"rfdetr_pilot{every}", every)
        t0 = time.time()
        Model(resolution=res).train(dataset_dir=str(pilot), epochs=1, output_dir=str(out_dir / f"pilot{every}"), **kw)
        times[every] = time.time() - t0
    per_epoch = (times[20] - times[40]) / (1 / 20 - 1 / 40)
    if per_epoch <= 0:  # noise on tiny data: fall back to the plain scale-up
        per_epoch = times[20] * 20
    epochs = max(1, int(budget // per_epoch))
    print(f"rfdetr: pilots {times[40]:.0f}s / {times[20]:.0f}s -> ~{per_epoch:.0f} s/epoch -> {epochs} epochs, "
          f"hard stop at {budget / 60:.0f} min", flush=True)
    ds = rfdetr_dataset(cfg["data_root"])
    orig = _rfdetr_max_time(budget)
    try:
        Model(resolution=res).train(dataset_dir=str(ds), epochs=epochs, output_dir=str(out_dir), **kw)
    finally:
        rt.build_trainer = orig
    return _rfdetr_checkpoint(out_dir)


def _rfdetr_checkpoint(out_dir):
    for cand in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "checkpoint.pth"):
        if (out_dir / cand).exists():
            return out_dir / cand
    raise FileNotFoundError(f"no rfdetr checkpoint in {out_dir}")


TRAINERS = {"ultralytics": train_ultralytics, "torchvision": train_maskrcnn, "rfdetr": train_rfdetr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=str(TRAIN_CFG))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.model not in cfg["models"]:
        raise SystemExit(f"unknown model {args.model}; choose from {list(cfg['models'])}")
    out_dir = Path(args.out or ROOT / "runs" / "train" / args.model).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir / "train_config.yaml")
    seed_all(cfg["seed"])
    t0 = time.time()
    weights = TRAINERS[cfg["models"][args.model]["family"]](args.model, cfg, out_dir)
    summary = dict(model=args.model, weights=str(weights), wall_clock_min=(time.time() - t0) / 60,
                   budget_min=cfg["train"]["time_budget_min"], env=env_info())
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
