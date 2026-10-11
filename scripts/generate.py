"""Generate a synthetic dataset (YOLO-seg + COCO + metadata.csv).

Example:
  python scripts/generate.py --out data/synth_v1 --n-rig 3000 --n-handheld 600 --workers 8
  python scripts/generate.py --out data/abl_noglare --set degrade.glare.enabled=false
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slideqr import export as E  # noqa: E402
from slideqr import pipeline as P  # noqa: E402
from slideqr.config import config_hash, dump_config, load_config  # noqa: E402

_CFG = None


def _init(cfg):
    global _CFG
    _CFG = cfg


def _work(job):
    mode, index, ppm, split, root, save_clean = job
    s = P.generate(_CFG, mode, index, ppm)
    stem = f"{mode}_{index:06d}"
    img_path = E.save_image(Path(root) / "images" / split / stem, s["image"], s["degrade"]["jpeg_quality"])
    if save_clean:
        E.save_image(Path(root) / "clean" / split / stem, s["clean"], 0)
    include_slide = _CFG["labels"]["include_slide_class"]
    lbl = Path(root) / "labels" / split / f"{stem}.txt"
    lbl.parent.mkdir(parents=True, exist_ok=True)
    lbl.write_text("\n".join(E.yolo_lines(s, include_slide)) + "\n", encoding="utf-8")
    W, H = s["size"]
    image = dict(file_name=img_path.name, width=W, height=H, mode=mode, ppm_target=round(ppm, 4),
                 surface=s["surface"], surface_variant=s["surface_variant"])
    anns = E.coco_anns(s, 0, 0, include_slide)
    rows = E.metadata_rows(s, split, f"images/{split}/{img_path.name}")
    return split, image, anns, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. code.CODE_SIZE_MM=6.5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rig", type=int, default=3000)
    ap.add_argument("--n-handheld", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-clean", action="store_true", help="also save undegraded renders")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, root / "config.yaml")
    rng = np.random.default_rng([int(cfg["seed"]), 99])

    jobs = []
    ppm_rig = P.stratified_ppm(cfg, args.n_rig, rng)
    splits = E.assign_splits(cfg, ppm_rig, rng)
    jobs += [("rig", i, float(p), s, str(root), args.save_clean) for i, (p, s) in enumerate(zip(ppm_rig, splits))]
    ppm_hh = P.stratified_ppm(cfg, args.n_handheld, rng)
    if cfg["split"]["handheld_test_only"]:
        hh_splits = ["test_handheld"] * args.n_handheld
    else:
        hh_splits = [f"{s}_handheld" for s in E.assign_splits(cfg, ppm_hh, rng)]
    jobs += [("handheld", i, float(p), s, str(root), args.save_clean) for i, (p, s) in enumerate(zip(ppm_hh, hh_splits))]

    t0 = time.time()
    by_split = {}
    rows = []
    with Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
        for split, image, anns, r in tqdm(pool.imap(_work, jobs, chunksize=2), total=len(jobs)):
            by_split.setdefault(split, []).append((image, anns))
            rows += r
    for split, items in by_split.items():
        images, all_anns = [], []
        for img_id, (image, anns) in enumerate(items, start=1):
            image["id"] = img_id
            images.append(image)
            for a in anns:
                a["image_id"] = img_id
                a["id"] = len(all_anns) + 1
                all_anns.append(a)
        E.write_coco(root / "annotations" / f"instances_{split}.json", images, all_anns)
    E.write_csv(root / "metadata.csv", rows)
    E.write_data_yaml(root, sorted(by_split))
    (root / "MANIFEST.txt").write_text(
        f"SYNTHETIC dataset - numbers derived from it are synthetic-only.\n"
        f"config_hash: {config_hash(cfg)}\nn_rig: {args.n_rig}\nn_handheld: {args.n_handheld}\n"
        f"splits: { {k: len(v) for k, v in by_split.items()} }\n"
        f"seconds: {time.time() - t0:.0f}\n", encoding="utf-8")
    print({k: len(v) for k, v in by_split.items()}, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
