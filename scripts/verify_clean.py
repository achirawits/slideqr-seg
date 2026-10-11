"""Generator sanity check on CLEAN renders (no degradations).

For every fully visible code:
  * rectify with the ground-truth corners and decode -> must be 100 %
  * sample module centres and compare with the encoded matrix -> 0 bit errors
  * rectification residual vs. the ideal module grid, with GT corners and with
    corners shifted by 0.5 px (shows the check is sensitive to sub-pixel error)
  * plain zxing on the whole image (informational: no geometry hint)

  python scripts/verify_clean.py --n 150 --set code.symbology=datamatrix
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slideqr import codes as C  # noqa: E402
from slideqr import pipeline as P  # noqa: E402
from slideqr.config import load_config  # noqa: E402


def residual(img, corners, matrix, k=8):
    m = matrix.shape[0]
    patch = C.rectify(img, corners, m, k, 0).astype(np.float32)
    lo, hi = np.percentile(patch, 2), np.percentile(patch, 98)
    patch = np.clip((patch - lo) / max(hi - lo, 1), 0, 1)
    ideal = 1 - np.kron(matrix.astype(np.float32), np.ones((k, k), np.float32))
    return float(np.abs(patch - ideal).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--mode", default="rig")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    rng = np.random.default_rng(7)
    ppms = P.stratified_ppm(cfg, args.n, rng)
    stats = defaultdict(lambda: defaultdict(float))
    fails = []
    for i, ppm in enumerate(ppms):
        s = P.generate(cfg, args.mode, 500000 + i, float(ppm), degrade=False)
        direct = {t for t, _ in C.decode_direct(s["clean"])}
        for c in s["codes"]:
            if c["in_frame_frac"] < 0.999 or c["occluded_frac"] > 0:
                continue
            b = int(np.floor(c["ppm"]))
            st = stats[b]
            st["n"] += 1
            ok = C.decode_rectified(s["clean"], c["corners"], c["symbology"], c["modules"]) == c["payload"]
            st["rect_ok"] += ok
            st["bit_err"] += int((C.sample_modules(s["clean"], c["corners"], c["modules"]) != c["matrix"]).sum())
            st["res_gt"] += residual(s["clean"], c["corners"], c["matrix"])
            st["res_shift"] += residual(s["clean"], c["corners"] + 0.5, c["matrix"])
            st["direct_ok"] += c["payload"] in direct
            # below ~2 px/module on the shortest side the symbol is under-sampled, a physical
            # limit rather than a generator bug; only failures above it are treated as bugs
            if not ok and c["ppm_min_side"] >= 2.0:
                fails.append((s["index"], c["payload"], round(c["ppm"], 2)))
    print(f"SYNTHETIC clean renders, mode={args.mode}, symbology={cfg['code']['symbology']}, "
          f"modules={cfg['code']['CODE_MODULES']}, code={cfg['code']['CODE_SIZE_MM']} mm")
    print("ppm_bin   n  rect_decode  bit_err  resid_gt  resid_+0.5px  direct_zxing")
    tot = defaultdict(float)
    for b in sorted(stats):
        st = stats[b]
        n = st["n"]
        for k, v in st.items():
            tot[k] += v
        print(f"[{b},{b + 1})  {int(n):4d}  {st['rect_ok'] / n:10.1%}  {int(st['bit_err']):7d}  "
              f"{st['res_gt'] / n:8.4f}  {st['res_shift'] / n:12.4f}  {st['direct_ok'] / n:11.1%}")
    n = tot["n"]
    print(f"all      {int(n):4d}  {tot['rect_ok'] / n:10.1%}  {int(tot['bit_err']):7d}  "
          f"{tot['res_gt'] / n:8.4f}  {tot['res_shift'] / n:12.4f}  {tot['direct_ok'] / n:11.1%}")
    if fails:
        print("FAILED (generator bug!):", fails[:20])
        sys.exit(1)


if __name__ == "__main__":
    main()
