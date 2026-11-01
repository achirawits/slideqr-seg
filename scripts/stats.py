"""Statistics for one training run per model (RM07: paired evaluation on the same test set).

Inputs: runs/eval/<model>/per_code_<split>.csv from scripts/per_code.py.
* 95% CI of every read rate: cluster bootstrap over images (codes in one image share
  lighting, blur and camera, so they are not independent).
* Paired tests on the same codes:
  - decoder inputs within a model: warp_then_crop vs bbox_crop / quad_warp
  - models with the same decoder input: every pair of the four equal-budget models
  exact McNemar test on the discordant codes, Holm correction over each family of
  tests, effect size = difference in read rate with its paired cluster-bootstrap 95% CI.
* Cochran's Q over the four models (more than two paired groups).

  python scripts/stats.py --out docs/stats_synthetic.md

All numbers are from SYNTHETIC images.
"""
import argparse
import csv
import itertools
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPLITS = {"test": "rig", "test_handheld": "handheld"}
MAIN = ["maskrcnn", "yolo11n", "yolo26n", "rfdetr"]
EXTRA = ["rfdetr_long", "oracle"]
INPUTS = ["full_image", "bbox_crop", "quad_warp", "warp_then_crop"]
BINS = {"all": (-np.inf, np.inf), "8MP [3.5,4)": (3.5, 4.0), "12MP [4.5,5)": (4.5, 5.0)}


def load(eval_dir, model, split):
    path = Path(eval_dir) / model / f"per_code_{split}.csv"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keys = np.array([(int(r["image_id"]), int(r["code_id"])) for r in rows])
    data = {k: np.array([int(r[k]) for r in rows]) for k in INPUTS + ["detected"]}
    data["ppm"] = np.array([float(r["ppm"]) for r in rows])
    data["image_id"] = keys[:, 0]
    data["key"] = [tuple(k) for k in keys]
    return data


def align(a, b):
    """Index arrays so that a[ia] and b[ib] are the same codes."""
    pos = {k: i for i, k in enumerate(b["key"])}
    ia = np.array([i for i, k in enumerate(a["key"]) if k in pos])
    ib = np.array([pos[a["key"][i]] for i in ia])
    return ia, ib


def cluster_index(image_ids):
    groups = {}
    for i, g in enumerate(image_ids):
        groups.setdefault(g, []).append(i)
    return [np.array(v) for v in groups.values()]


def bootstrap(clusters, stat, n_boot, rng):
    """Cluster bootstrap of a mean-type statistic. `stat(idx)` must be a mean over codes
    idx; it is evaluated per cluster once, then resampled as weighted sums."""
    sums = np.array([stat(c) * len(c) for c in clusters])
    sizes = np.array([len(c) for c in clusters], float)
    k = len(clusters)
    pick = rng.integers(0, k, (n_boot, k))
    vals = sums[pick].sum(1) / sizes[pick].sum(1)
    return np.percentile(vals, [2.5, 97.5])


def mcnemar_exact(x, y):
    """Two-sided exact McNemar p-value from paired 0/1 outcomes."""
    b = int(np.sum((x == 1) & (y == 0)))
    c = int(np.sum((x == 0) & (y == 1)))
    n = b + c
    if n == 0:
        return 1.0, b, c
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p), b, c


def cochran_q(mat):
    """Cochran's Q for k paired binary columns (rows = codes). Returns Q, df, p."""
    from scipy.stats import chi2
    k = mat.shape[1]
    col = mat.sum(0)
    row = mat.sum(1)
    n = mat.sum()
    den = k * n - np.sum(row ** 2)
    if den == 0:
        return 0.0, k - 1, 1.0
    q = (k - 1) * (k * np.sum(col ** 2) - n ** 2) / den
    return float(q), k - 1, float(chi2.sf(q, k - 1))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (m - r) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(ROOT / "runs" / "eval"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "docs" / "stats_synthetic.md"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    L = ["# Statistics on the SYNTHETIC test sets (one training run per model)", "",
         "All numbers come from synthetic renders; they are not real-photo results.", "",
         f"* 95% CI: cluster bootstrap over images ({args.n_boot} resamples, seed {args.seed}); codes "
         "in one image are not independent, so whole images are resampled.",
         "* Paired tests on the same codes: exact McNemar (b = only the first read it, c = only the "
         f"second), Holm-corrected within each table, alpha = {args.alpha}.",
         "* Each model was trained once (one seed), so these intervals cover test-set variability, "
         "not training variability (RM07: bootstrap CI is not training variance).", ""]

    for split, mode in SPLITS.items():
        D = {m: load(args.eval_dir, m, split) for m in MAIN + EXTRA}
        D = {m: d for m, d in D.items() if d is not None}
        if not D:
            continue
        L += [f"## {mode} ({split})", ""]
        # read rate with CI
        for bname, (lo, hi) in BINS.items():
            L += [f"### Read rate, {bname}", "",
                  "| model | codes | " + " | ".join(INPUTS) + " |", "|---|---|" + "---|" * len(INPUTS)]
            for m, d in D.items():
                sel = np.where((d["ppm"] >= lo) & (d["ppm"] < hi))[0]
                cl = cluster_index(d["image_id"][sel])
                cells = []
                for k in INPUTS:
                    v = d[k][sel]
                    ci = bootstrap(cl, lambda idx, v=v: v[idx].mean(), args.n_boot, rng)
                    cells.append(f"{100 * v.mean():.1f} [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]")
                L.append(f"| {m} | {len(sel)} | " + " | ".join(cells) + " |")
            L.append("")
        # decoder inputs within a model
        for bname, (lo, hi) in BINS.items():
            rows, ps = [], []
            for m in [x for x in MAIN + ["rfdetr_long"] if x in D]:
                d = D[m]
                sel = np.where((d["ppm"] >= lo) & (d["ppm"] < hi))[0]
                cl = cluster_index(d["image_id"][sel])
                for other in ["bbox_crop", "quad_warp"]:
                    x, y = d["warp_then_crop"][sel], d[other][sel]
                    p, b, c = mcnemar_exact(x, y)
                    ci = bootstrap(cl, lambda idx, x=x, y=y: x[idx].mean() - y[idx].mean(), args.n_boot, rng)
                    rows.append([m, f"warp_then_crop vs {other}", b, c,
                                 f"{100 * (x.mean() - y.mean()):+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]", p])
                    ps.append(p)
            adj = holm(np.array(ps))
            L += [f"### Decoder inputs within each model, {bname} (RQ2)", "",
                  "| model | comparison | b | c | difference, pp [95% CI] | p (Holm) | reject H0 |",
                  "|---|---|---|---|---|---|---|"]
            for r, pa in zip(rows, adj):
                L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {fmt_p(pa)} | {'yes' if pa < args.alpha else 'no'} |")
            L.append("")
        # models, same decoder input (hybrid), four equal-budget models
        mains = [m for m in MAIN if m in D]
        for bname, (lo, hi) in BINS.items():
            base = D[mains[0]]
            sel0 = np.where((base["ppm"] >= lo) & (base["ppm"] < hi))[0]
            keys = [base["key"][i] for i in sel0]
            cols, ok = [], True
            for m in mains:
                pos = {k: i for i, k in enumerate(D[m]["key"])}
                if not all(k in pos for k in keys):
                    ok = False
                    break
                cols.append(np.array([D[m]["warp_then_crop"][pos[k]] for k in keys]))
            if not ok or not keys:
                continue
            mat = np.stack(cols, 1)
            q, df, pq = cochran_q(mat)
            img = base["image_id"][sel0]
            cl = cluster_index(img)
            rows, ps = [], []
            for i, j in itertools.combinations(range(len(mains)), 2):
                x, y = mat[:, i], mat[:, j]
                p, b, c = mcnemar_exact(x, y)
                ci = bootstrap(cl, lambda idx, x=x, y=y: x[idx].mean() - y[idx].mean(), args.n_boot, rng)
                rows.append([f"{mains[i]} vs {mains[j]}", b, c,
                             f"{100 * (x.mean() - y.mean()):+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]", p])
                ps.append(p)
            adj = holm(np.array(ps))
            L += [f"### Models with warp_then_crop, {bname} (RQ1)", "",
                  f"Cochran's Q over {len(mains)} models: Q = {q:.2f}, df = {df}, p = {fmt_p(pq)} ({len(keys)} codes)", "",
                  "| pair | b | c | difference, pp [95% CI] | p (Holm) | reject H0 |", "|---|---|---|---|---|---|"]
            for r, pa in zip(rows, adj):
                L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {fmt_p(pa)} | {'yes' if pa < args.alpha else 'no'} |")
            L.append("")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
