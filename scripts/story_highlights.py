"""Three presentation figures for one model on the rig test set (SYNTHETIC images):

  1. every code on the slide row is read
  2. the 4-corner warp fails on a code but warp-then-crop recovers it
  3. a code that is still not read, with the cause

Each figure: original | predicted masks | overlay with per-code status, a title with the
model name and "synthetic image", and a zoomed inset of the code the story is about.
Saved as PNG for slides.

  python scripts/story_highlights.py --model yolo11n --font /mnt/c/Windows/Fonts/tahoma.ttf
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slideqr import evaluation as EV  # noqa: E402
from slideqr.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GREEN, ORANGE, RED, GREY = (46, 160, 67), (230, 140, 0), (210, 40, 40), (120, 120, 120)   # RGB
NAMES = {"yolo11n": "YOLO11n-seg", "yolo26n": "YOLO26n-seg", "maskrcnn": "Mask R-CNN R50-FPN",
         "rfdetr": "RF-DETR-Seg Nano", "rfdetr_long": "RF-DETR-Seg Nano (long run)"}


def code_status(img, gts, preds, ec):
    """Per ground-truth code: detected, warp read, hybrid read, quad, prediction."""
    H, W = img.shape[:2]
    ps = [p for p in preds if p["score"] >= ec["score_thr"]]
    scored = [g for g in gts if not g["iscrowd"]]
    m = EV.match(ps, scored, H, W, ec["iou_match"])
    own = EV.owners(ps, gts, H, W)
    st = {g["id"]: dict(g=g, pred=None, quad=None, warp=False, crop=False, hybrid=False) for g in scored}
    for i, p in enumerate(ps):
        quad = EV.poly_to_quad(p["poly"])
        reads = EV.decode_all(img, p["poly"], quad, ec)
        o = gts[own[i]] if own[i] is not None else None
        if i in m:
            s = st[scored[m[i]]["id"]]
            s["pred"], s["quad"] = p, quad
        if o is not None and o["id"] in st:
            s = st[o["id"]]
            s["warp"] |= reads["quad_warp"] == o["payload"]
            s["crop"] |= reads["bbox_crop"] == o["payload"]
            s["hybrid"] |= reads["warp_then_crop"] == o["payload"]
    return list(st.values())


def oracle_reads(img, g, ec):
    q = np.asarray(g["corners"], float).reshape(4, 2)
    return EV.decode_all(img, q, q, ec)


def explain(img, s, meta_img, meta_code, ec):
    """Plain-language cause for a code that is not read (Thai, for the slide)."""
    g = s["g"]
    ppm = g["ppm"]
    if s["pred"] is None:
        return f"โมเดลหารหัสนี้ไม่เจอ ({ppm:.1f} px/module)"
    o = oracle_reads(img, g, ec)
    degr = []
    if float(meta_img["deg_defocus_sigma"]) > 0:
        degr.append(f"ภาพเบลอจากโฟกัส σ = {float(meta_img['deg_defocus_sigma']):.1f} px")
    if float(meta_img["deg_motion_len"]) > 0:
        degr.append(f"เบลอจากการสั่น {float(meta_img['deg_motion_len']):.1f} px")
    if float(meta_img["deg_glare"]) > 0.6:
        degr.append(f"แสงสะท้อนแรง {float(meta_img['deg_glare']):.2f}")
    if float(meta_code["label_stain"]) > 0:
        degr.append("ฉลากเปื้อน")
    if float(meta_img["deg_noise_sigma"]) > 4:
        degr.append(f"สัญญาณรบกวนสูง σ = {float(meta_img['deg_noise_sigma']):.1f}")
    if o["warp_then_crop"] is None:
        why = "ภาพเสียเกินกว่าจะอ่านได้: ใช้มุมจริงจากตัวสร้างภาพก็ยังอ่านไม่ออก"
        return f"{why} ({ppm:.1f} px/module" + (", " + ", ".join(degr) if degr else "") + ")"
    err = EV.corner_error(s["quad"], g["corners"]) if s["quad"] is not None else float("nan")
    return (f"มุมที่โมเดลหาได้คลาด {err:.1f} px (≈ {err / ppm:.1f} module) จึงดัดภาพผิดรูป "
            f"ถ้าใช้มุมจริงจะอ่านได้ ({ppm:.1f} px/module)")


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def poly_px(pts):
    return np.round((np.asarray(pts) - 0.5) * 16).astype(np.int32)


def render(img, sts, focus_id, title, subtitle, caption, fontpath, out_path, width=1920):
    H, W = img.shape[:2]
    # crop to the slide row (all codes + margin)
    allc = np.concatenate([np.asarray(s["g"]["corners"]).reshape(4, 2) for s in sts])
    x0, y0 = allc.min(0)
    x1, y1 = allc.max(0)
    mx = max(60, (x1 - x0) * 0.08)
    my = max(80, (y1 - y0) * 1.2)
    x0, y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
    x1, y1 = int(min(W, x1 + mx)), int(min(H, y1 + my * 2.5))
    over = img.copy()
    mask_panel = np.zeros_like(img)
    focus = None
    for s in sts:
        g = s["g"]
        col = GREEN if s["warp"] else (ORANGE if s["hybrid"] else RED)
        bgr = tuple(int(c) for c in col[::-1])
        if s["pred"] is not None:
            m = mask_utils.decode(s["pred"]["rle"]).astype(bool)
            mask_panel[m] = bgr
            over[m] = (0.5 * over[m] + 0.5 * np.array(bgr)).astype(np.uint8)
        q = np.asarray(g["corners"]).reshape(4, 2) if s["quad"] is None else s["quad"]
        cv2.polylines(over, [poly_px(q)], True, bgr, max(3, W // 900), cv2.LINE_AA, 4)
        if g["id"] == focus_id:
            focus = (q, bgr)
    panels = [c[y0:y1, x0:x1] for c in (img, mask_panel, over)]
    gap = 10
    ph = panels[0].shape[0]
    pw = (width - 2 * gap - 4 * 20) // 3
    scale = pw / panels[0].shape[1]
    ph_s = int(ph * scale)
    row = np.full((ph_s, width, 3), 255, np.uint8)
    for k, p in enumerate(panels):
        r = cv2.resize(p, (pw, ph_s), interpolation=cv2.INTER_AREA)
        xo = 20 + k * (pw + gap + 20)
        row[:, xo:xo + pw] = r
    # zoomed inset of the focus code
    inset = None
    if focus is not None:
        q, bgr = focus
        c = q.mean(0)
        half = np.abs(q - c).max() * 1.8
        a, b = int(max(0, c[1] - half)), int(min(H, c[1] + half))
        l, r_ = int(max(0, c[0] - half)), int(min(W, c[0] + half))
        z = over[a:b, l:r_]
        inset = cv2.resize(z, (260, int(260 * z.shape[0] / max(1, z.shape[1]))), interpolation=cv2.INTER_NEAREST)
    # compose with PIL for Thai text
    head_h, foot_h = 130, 230
    canvas = Image.new("RGB", (width, head_h + ph_s + foot_h), (255, 255, 255))
    canvas.paste(Image.fromarray(row[..., ::-1]), (0, head_h))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, width, 92], fill=(30, 41, 59))
    d.text((24, 14), title, font=font(fontpath, 40), fill=(255, 255, 255))
    d.text((24, 62), subtitle, font=font(fontpath, 22), fill=(203, 213, 225))
    tag = "ภาพจำลอง (SYNTHETIC) - ไม่ใช่ภาพจริง"
    tw = d.textlength(tag, font=font(fontpath, 24))
    d.rounded_rectangle([width - tw - 48, 22, width - 20, 70], radius=10, fill=(234, 88, 12))
    d.text((width - tw - 34, 30), tag, font=font(fontpath, 24), fill=(255, 255, 255))
    for k, lab in enumerate(["ภาพจากกล้อง", "mask ที่โมเดลทำนาย", "ผลการอ่านแต่ละรหัส"]):
        d.text((20 + k * (pw + gap + 20), head_h - 32), lab, font=font(fontpath, 22), fill=(51, 65, 85))
    y = head_h + ph_s + 16
    legend = [(GREEN, "อ่านได้จากการดัดภาพตาม 4 มุม"), (ORANGE, "ดัดภาพไม่ออก แต่ตัดตามกรอบอ่านได้ (วิธีผสม)"),
              (RED, "ยังอ่านไม่ได้")]
    x = 24
    for col, text in legend:
        d.rectangle([x, y + 4, x + 22, y + 26], fill=col)
        d.text((x + 30, y), text, font=font(fontpath, 20), fill=(30, 30, 30))
        x += 40 + int(d.textlength(text, font=font(fontpath, 20)))
    tx = 24
    if inset is not None:
        canvas.paste(Image.fromarray(inset[..., ::-1]), (width - inset.shape[1] - 24, y + 40))
        d.text((width - inset.shape[1] - 24, y + 44 + inset.shape[0]), "ขยายรหัสที่เป็นประเด็น",
               font=font(fontpath, 16), fill=(100, 116, 139))
    wrap_w = width - (320 if inset is not None else 0) - 48
    lines, cur = [], ""
    for word in caption.split(" "):
        test = (cur + " " + word).strip()
        if d.textlength(test, font=font(fontpath, 24)) > wrap_w:
            lines.append(cur)
            cur = word
        else:
            cur = test
    lines.append(cur)
    for k, line in enumerate(lines[:5]):
        d.text((tx, y + 46 + k * 34), line, font=font(fontpath, 24), fill=(15, 23, 42))
    canvas.save(out_path, optimize=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n")
    ap.add_argument("--split", default="test")
    ap.add_argument("--font", default="/mnt/c/Windows/Fonts/tahoma.ttf",
                    help="a font with Thai glyphs, used only for these figures")
    ap.add_argument("--config", default=str(ROOT / "configs" / "train.yaml"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "story"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    ec = cfg["eval"]
    root = Path(cfg["data_root"])
    coco = COCO(str(root / "annotations" / f"instances_{args.split}.json"))
    preds = pickle.load(open(ROOT / "runs" / "eval" / args.model / f"preds_{args.split}.pkl", "rb"))
    meta_img, meta_code = {}, {}
    for r in csv.DictReader(open(root / "metadata.csv", encoding="utf-8")):
        meta_img.setdefault(Path(r["image"]).name, r)
        meta_code[r["payload"]] = r
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = NAMES.get(args.model, args.model)

    cands = {1: [], 2: [], 3: []}
    for iid in sorted(coco.getImgIds()):
        info = coco.imgs[iid]
        gts = EV.gt_instances(coco, iid, 1)
        scored = [g for g in gts if not g["iscrowd"]]
        if len(scored) < 3 or len(scored) != len(gts):
            continue  # stories use rows with several codes, all fully visible
        img = cv2.imread(str(root / "images" / args.split / info["file_name"]))
        sts = code_status(img, gts, preds.get(iid, []), ec)
        n = len(sts)
        warp = sum(s["warp"] for s in sts)
        hyb = sum(s["hybrid"] for s in sts)
        med = float(np.median([s["g"]["ppm"] for s in sts]))
        rec = dict(iid=iid, img=img, sts=sts, n=n, med=med, file=info["file_name"])
        if warp == n:
            cands[1].append(rec)
        rescued = [s for s in sts if not s["warp"] and s["hybrid"]]
        if rescued and hyb == n:
            rec2 = dict(rec, focus=rescued[0]["g"]["id"])
            cands[2].append(rec2)
        failed = [s for s in sts if not s["hybrid"]]
        if failed and hyb >= n - 1:
            f = failed[0]
            o = oracle_reads(img, f["g"], ec)
            cands[3].append(dict(rec, focus=f["g"]["id"], oracle_fail=o["warp_then_crop"] is None,
                                 failed=f))
    picked = {}
    # 1: many slides, around the camera-decision range
    picked[1] = max(cands[1], key=lambda r: (r["n"], -abs(r["med"] - 4.0)))
    # 2: one rescued code among many, mid ppm
    picked[2] = max(cands[2], key=lambda r: (r["n"], -abs(r["med"] - 4.0)))
    # 3: prefer a clear image-quality cause (oracle fails too), many slides
    picked[3] = max(cands[3], key=lambda r: (r["oracle_fail"], r["n"], -abs(r["med"] - 4.0)))

    summary = []
    for k, r in picked.items():
        n = r["n"]
        warp = sum(s["warp"] for s in r["sts"])
        hyb = sum(s["hybrid"] for s in r["sts"])
        sub = f"{name}  |  แท่น rig มองตรงลง  |  {r['file']}  |  {n} สไลด์, ค่ากลาง {r['med']:.1f} px/module"
        if k == 1:
            title = f"1) อ่านได้ครบทุกแผ่น: {hyb}/{n}"
            cap = (f"ทุกรหัสในแถวถูกตรวจพบและอ่านได้ตั้งแต่ขั้นแรก (ดัดภาพตาม 4 มุมที่โมเดลหาได้) "
                   f"ผลอ่านตรงกับข้อมูลจริงครบ {n} แผ่น ไม่มีการอ่านผิด")
            focus = None
        elif k == 2:
            g = next(s for s in r["sts"] if s["g"]["id"] == r["focus"])
            err = EV.corner_error(g["quad"], g["g"]["corners"]) if g["quad"] is not None else float("nan")
            title = f"2) วิธีผสมกู้กลับมาได้: ดัดภาพ {warp}/{n} → ผสม {hyb}/{n}"
            cap = (f"รหัสสีส้ม ({g['g']['ppm']:.1f} px/module) การดัดภาพตาม 4 มุมอ่านไม่ออกเพราะมุมคลาด "
                   f"{err:.1f} px จึงลองตัดภาพตามกรอบสี่เหลี่ยมแทน และอ่านได้ถูกต้อง "
                   f"การลองสองวิธีต่อกันจึงเก็บรหัสที่หลุดกลับมาได้โดยไม่เพิ่มการอ่านผิด")
            focus = r["focus"]
        else:
            f = r["failed"]
            cause = explain(r["img"], f, meta_img[r["file"]], meta_code[f["g"]["payload"]], ec)
            title = f"3) ยังอ่านไม่ได้: {hyb}/{n}"
            cap = (f"รหัสสีแดง: {cause} ระบบไม่คืนค่าที่ผิด แต่รายงานว่าอ่านไม่ได้ "
                   f"เพื่อให้ผู้ใช้ตรวจสไลด์แผ่นนี้ด้วยตนเอง")
            focus = r["focus"]
        path = render(r["img"], r["sts"], focus, title, sub, cap, args.font,
                      out / f"story{k}_{args.model}_{Path(r['file']).stem}.png")
        summary.append(dict(story=k, file=r["file"], n=n, warp=warp, hybrid=hyb, median_ppm=round(r["med"], 2),
                            caption=cap, png=str(path)))
        print(json.dumps(summary[-1], ensure_ascii=False))
    (out / f"stories_{args.model}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
