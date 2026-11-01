"""Crop the key table/figure of each paper and locate the numbers to circle on the slides.

  python paper_figs.py <papers/core_6> <deck/assets>

Writes paper_<n>.png and prints the circle positions (percent of the crop) for the HTML.
The crops are from the published papers; the slides cite each paper next to its crop.
"""
import json
import sys

import pymupdf

src, out = sys.argv[1], sys.argv[2]
# key: (file, page index, clip in PDF points, [(search text, occurrence on that page)])
JOBS = {
    "1": ("1_Ershova2024_YOLO-Barcode.pdf", 4, (52, 470, 545, 583), [("98.6", 0), ("88.9", 0)]),
    "2": ("2_Pandya2024_INNOCOMP_YOLOv8_BarcodeQR.pdf", 0, (55, 428, 296, 474), [("97.10", 0)]),
    "3": ("3_Alsulami2026_UAV_BarcodeDecoding.pdf", 15, (150, 532, 452, 630), [("98.5", 0), ("80.5", 0)]),
    "4": ("4_Muallim2025_CMES_BarcodeRestoration.pdf", 5, (60, 62, 556, 250), [("Restore", 0)]),
    "5": ("5_Rybakova2025_DataMatrix_FastHough.pdf", 6, (96, 232, 508, 304), [("0.903", 1), ("0.858", 2)]),
    "6": ("6_Robinson2026_RF-DETR.pdf", 3, (100, 58, 515, 368), [("ViT Backbone", 0), ("Decoder Group", 0), ("Segmentation", 0)]),
}
res = {}
for k, (f, p, clip, terms) in JOBS.items():
    doc = pymupdf.open(f"{src}/{f}")
    pg = doc[p]
    c = pymupdf.Rect(clip)
    pg.get_pixmap(dpi=220, clip=c).save(f"{out}/paper_{k}.png")
    marks = []
    for t, occ in terms:
        hits = [r for r in pg.search_for(t) if c.contains(r)]
        if len(hits) <= occ:
            hits = [r for r in pg.search_for(t) if c.intersects(r)]
        if not hits:
            print("NOT FOUND", k, t)
            continue
        r = hits[min(occ, len(hits) - 1)]
        marks.append(dict(t=t, l=round((r.x0 - c.x0) / c.width * 100, 1), tp=round((r.y0 - c.y0) / c.height * 100, 1),
                          w=round(r.width / c.width * 100, 1), h=round(r.height / c.height * 100, 1)))
    res[k] = dict(aspect=round(c.width / c.height, 3), marks=marks)
print(json.dumps(res, ensure_ascii=False, indent=1))
