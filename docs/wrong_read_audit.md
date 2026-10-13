# Wrong-read audit (SYNTHETIC test images)

A wrong read is the most dangerous failure in a patient-identity check, so every one is
traced. Tool: `scripts/audit_wrong_reads.py` (all models, rig + handheld, every decoder
input). Evidence images are written to `runs/wrong_reads/` and are **not committed**
(the repo never stores image files).

## Current result

**0 wrong reads** for Mask R-CNN, YOLO11n-seg, YOLO26n-seg and RF-DETR-Seg Nano over
2,029 rig codes and 2,725 handheld codes, for the axis-aligned crop, the 4-corner warp
and warp-then-crop; and 0 for the whole image sent to the decoder.

## The one case in the first report, and why it was not a misread

| field | value |
|---|---|
| image | `test/rig_000472.png` (rig, 8 codes) |
| model | RF-DETR-Seg Nano, 60-min run (2 epochs) |
| decoder input | axis-aligned crop (and therefore warp-then-crop, which fell back to the crop) |
| prediction | score 0.61, mask area 881 px - a malformed C-shaped mask |
| text read | `S67-59588 G 5A` |
| code under the mask | `S67-59588 G 5A` - 98% of the predicted mask lies on this code |
| IoU with that code | 0.49 (match threshold 0.5) |

**What happened.** The mask was badly shaped, so its IoU with the code fell just below
the 0.5 match threshold and it was matched to no code. Its axis-aligned crop (bounding
box + 25% margin) still covered the whole symbol, and the decoder read that symbol
correctly. The first scoring rule compared the text only with the IoU-matched code, so
this correct read was counted twice against the model: once as a missed code and once
as a wrong read.

**Fix in the scoring (commit 322a0ac).** A decoded text is now judged against the code
that physically lies under the prediction (the code covering at least 50% of the
predicted mask). It is a wrong read only if it differs from that code's payload or if
no code lies under the prediction. Detection rate and corner error still use the
IoU >= 0.5 match. With this rule the case counts as a correct read of
`S67-59588 G 5A`.

**What would have made it a real danger.** A read of *another* slide's payload from
under a prediction, or a text found on no label. The audit reports both kinds
separately ("wrong slide" and "misdecode"); neither occurred.

**Residual risk for the real system.** Zero wrong reads on synthetic data does not prove
the real rate is zero. DataMatrix carries Reed-Solomon error correction, so a damaged
symbol is far more likely to give no read than a wrong read, but the real system should
still confirm each read against the expected worklist and never assign a read to a
slide position from a mask that does not overlap that position.

Evidence images of the first-report case (local only): `runs/wrong_reads/case00_context.png`
(4x; green = true outline, red = predicted mask), `case00_crop.png` (what the decoder saw),
`case01_warp.png`.
