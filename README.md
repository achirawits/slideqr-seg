# slideqr-seg

Instance segmentation of the small 2D code (QR / DataMatrix) on pathology slide
labels, predicting the code's 4 corners as a polygon, plus a model comparison.

> **All numbers produced by this repo come from SYNTHETIC images** until the
> models are tested on real photos. Real photos (`real/`) contain patient sample
> numbers and are never committed (see `.gitignore`).

Status: steps 1-3 done on synthetic data - results in `docs/results_synthetic.md`,
team notes (Thai) in `docs/LEARNING.md`. Step 4 highlight images: `scripts/highlight.py`.
Not yet tested on real photos.

## Setup

Training runs in **WSL2 Ubuntu 24.04**. On this Windows 11 machine Smart App Control
blocks the unsigned `torch.dll` (`WinError 4551`), so PyTorch cannot load natively;
the Linux build inside WSL2 uses the same RTX 5070. The data generator (CPU only) runs
on either side.

### WSL2 (training + data)

1. Windows, PowerShell as administrator, then reboot:
   `wsl --install -d Ubuntu-24.04`
2. Check the GPU is visible inside Ubuntu: `nvidia-smi` (the Windows NVIDIA driver is
   enough; do **not** install a Linux driver inside WSL).
3. System packages (needs your sudo password):
   `sudo apt update && sudo apt install -y python3-venv python3-pip libdmtx0b fonts-dejavu`
   (`libdmtx0b` encodes DataMatrix; `fonts-dejavu` provides the label fonts)
4. Put the repo in the **Linux file system**, not under `/mnt/c`. Reading thousands
   of images through `/mnt/c` makes training I/O very slow.
   `git clone <repo url> ~/slideqr-seg && cd ~/slideqr-seg`
5. Python env with PyTorch **cu128** (RTX 5070 = sm_120) and the training packages:
   `bash scripts/setup_wsl.sh`
6. Regenerate the dataset inside Linux from the same seed (deterministic per image):
   `.venv/bin/python scripts/generate.py --out data/synth_v1 --n-rig 3000 --n-handheld 600 --workers 8`
   WSL gets half of the Windows RAM by default (15 GB here); ~1.2 GB per worker.

### Label fonts

Labels use **DejaVu** fonts (free licence) so every teammate renders the same text:
`scene.fonts.dir` (default `/usr/share/fonts/truetype/dejavu`) and
`scene.fonts.families` in `configs/synth.yaml` - DejaVu Sans, Sans Condensed, Sans
Mono, Serif, each with its bold face (the sample-id line is bold). A missing font file
stops the generator with an error; there is no silent fallback. No font files are
stored in the repo.

The reference dataset is generated on Ubuntu 24.04 (WSL2). On another OS, point
`scene.fonts.dir` at a folder holding the same DejaVu .ttf files; images will match
closely but are not guaranteed bit-identical (OpenCV/JPEG builds differ), so train
and test on data generated on one platform.

## Licences of the compared models

| model | code | pretrained weights | source checked |
|---|---|---|---|
| Mask R-CNN R50-FPN | BSD-3-Clause (torchvision) | COCO-pretrained, distributed by torchvision | torchvision |
| YOLO11n-seg, YOLO26n-seg | AGPL-3.0 or Ultralytics Enterprise License | same terms as the code | Ultralytics |
| RF-DETR-Seg Nano | Apache-2.0 (`rfdetr`) | **Apache-2.0** - the official README lists every RF-DETR-Seg size (Nano to 2XLarge) as Apache 2.0 | github.com/roboflow/rf-detr, 2026-10-13 |

Under the same README, only some *detection* sizes (Atto, Femto, Pico, XL, 2XL) and the
`rfdetr_plus` extension are under PML 1.0; none of them is used here. AGPL-3.0 matters if
the YOLO-based system is distributed or offered as a network service: the full source
must then be released under AGPL, or an Ultralytics licence bought.

## Library versions used (WSL2 Ubuntu 24.04, RTX 5070)

| package | version |
|---|---|
| Python | 3.12.3 |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 (sm_120 in arch list) |
| ultralytics | 8.4.175 (yolo11n-seg.pt, yolo26n-seg.pt) |
| rfdetr | 1.11.2 with `[train,loggers]` extras, pytorch-lightning 2.6.6 |
| supervision / transformers | 0.30.9 / 5.19.0 |
| pycocotools | 2.0.11 |
| numpy / opencv-python / pillow | 2.5.4 / 5.0.0.93 / 12.3.0 |
| segno / pylibdmtx / zxing-cpp | 1.6.6 / 0.1.10 / 3.1.1 |
| system | libdmtx0t64 0.7.7, fonts-dejavu |

Each training run also saves the versions it actually used in
`runs/train/<model>/train_summary.json`.

### API adjustments found on the first run

* **RF-DETR 1.11.2**
  * Training needs the extras: `pip install "rfdetr[train,loggers]"` (PyTorch Lightning).
  * Smallest segmentation model: `RFDETRSegNano` (patch 12, default 312 px). The
    resolution must be a multiple of `patch_size x num_windows` = 12, so the shared
    1280 px becomes **1272 px**.
  * Batch 2 at 1272 px runs out of the 12 GB; **batch 1 x gradient accumulation 8**
    peaks at ~8 GiB (effective batch 8, same as YOLO). `gradient_checkpointing` is a
    model-config field, not a `train()` argument.
  * `predict()` returns `class_id` as the **0-based index into `model.class_names`**,
    not the COCO category id. Mapping it as COCO ids scored slides as codes; evaluation
    now maps through `class_names` and checks the names.
  * `train()` has no wall-clock limit and rejects unknown Trainer arguments. The
    budget is enforced with Lightning `max_time` by wrapping
    `rfdetr.training.build_trainer`; the epoch count for the LR schedule comes from
    two short pilots (1/40 and 1/20 of the data, outside the budget).
* **Ultralytics 8.4**: settings are written to `runs/ultralytics_config`
  (`YOLO_CONFIG_DIR`), because `~/.config` was not writable in this WSL setup.
* **pycocotools**: `COCO.loadRes` adds keys to the result dicts in place, so it gets
  copies when called more than once.
* **Latency** is reported twice: the model alone (pre-processing, forward pass, the
  model's own post-processing) and the full pipeline (plus full-resolution masks, RLE,
  contours). The first version timed only the pipeline, where the CPU-side mask work
  hid the difference between GPU and CPU.

## Steps 2-3 - training and evaluation

Shared conditions in `configs/train.yaml`: same split, input long side 1280 for every
model, same wall-clock budget (default 60 min, validation included), seed 0. The
resolved config and package versions are saved in `runs/train/<model>/`.

```bash
bash scripts/run_all.sh                               # all four, one after another
.venv/bin/python scripts/train.py --model maskrcnn    # or one model: yolo11n, yolo26n, rfdetr
.venv/bin/python scripts/evaluate.py --model maskrcnn
.venv/bin/python scripts/report.py --out docs/results_synthetic.md
```

Evaluation (rig test set and handheld test set reported separately): mask AP50 and
AP50:95; corner error after turning each mask into a 4-corner quad; decoding with three
ways of feeding the decoder: the whole image, an axis-aligned crop of the predicted
mask, and a perspective warp from the predicted 4 corners; read rate and wrong reads
per ppm bin, with the camera-decision bins 3.5-4.0 ppm (8 MP on the PLA base) and
4.5-5.0 ppm (12 MP) listed separately; latency on GPU and CPU, parameter count,
license. How each framework is held to the budget: Ultralytics `time=`, Mask R-CNN a
time-based loop (cosine LR over the budget), RF-DETR Lightning `max_time` (see above).

## Step 1 - synthetic scenes

Scene: 1-10 glass slides (75 x 25 mm, portrait, one row, label always at the top end)
with H&E tissue and a 25 x 20 mm matte label (3-4 random text lines + code), lying on a
randomly chosen surface (`scene.surface.types` / `weights`, recorded in `metadata.csv`
as `surface`, `surface_variant`):

| surface | look | slide placement |
|---|---|---|
| `pla_base` (real rig) | flat light-grey PLA plate 380 x 150 mm, faint layer lines, seam in the middle, engraved 300 x 90 mm frame, 10 ticks (6 mm, straddling the line) at 28 mm on its top edge; no grooves, no stops | free inside the frame, gaps 2-10 mm, tilt <= 5 deg (`scene.pla_base`) |
| `steel_tray` | brushed stainless, slot markers, holes, ruler ticks | slots, 28 mm pitch, tilt <= 2.5 deg |
| `plastic_tray` | white / grey, rim and slot ribs | slots, 28 mm pitch, tilt <= 2.5 deg |
| `desk` | laminate white / grey / beige, light / dark wood | slots, 28 mm pitch, tilt <= 2.5 deg |
| `paper` | white sheet, yellow or manila folder | slots, 28 mm pitch, tilt <= 2.5 deg |
| `slide_box` | black / grey (implemented, not in the default mix) | slots |

Slides never overlap; every layout is checked (`scene.check_layout`) before rendering.
The `rig` camera looks straight down (tilt <= 1 deg mounting tolerance) from above the
middle of the row, like the bridge rig; `handheld` is oblique.

Around it: papers/folders in a white wire basket.
All slides in a row share one orientation; larger rotation comes only from the camera.
Optional loose paper edges; the hand generator exists but is off (`scene.hand.prob: 0`).
Two cameras: `rig` (main set) and `handheld` (robustness test set only).

Every object is drawn in its own millimetre frame and warped to the image once with
the full pinhole-camera homography, so the reported corners are the exact projection
of the printed symbol corners (sub-pixel; see `verify_clean.py`).

### Commands

```bash
# review pack: 12 scenes, overlays, zoomed corner crops, clean decode check
.venv\Scripts\python scripts\preview.py --out runs\preview_step1

# generator self-test on clean renders (must print 100% rect_decode, 0 bit errors)
.venv\Scripts\python scripts\verify_clean.py --n 150
.venv\Scripts\python scripts\verify_clean.py --n 150 --set code.symbology=datamatrix

# full dataset (rig train/val/test + handheld test)
.venv\Scripts\python scripts\generate.py --out data\synth_v1 --n-rig 3000 --n-handheld 600 --workers 8

# ablation: any config key can be overridden
.venv\Scripts\python scripts\generate.py --out data\abl_noglare --set degrade.glare.enabled=false
```

Same config + same seed => identical dataset (each image has its own RNG seeded by
`(seed, mode, index)`). The resolved config is saved as `<out>/config.yaml`.

### Output layout

```
data/synth_v1/
  images/{train,val,test,test_handheld}/*.jpg   (.png when JPEG degradation is off)
  labels/{split}/*.txt          YOLO-seg: class x1 y1 ... (normalised); 0=code 1=slide
  annotations/instances_{split}.json   COCO; extra per-annotation fields:
                                corners (amodal TL,TR,BR,BL), payload, symbology, ppm
  metadata.csv                  one row per code (see below)
  data.yaml                     Ultralytics dataset file
  config.yaml, MANIFEST.txt
```

### Label conventions

* `code` polygon = 4 corners, **TL of the symbol itself** (matrix row 0 / col 0) then
  clockwise, quiet zone excluded. When the camera rolls, TL follows the symbol, not the image.
* Coordinates are continuous pixel coordinates with (0,0) at the top-left *edge* of
  the image (COCO/YOLO convention).
* Codes cut by the frame get the clipped polygon (> 4 points) and
  `flag_out_of_frame=1`; their full corners stay in COCO `corners` / metadata.
* `visible_frac < labels.min_visible_frac` (0.5): `ignore=1` -> COCO `iscrowd=1`
  (ignored in AP), left out of YOLO labels (YOLO has no ignore regions).
* `slide` class can be turned off: `--set labels.include_slide_class=false`.

### metadata.csv columns

payload, symbology, modules, `ppm` (= sqrt(quad area)/modules, measured on the
rendered corners), `ppm_min_side`/`ppm_max_side`, `in_frame_frac`, `occluded_frac`,
`visible_frac`, flags, corners `c0x..c3y`, camera (`cam_*`), every degradation
parameter (`deg_*`, 0 = not applied), label stain/wrinkle strength, hand / paper edge.

### Splits

`rig` images get stratified target ppm over `scale.ppm_range` and are split
70/15/15 by walking them in ppm order and giving each image to the split furthest
behind its quota, so train/val/test have the same ppm distribution and exact sizes. `handheld` images all go to `test_handheld`.

Note: in `handheld` the ppm varies across the image (perspective), so measured
per-code ppm can fall below 2.0; filter on `metadata.csv:ppm` when binning.

## Values to change after measuring real labels

| key in `configs/synth.yaml` | now | what to measure |
|---|---|---|
| `code.CODE_SIZE_MM` | 6.5 (estimated from photos) | symbol edge length without quiet zone (mm) |
| `code.CODE_MODULES.qr` / `.datamatrix` | 21 / 18 | modules per side (count on a macro photo) |
| `code.symbology` | datamatrix | `qr`, `datamatrix` or `mixed` |
| `scene.slide_pitch_mm` | 28 | slot pitch of the tray-type surfaces |
| `scene.pla_base.*` | 380 x 150 plate, 300 x 90 frame, gaps 2-10, tilt 5 | real PLA base dimensions |
| `code.qr_error` | m | QR error-correction level printed by the label printer |
| `code.payload` | sample_id | what the code actually encodes (format only, never real IDs) |
| `scene.label_mm` | 25 x 20 | label size |
| `camera.image_size` | 3264 x 2448 | output resolution (any camera; not tied to a model) |
| `camera.rig.hfov_deg` | 50-65 | rig lens |
| `scale.ppm_range` | 2-8 | keep, or narrow around the real rig's ppm |

Choosing a camera: `scripts/camera_table.py` writes `docs/camera_table.md` (+ `.csv`),
ppm for code sizes 5-10 mm, QR 21 / DataMatrix 18, 1/5/10 slides per image, and the
image width needed for 3 and 4 ppm. Re-run it after changing the config.
With 3264 px, 6.5 mm DataMatrix 18 and 10 slides (~287 mm) the maximum is ~4.1 ppm,
so high-ppm synthetic scenes show fewer slides.

## Project rules

* Never bake config values into function argument defaults (use `None` and read
  the config at call time) so ablation overrides take effect.
* Every reported number is labelled synthetic until tested on real images.
* Odd results -> look for a bug before drawing conclusions.
