# Statistics on the SYNTHETIC test sets (one training run per model)

All numbers come from synthetic renders; they are not real-photo results.

* 95% CI: cluster bootstrap over images (2000 resamples, seed 0); codes in one image are not independent, so whole images are resampled.
* Paired tests on the same codes: exact McNemar (b = only the first read it, c = only the second), Holm-corrected within each table, alpha = 0.05.
* Each model was trained once (one seed), so these intervals cover test-set variability, not training variability (RM07: bootstrap CI is not training variance).

## rig (test)

### Read rate, all

| model | codes | full_image | bbox_crop | quad_warp | warp_then_crop |
|---|---|---|---|---|---|
| maskrcnn | 2029 | 80.7 [77.4, 84.0] | 87.8 [85.3, 90.4] | 84.2 [81.7, 86.6] | 94.1 [91.9, 96.1] |
| yolo11n | 2029 | 80.7 [77.3, 84.0] | 89.0 [86.5, 91.3] | 87.1 [84.5, 89.5] | 94.9 [92.8, 96.8] |
| yolo26n | 2029 | 80.7 [77.3, 83.9] | 88.5 [85.9, 90.9] | 88.4 [86.0, 90.6] | 94.7 [92.6, 96.7] |
| rfdetr | 2029 | 80.7 [77.3, 83.9] | 88.7 [86.0, 91.0] | 58.8 [54.7, 63.0] | 92.7 [90.4, 94.8] |
| rfdetr_long | 2029 | 80.7 [77.2, 84.3] | 89.1 [86.5, 91.4] | 61.6 [57.4, 65.7] | 93.1 [90.7, 95.3] |
| oracle | 2029 | 80.7 [77.2, 83.9] | 88.2 [85.6, 90.7] | 93.9 [91.7, 96.0] | 94.7 [92.5, 96.7] |

### Read rate, 8MP [3.5,4)

| model | codes | full_image | bbox_crop | quad_warp | warp_then_crop |
|---|---|---|---|---|---|
| maskrcnn | 209 | 64.1 [51.2, 77.5] | 87.1 [78.5, 94.2] | 89.0 [82.9, 94.1] | 96.7 [92.8, 99.1] |
| yolo11n | 209 | 64.1 [51.7, 77.9] | 88.5 [81.6, 94.4] | 88.0 [81.2, 93.7] | 95.7 [91.5, 99.1] |
| yolo26n | 209 | 64.1 [51.7, 76.4] | 85.6 [77.2, 93.3] | 86.6 [80.2, 92.4] | 95.7 [91.7, 99.0] |
| rfdetr | 209 | 64.1 [50.9, 77.2] | 86.6 [78.8, 93.7] | 35.9 [29.7, 41.9] | 89.0 [81.8, 94.6] |
| rfdetr_long | 209 | 64.1 [51.6, 77.7] | 90.0 [83.2, 95.1] | 42.6 [33.5, 51.3] | 91.9 [85.4, 96.7] |
| oracle | 209 | 64.1 [51.6, 77.6] | 88.0 [79.1, 95.2] | 94.7 [89.8, 98.8] | 96.7 [92.2, 100.0] |

### Read rate, 12MP [4.5,5)

| model | codes | full_image | bbox_crop | quad_warp | warp_then_crop |
|---|---|---|---|---|---|
| maskrcnn | 165 | 79.4 [68.4, 88.6] | 93.9 [88.0, 98.7] | 81.8 [75.5, 87.7] | 97.0 [92.6, 100.0] |
| yolo11n | 165 | 79.4 [69.0, 88.4] | 96.4 [91.5, 100.0] | 96.4 [92.8, 99.3] | 99.4 [98.0, 100.0] |
| yolo26n | 165 | 79.4 [68.9, 88.7] | 95.2 [89.4, 99.3] | 96.4 [93.3, 98.8] | 99.4 [98.2, 100.0] |
| rfdetr | 165 | 79.4 [68.9, 88.4] | 94.5 [89.0, 99.3] | 75.2 [67.7, 82.5] | 98.8 [96.1, 100.0] |
| rfdetr_long | 165 | 79.4 [69.2, 88.2] | 95.2 [89.8, 99.4] | 78.8 [72.9, 84.7] | 98.8 [97.0, 100.0] |
| oracle | 165 | 79.4 [68.8, 88.3] | 94.5 [88.8, 98.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |

### Decoder inputs within each model, all (RQ2)

| model | comparison | b | c | difference, pp [95% CI] | p (Holm) | reject H0 |
|---|---|---|---|---|---|---|
| maskrcnn | warp_then_crop vs bbox_crop | 128 | 0 | +6.3 [+4.9, +7.7] | < 0.001 | yes |
| maskrcnn | warp_then_crop vs quad_warp | 201 | 0 | +9.9 [+8.3, +11.5] | < 0.001 | yes |
| yolo11n | warp_then_crop vs bbox_crop | 121 | 0 | +6.0 [+4.6, +7.3] | < 0.001 | yes |
| yolo11n | warp_then_crop vs quad_warp | 159 | 0 | +7.8 [+6.3, +9.5] | < 0.001 | yes |
| yolo26n | warp_then_crop vs bbox_crop | 126 | 0 | +6.2 [+4.9, +7.6] | < 0.001 | yes |
| yolo26n | warp_then_crop vs quad_warp | 129 | 0 | +6.4 [+5.1, +7.8] | < 0.001 | yes |
| rfdetr | warp_then_crop vs bbox_crop | 81 | 0 | +4.0 [+3.0, +5.1] | < 0.001 | yes |
| rfdetr | warp_then_crop vs quad_warp | 687 | 0 | +33.9 [+30.2, +37.5] | < 0.001 | yes |
| rfdetr_long | warp_then_crop vs bbox_crop | 82 | 0 | +4.0 [+3.0, +5.3] | < 0.001 | yes |
| rfdetr_long | warp_then_crop vs quad_warp | 640 | 0 | +31.5 [+27.6, +35.2] | < 0.001 | yes |

### Decoder inputs within each model, 8MP [3.5,4) (RQ2)

| model | comparison | b | c | difference, pp [95% CI] | p (Holm) | reject H0 |
|---|---|---|---|---|---|---|
| maskrcnn | warp_then_crop vs bbox_crop | 20 | 0 | +9.6 [+4.6, +15.6] | < 0.001 | yes |
| maskrcnn | warp_then_crop vs quad_warp | 16 | 0 | +7.7 [+3.6, +12.4] | < 0.001 | yes |
| yolo11n | warp_then_crop vs bbox_crop | 15 | 0 | +7.2 [+3.5, +11.1] | < 0.001 | yes |
| yolo11n | warp_then_crop vs quad_warp | 16 | 0 | +7.7 [+4.0, +11.5] | < 0.001 | yes |
| yolo26n | warp_then_crop vs bbox_crop | 21 | 0 | +10.0 [+5.1, +15.8] | < 0.001 | yes |
| yolo26n | warp_then_crop vs quad_warp | 19 | 0 | +9.1 [+5.0, +13.7] | < 0.001 | yes |
| rfdetr | warp_then_crop vs bbox_crop | 5 | 0 | +2.4 [+0.5, +4.5] | 0.125 | no |
| rfdetr | warp_then_crop vs quad_warp | 111 | 0 | +53.1 [+47.3, +59.6] | < 0.001 | yes |
| rfdetr_long | warp_then_crop vs bbox_crop | 4 | 0 | +1.9 [+0.4, +3.8] | 0.125 | no |
| rfdetr_long | warp_then_crop vs quad_warp | 103 | 0 | +49.3 [+41.8, +56.7] | < 0.001 | yes |

### Decoder inputs within each model, 12MP [4.5,5) (RQ2)

| model | comparison | b | c | difference, pp [95% CI] | p (Holm) | reject H0 |
|---|---|---|---|---|---|---|
| maskrcnn | warp_then_crop vs bbox_crop | 5 | 0 | +3.0 [+0.6, +6.4] | 0.250 | no |
| maskrcnn | warp_then_crop vs quad_warp | 25 | 0 | +15.2 [+10.1, +21.0] | < 0.001 | yes |
| yolo11n | warp_then_crop vs bbox_crop | 5 | 0 | +3.0 [+0.0, +6.8] | 0.250 | no |
| yolo11n | warp_then_crop vs quad_warp | 5 | 0 | +3.0 [+0.6, +6.3] | 0.250 | no |
| yolo26n | warp_then_crop vs bbox_crop | 7 | 0 | +4.2 [+0.7, +8.9] | 0.109 | no |
| yolo26n | warp_then_crop vs quad_warp | 5 | 0 | +3.0 [+0.6, +5.9] | 0.250 | no |
| rfdetr | warp_then_crop vs bbox_crop | 7 | 0 | +4.2 [+0.7, +8.2] | 0.109 | no |
| rfdetr | warp_then_crop vs quad_warp | 39 | 0 | +23.6 [+16.9, +30.5] | < 0.001 | yes |
| rfdetr_long | warp_then_crop vs bbox_crop | 6 | 0 | +3.6 [+0.0, +8.2] | 0.156 | no |
| rfdetr_long | warp_then_crop vs quad_warp | 33 | 0 | +20.0 [+14.0, +26.4] | < 0.001 | yes |

### Models with warp_then_crop, all (RQ1)

Cochran's Q over 4 models: Q = 51.52, df = 3, p = < 0.001 (2029 codes)

| pair | b | c | difference, pp [95% CI] | p (Holm) | reject H0 |
|---|---|---|---|---|---|
| maskrcnn vs yolo11n | 15 | 31 | -0.8 [-1.5, -0.1] | 0.078 | no |
| maskrcnn vs yolo26n | 16 | 28 | -0.6 [-1.3, -0.0] | 0.192 | no |
| maskrcnn vs rfdetr | 47 | 18 | +1.4 [+0.6, +2.3] | 0.002 | yes |
| yolo11n vs yolo26n | 14 | 10 | +0.2 [-0.2, +0.7] | 0.541 | no |
| yolo11n vs rfdetr | 50 | 5 | +2.2 [+1.3, +3.3] | < 0.001 | yes |
| yolo26n vs rfdetr | 48 | 7 | +2.0 [+1.2, +2.9] | < 0.001 | yes |

### Models with warp_then_crop, 8MP [3.5,4) (RQ1)

Cochran's Q over 4 models: Q = 23.43, df = 3, p = < 0.001 (209 codes)

| pair | b | c | difference, pp [95% CI] | p (Holm) | reject H0 |
|---|---|---|---|---|---|
| maskrcnn vs yolo11n | 6 | 4 | +1.0 [-1.0, +3.4] | 1.000 | no |
| maskrcnn vs yolo26n | 6 | 4 | +1.0 [-0.6, +2.9] | 1.000 | no |
| maskrcnn vs rfdetr | 19 | 3 | +7.7 [+3.6, +12.5] | 0.005 | yes |
| yolo11n vs yolo26n | 3 | 3 | +0.0 [-1.4, +1.3] | 1.000 | no |
| yolo11n vs rfdetr | 16 | 2 | +6.7 [+2.3, +11.7] | 0.007 | yes |
| yolo26n vs rfdetr | 16 | 2 | +6.7 [+2.7, +11.4] | 0.007 | yes |

### Models with warp_then_crop, 12MP [4.5,5) (RQ1)

Cochran's Q over 4 models: Q = 9.92, df = 3, p = 0.019 (165 codes)

| pair | b | c | difference, pp [95% CI] | p (Holm) | reject H0 |
|---|---|---|---|---|---|
| maskrcnn vs yolo11n | 0 | 4 | -2.4 [-5.4, +0.0] | 0.750 | no |
| maskrcnn vs yolo26n | 0 | 4 | -2.4 [-5.4, +0.0] | 0.750 | no |
| maskrcnn vs rfdetr | 0 | 3 | -1.8 [-3.9, +0.0] | 1.000 | no |
| yolo11n vs yolo26n | 0 | 0 | +0.0 [+0.0, +0.0] | 1.000 | no |
| yolo11n vs rfdetr | 1 | 0 | +0.6 [+0.0, +1.9] | 1.000 | no |
| yolo26n vs rfdetr | 1 | 0 | +0.6 [+0.0, +2.0] | 1.000 | no |

## handheld (test_handheld)

### Read rate, all

| model | codes | full_image | bbox_crop | quad_warp | warp_then_crop |
|---|---|---|---|---|---|
| maskrcnn | 2725 | 65.2 [61.7, 68.8] | 73.4 [70.2, 76.8] | 19.9 [17.5, 22.3] | 74.5 [71.3, 78.0] |
| yolo11n | 2725 | 65.2 [61.5, 68.7] | 71.9 [68.3, 75.5] | 27.5 [25.0, 30.2] | 73.7 [70.6, 77.0] |
| yolo26n | 2725 | 65.2 [61.7, 68.7] | 72.9 [69.4, 76.1] | 36.8 [33.9, 39.8] | 75.2 [72.0, 78.4] |
| rfdetr | 2725 | 65.2 [61.7, 68.7] | 74.0 [70.8, 77.3] | 27.6 [24.9, 30.2] | 75.6 [72.3, 78.9] |
| rfdetr_long | 2725 | 65.2 [61.7, 68.7] | 72.9 [69.4, 76.3] | 23.4 [20.7, 26.0] | 74.6 [71.2, 78.0] |
| oracle | 2725 | 65.2 [61.7, 68.8] | 73.3 [70.0, 76.5] | 78.7 [75.3, 82.0] | 80.9 [77.7, 84.0] |

### Read rate, 8MP [3.5,4)

| model | codes | full_image | bbox_crop | quad_warp | warp_then_crop |
|---|---|---|---|---|---|
| maskrcnn | 267 | 59.2 [51.3, 66.7] | 80.5 [74.5, 85.9] | 19.9 [14.4, 26.2] | 81.6 [75.8, 87.2] |
| yolo11n | 267 | 59.2 [51.2, 66.8] | 75.7 [69.3, 82.0] | 21.7 [16.2, 27.7] | 77.5 [71.1, 83.8] |
| yolo26n | 267 | 59.2 [51.2, 66.8] | 76.0 [69.0, 82.1] | 22.8 [16.5, 29.2] | 78.3 [71.9, 84.2] |
| rfdetr | 267 | 59.2 [50.9, 66.9] | 77.5 [71.2, 83.5] | 10.9 [7.1, 15.2] | 77.9 [72.3, 83.8] |
| rfdetr_long | 267 | 59.2 [51.1, 66.9] | 77.5 [71.4, 83.2] | 6.4 [3.5, 9.8] | 77.9 [71.4, 83.7] |
| oracle | 267 | 59.2 [50.8, 67.0] | 74.2 [67.4, 80.2] | 80.1 [73.7, 85.8] | 82.8 [76.8, 87.9] |

### Read rate, 12MP [4.5,5)

| model | codes | full_image | bbox_crop | quad_warp | warp_then_crop |
|---|---|---|---|---|---|
| maskrcnn | 245 | 66.9 [60.1, 73.7] | 89.4 [84.7, 93.6] | 23.3 [17.0, 29.3] | 90.6 [86.3, 94.5] |
| yolo11n | 245 | 66.9 [59.8, 74.1] | 89.0 [84.2, 93.2] | 42.4 [35.7, 49.0] | 91.8 [88.1, 95.2] |
| yolo26n | 245 | 66.9 [59.8, 74.0] | 89.4 [85.2, 93.6] | 44.9 [38.0, 51.5] | 91.8 [87.9, 95.2] |
| rfdetr | 245 | 66.9 [59.8, 74.0] | 91.0 [86.9, 94.9] | 29.8 [23.7, 36.6] | 92.2 [88.6, 95.5] |
| rfdetr_long | 245 | 66.9 [59.6, 74.0] | 88.6 [83.6, 93.1] | 30.2 [24.0, 36.2] | 90.6 [86.4, 94.4] |
| oracle | 245 | 66.9 [59.8, 74.0] | 89.0 [84.6, 93.1] | 93.9 [90.5, 96.8] | 97.1 [94.7, 99.2] |

### Decoder inputs within each model, all (RQ2)

| model | comparison | b | c | difference, pp [95% CI] | p (Holm) | reject H0 |
|---|---|---|---|---|---|---|
| maskrcnn | warp_then_crop vs bbox_crop | 30 | 0 | +1.1 [+0.7, +1.6] | < 0.001 | yes |
| maskrcnn | warp_then_crop vs quad_warp | 1489 | 0 | +54.6 [+51.5, +58.0] | < 0.001 | yes |
| yolo11n | warp_then_crop vs bbox_crop | 49 | 0 | +1.8 [+1.3, +2.4] | < 0.001 | yes |
| yolo11n | warp_then_crop vs quad_warp | 1259 | 0 | +46.2 [+43.3, +49.3] | < 0.001 | yes |
| yolo26n | warp_then_crop vs bbox_crop | 64 | 0 | +2.3 [+1.7, +3.1] | < 0.001 | yes |
| yolo26n | warp_then_crop vs quad_warp | 1048 | 0 | +38.5 [+35.6, +41.4] | < 0.001 | yes |
| rfdetr | warp_then_crop vs bbox_crop | 42 | 0 | +1.5 [+1.1, +2.1] | < 0.001 | yes |
| rfdetr | warp_then_crop vs quad_warp | 1307 | 0 | +48.0 [+45.0, +51.2] | < 0.001 | yes |
| rfdetr_long | warp_then_crop vs bbox_crop | 46 | 0 | +1.7 [+1.1, +2.3] | < 0.001 | yes |
| rfdetr_long | warp_then_crop vs quad_warp | 1395 | 0 | +51.2 [+47.8, +54.5] | < 0.001 | yes |

### Decoder inputs within each model, 8MP [3.5,4) (RQ2)

| model | comparison | b | c | difference, pp [95% CI] | p (Holm) | reject H0 |
|---|---|---|---|---|---|---|
| maskrcnn | warp_then_crop vs bbox_crop | 3 | 0 | +1.1 [+0.0, +2.6] | 0.750 | no |
| maskrcnn | warp_then_crop vs quad_warp | 165 | 0 | +61.8 [+54.5, +68.6] | < 0.001 | yes |
| yolo11n | warp_then_crop vs bbox_crop | 5 | 0 | +1.9 [+0.4, +3.6] | 0.250 | no |
| yolo11n | warp_then_crop vs quad_warp | 149 | 0 | +55.8 [+48.9, +62.1] | < 0.001 | yes |
| yolo26n | warp_then_crop vs bbox_crop | 6 | 0 | +2.2 [+0.4, +4.6] | 0.156 | no |
| yolo26n | warp_then_crop vs quad_warp | 148 | 0 | +55.4 [+48.5, +62.5] | < 0.001 | yes |
| rfdetr | warp_then_crop vs bbox_crop | 1 | 0 | +0.4 [+0.0, +1.2] | 1.000 | no |
| rfdetr | warp_then_crop vs quad_warp | 179 | 0 | +67.0 [+60.2, +73.3] | < 0.001 | yes |
| rfdetr_long | warp_then_crop vs bbox_crop | 1 | 0 | +0.4 [+0.0, +1.2] | 1.000 | no |
| rfdetr_long | warp_then_crop vs quad_warp | 191 | 0 | +71.5 [+64.9, +77.6] | < 0.001 | yes |

### Decoder inputs within each model, 12MP [4.5,5) (RQ2)

| model | comparison | b | c | difference, pp [95% CI] | p (Holm) | reject H0 |
|---|---|---|---|---|---|---|
| maskrcnn | warp_then_crop vs bbox_crop | 3 | 0 | +1.2 [+0.0, +3.4] | 0.500 | no |
| maskrcnn | warp_then_crop vs quad_warp | 165 | 0 | +67.3 [+60.4, +74.3] | < 0.001 | yes |
| yolo11n | warp_then_crop vs bbox_crop | 7 | 0 | +2.9 [+0.5, +5.8] | 0.078 | no |
| yolo11n | warp_then_crop vs quad_warp | 121 | 0 | +49.4 [+42.9, +55.9] | < 0.001 | yes |
| yolo26n | warp_then_crop vs bbox_crop | 6 | 0 | +2.4 [+0.4, +4.9] | 0.125 | no |
| yolo26n | warp_then_crop vs quad_warp | 115 | 0 | +46.9 [+40.0, +53.7] | < 0.001 | yes |
| rfdetr | warp_then_crop vs bbox_crop | 3 | 0 | +1.2 [+0.0, +2.8] | 0.500 | no |
| rfdetr | warp_then_crop vs quad_warp | 153 | 0 | +62.4 [+55.3, +69.7] | < 0.001 | yes |
| rfdetr_long | warp_then_crop vs bbox_crop | 5 | 0 | +2.0 [+0.0, +4.6] | 0.188 | no |
| rfdetr_long | warp_then_crop vs quad_warp | 148 | 0 | +60.4 [+53.9, +67.1] | < 0.001 | yes |

### Models with warp_then_crop, all (RQ1)

Cochran's Q over 4 models: Q = 14.08, df = 3, p = 0.003 (2725 codes)

| pair | b | c | difference, pp [95% CI] | p (Holm) | reject H0 |
|---|---|---|---|---|---|
| maskrcnn vs yolo11n | 121 | 99 | +0.8 [-0.3, +1.9] | 0.470 | no |
| maskrcnn vs yolo26n | 104 | 123 | -0.7 [-1.9, +0.4] | 0.470 | no |
| maskrcnn vs rfdetr | 81 | 109 | -1.0 [-2.0, -0.1] | 0.199 | no |
| yolo11n vs yolo26n | 77 | 118 | -1.5 [-2.5, -0.5] | 0.020 | yes |
| yolo11n vs rfdetr | 87 | 137 | -1.8 [-3.1, -0.6] | 0.006 | yes |
| yolo26n vs rfdetr | 95 | 104 | -0.3 [-1.4, +0.8] | 0.571 | no |

### Models with warp_then_crop, 8MP [3.5,4) (RQ1)

Cochran's Q over 4 models: Q = 5.50, df = 3, p = 0.139 (267 codes)

| pair | b | c | difference, pp [95% CI] | p (Holm) | reject H0 |
|---|---|---|---|---|---|
| maskrcnn vs yolo11n | 17 | 6 | +4.1 [+0.0, +8.2] | 0.208 | no |
| maskrcnn vs yolo26n | 17 | 8 | +3.4 [-0.4, +7.2] | 0.431 | no |
| maskrcnn vs rfdetr | 16 | 6 | +3.7 [+0.7, +6.9] | 0.262 | no |
| yolo11n vs yolo26n | 13 | 15 | -0.7 [-4.3, +2.6] | 1.000 | no |
| yolo11n vs rfdetr | 16 | 17 | -0.4 [-4.9, +4.2] | 1.000 | no |
| yolo26n vs rfdetr | 19 | 18 | +0.4 [-4.0, +5.3] | 1.000 | no |

### Models with warp_then_crop, 12MP [4.5,5) (RQ1)

Cochran's Q over 4 models: Q = 1.17, df = 3, p = 0.759 (245 codes)

| pair | b | c | difference, pp [95% CI] | p (Holm) | reject H0 |
|---|---|---|---|---|---|
| maskrcnn vs yolo11n | 7 | 10 | -1.2 [-4.8, +2.3] | 1.000 | no |
| maskrcnn vs yolo26n | 7 | 10 | -1.2 [-5.1, +2.5] | 1.000 | no |
| maskrcnn vs rfdetr | 3 | 7 | -1.6 [-4.5, +0.8] | 1.000 | no |
| yolo11n vs yolo26n | 6 | 6 | +0.0 [-2.6, +2.3] | 1.000 | no |
| yolo11n vs rfdetr | 9 | 10 | -0.4 [-3.8, +3.1] | 1.000 | no |
| yolo26n vs rfdetr | 8 | 9 | -0.4 [-3.6, +2.8] | 1.000 | no |

