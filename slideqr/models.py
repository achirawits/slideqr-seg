"""One predictor interface for every model.

predict(img_bgr) -> list of det records (evaluation.det_to_record) with masks at the
ORIGINAL image resolution, so every model is scored and decoded the same way.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .evaluation import det_to_record

CLASSES = ["code", "slide"]


def long_side_scale(w, h, imgsz):
    return imgsz / max(w, h)


# ---------------------------------------------------------------- Mask R-CNN (torchvision)

def build_maskrcnn(num_classes, imgsz, pretrained=True):
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    # images are resized to `imgsz` on the long side before they reach the model, so the
    # internal transform must not resize again: min/max size set to the same scale
    model = maskrcnn_resnet50_fpn(weights=weights, min_size=int(round(imgsz * 0.75)), max_size=imgsz,
                                  box_detections_per_img=100)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes + 1)
    in_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, num_classes + 1)
    del torchvision
    return model


class MaskRCNNPredictor_:
    name = "maskrcnn"

    def __init__(self, weights_path, imgsz, device, score_floor, max_det):
        import torch
        self.torch = torch
        self.imgsz, self.device, self.score_floor, self.max_det = imgsz, device, score_floor, max_det
        self.model = build_maskrcnn(len(CLASSES), imgsz, pretrained=False)
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.to(device).eval()

    def predict(self, img_bgr):
        torch = self.torch
        H, W = img_bgr.shape[:2]
        s = long_side_scale(W, H, self.imgsz)
        small = cv2.resize(img_bgr, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small[..., ::-1].copy()).permute(2, 0, 1).float().div(255).to(self.device)
        with torch.inference_mode():
            out = self.model([x])[0]
        keep = (out["scores"] >= self.score_floor).nonzero().flatten()[: self.max_det]
        masks = out["masks"][keep, 0].float().cpu().numpy()
        recs = []
        for m, sc, lb in zip(masks, out["scores"][keep].tolist(), out["labels"][keep].tolist()):
            full = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
            recs.append(det_to_record(sc, CLASSES[lb - 1], full))
        return recs

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())


# ---------------------------------------------------------------- Ultralytics YOLO

class YoloPredictor:
    def __init__(self, name, weights_path, imgsz, device, score_floor, max_det):
        from ultralytics import YOLO
        self.name = name
        self.model = YOLO(str(weights_path))
        self.imgsz, self.device, self.score_floor, self.max_det = imgsz, device, score_floor, max_det

    def predict(self, img_bgr):
        r = self.model.predict(img_bgr, imgsz=self.imgsz, conf=self.score_floor, max_det=self.max_det,
                               retina_masks=True, device=self.device, verbose=False)[0]
        if r.masks is None:
            return []
        masks = r.masks.data.cpu().numpy() > 0.5      # (N, H, W) at the original size
        H, W = img_bgr.shape[:2]
        recs = []
        for m, sc, c in zip(masks, r.boxes.conf.tolist(), r.boxes.cls.int().tolist()):
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            recs.append(det_to_record(sc, CLASSES[c], m))
        return recs

    def n_params(self):
        return sum(p.numel() for p in self.model.model.parameters())


# ---------------------------------------------------------------- RF-DETR

class RFDETRPredictor:
    name = "rfdetr"

    def __init__(self, model_obj, class_map, device, score_floor, max_det):
        self.model = model_obj
        self.class_map = class_map          # rfdetr class id -> our class name
        self.device, self.score_floor, self.max_det = device, score_floor, max_det

    def predict(self, img_bgr):
        from PIL import Image
        det = self.model.predict(Image.fromarray(img_bgr[..., ::-1]), threshold=self.score_floor)
        if det.mask is None or len(det) == 0:
            return []
        order = np.argsort(-det.confidence)[: self.max_det]
        H, W = img_bgr.shape[:2]
        recs = []
        for i in order:
            name = self.class_map.get(int(det.class_id[i]))
            if name is None:
                continue
            m = det.mask[i]
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            recs.append(det_to_record(float(det.confidence[i]), name, m))
        return recs

    def n_params(self):
        inner = getattr(self.model, "model", None)
        net = getattr(inner, "model", inner)
        return sum(p.numel() for p in net.parameters()) if net is not None else -1


# ---------------------------------------------------------------- timing

def time_predict(pred, images, device_sync=None, warmup=3):
    for im in images[:warmup]:
        pred.predict(im)
    t = []
    for im in images:
        if device_sync:
            device_sync()
        t0 = time.perf_counter()
        pred.predict(im)
        if device_sync:
            device_sync()
        t.append(time.perf_counter() - t0)
    return dict(mean_s=float(np.mean(t)), median_s=float(np.median(t)), n=len(t))


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1), encoding="utf-8")
