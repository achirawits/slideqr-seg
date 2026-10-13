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


def area_resize_long(img, imgsz):
    """Antialiased (INTER_AREA) downscale to the network input size (Mask R-CNN; RF-DETR
    uses the square variant). YOLO keeps its native pipeline, see YoloPredictor.infer."""
    H, W = img.shape[:2]
    s = long_side_scale(W, H, imgsz)
    if s >= 1:
        return img
    return cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)


def upscale_mask(m, W, H):
    """Mask (probability or 0/1) at network size -> full-resolution bool, the same way for every model."""
    m = np.asarray(m, np.float32)
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
    return m > 0.5


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

    def infer(self, img_bgr):
        """Resize + forward pass + the model's own post-processing (network resolution)."""
        torch = self.torch
        small = area_resize_long(img_bgr, self.imgsz)
        x = torch.from_numpy(small[..., ::-1].copy()).permute(2, 0, 1).float().div(255).to(self.device)
        with torch.inference_mode():
            return self.model([x])[0]

    def predict(self, img_bgr):
        H, W = img_bgr.shape[:2]
        out = self.infer(img_bgr)
        keep = (out["scores"] >= self.score_floor).nonzero().flatten()[: self.max_det]
        masks = out["masks"][keep, 0].float().cpu().numpy()
        recs = []
        for m, sc, lb in zip(masks, out["scores"][keep].tolist(), out["labels"][keep].tolist()):
            recs.append(det_to_record(sc, CLASSES[lb - 1], upscale_mask(m, W, H)))
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

    def infer(self, img_bgr, retina=False):
        """Ultralytics' own pipeline on the full image: letterbox, forward, NMS, masks.

        Kept native on purpose: with retina_masks the mask logits are upsampled to the
        original resolution, which gives the sharpest boundary. An INTER_AREA pre-resize
        was tried and made the masks coarser (corner error 1.5 -> 2.4 px) while YOLO had
        no aliasing failure to fix (0 test images without a code prediction)."""
        return self.model.predict(img_bgr, imgsz=self.imgsz, conf=self.score_floor, max_det=self.max_det,
                                  retina_masks=retina, device=self.device, verbose=False)[0]

    def predict(self, img_bgr):
        r = self.infer(img_bgr, retina=True)            # masks at the original image size
        if r.masks is None:
            return []
        H, W = img_bgr.shape[:2]
        masks = r.masks.data.cpu().numpy()
        return [det_to_record(sc, CLASSES[c], upscale_mask(m, W, H))
                for m, sc, c in zip(masks, r.boxes.conf.tolist(), r.boxes.cls.int().tolist())]

    def n_params(self):
        return sum(p.numel() for p in self.model.model.parameters())


# ---------------------------------------------------------------- RF-DETR

class RFDETRPredictor:
    name = "rfdetr"

    def __init__(self, model_obj, class_map, device, score_floor, max_det, resolution):
        self.model = model_obj
        self.class_map = class_map          # rfdetr class id -> our class name
        self.device, self.score_floor, self.max_det = device, score_floor, max_det
        self.resolution = int(resolution)

    def infer(self, img_bgr):
        """Area-downscale to the square training resolution, then rfdetr's predict.

        Feeding the full 3264 px image lets predict() downsample it itself; on sharp,
        noisy images that aliased so badly the model found no code at all (46/46 codes
        recovered on the affected images once the image is area-downscaled first)."""
        from PIL import Image
        small = cv2.resize(img_bgr, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        return self.model.predict(Image.fromarray(small[..., ::-1]), threshold=self.score_floor)

    def predict(self, img_bgr):
        det = self.infer(img_bgr)
        if det.mask is None or len(det) == 0:
            return []
        order = np.argsort(-det.confidence)[: self.max_det]
        H, W = img_bgr.shape[:2]
        recs = []
        for i in order:
            name = self.class_map.get(int(det.class_id[i]))
            if name is None:
                continue
            recs.append(det_to_record(float(det.confidence[i]), name, upscale_mask(det.mask[i], W, H)))
        return recs

    def n_params(self):
        inner = getattr(self.model, "model", None)
        net = getattr(inner, "model", inner)
        return sum(p.numel() for p in net.parameters()) if net is not None else -1


# ---------------------------------------------------------------- timing

def time_predict(pred, images, device_sync=None, warmup=3):
    """Two numbers per image, one image at a time:
    model_s     - pred.infer: pre-processing, forward pass and the model's own
                  post-processing at network resolution (comparable across models)
    pipeline_s  - pred.predict: also full-resolution masks, RLE and contours
                  (CPU-bound, the same work for every model)"""
    def run(fn):
        for im in images[:warmup]:
            fn(im)
        t = []
        for im in images:
            if device_sync:
                device_sync()
            t0 = time.perf_counter()
            fn(im)
            if device_sync:
                device_sync()
            t.append(time.perf_counter() - t0)
        return t
    tm, tp = run(pred.infer), run(pred.predict)
    return dict(model_mean_s=float(np.mean(tm)), model_median_s=float(np.median(tm)),
                pipeline_mean_s=float(np.mean(tp)), pipeline_median_s=float(np.median(tp)), n=len(images))


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1), encoding="utf-8")
