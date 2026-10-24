"""Optional pretrained detector. The first call downloads TorchVision weights."""

from __future__ import annotations

import cv2
import numpy as np
import torch


_LOADED: dict[str, tuple[object, object, list[str]]] = {}


def _load_detector(device: torch.device) -> tuple[object, object, list[str]]:
    """Load the pretrained detector once per device so a threshold change can rerun quickly."""
    key = f"{device.type}:{device.index if device.index is not None else 0}"
    loaded = _LOADED.get(key)
    if loaded is not None:
        return loaded
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    detector = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
    loaded = (detector, weights.transforms(), list(weights.meta["categories"]))
    _LOADED[key] = loaded
    return loaded


def detect_objects(
    image_bgr: np.ndarray,
    threshold: float = 0.5,
    device: torch.device | None = None,
) -> dict[str, object]:
    from PIL import Image

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector, preprocess, labels = _load_detector(device)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor_image = preprocess(Image.fromarray(rgb)).to(device)

    with torch.inference_mode():
        prediction = detector([tensor_image])[0]

    scores = prediction["scores"].detach().cpu().numpy()
    boxes = prediction["boxes"].detach().cpu().numpy()
    classes = prediction["labels"].detach().cpu().numpy()

    visual = image_bgr.copy()
    for box, score, class_id in zip(boxes, scores, classes):
        if float(score) < threshold:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(visual, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{labels[int(class_id)]}: {float(score):.2f}"
        cv2.putText(
            visual,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return {
        "count": int((scores >= threshold).sum()),
        "visual_bgr": visual,
        "scores": scores,
    }
