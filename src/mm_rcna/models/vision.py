from __future__ import annotations

from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image

from mm_rcna.schemas import VisionOutput


class SharedVisionBackbone:
    def __init__(self, lesion_labels, lung_regions, dim: int = 64) -> None:
        self.lesion_labels = lesion_labels
        self.lung_regions = lung_regions
        self.dim = dim
        self.lesion_bias = {k: 0.0 for k in lesion_labels}
        self.region_bias = {k: 0.0 for k in lung_regions}

    def load_checkpoint(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        obj = json.loads(p.read_text(encoding='utf-8'))
        self.lesion_bias.update(obj.get('lesion_bias', {}))
        self.region_bias.update(obj.get('region_bias', {}))

    def encode_image(self, path: str) -> np.ndarray:
        if not path or not Path(path).exists():
            return np.zeros(self.dim, dtype=np.float32)
        img = Image.open(path).convert('L').resize((32, 32))
        arr = np.asarray(img, dtype=np.float32).flatten() / 255.0
        if len(arr) < self.dim:
            pad = np.zeros(self.dim - len(arr), dtype=np.float32)
            arr = np.concatenate([arr, pad])
        else:
            arr = arr[: self.dim]
        norm = np.linalg.norm(arr)
        return arr / norm if norm > 0 else arr

    def predict_heads(self, feature: np.ndarray):
        mean_val = float(feature.mean()) if feature.size else 0.0
        lesion_scores = {k: float(np.clip(mean_val + self.lesion_bias.get(k, 0.0), 0.0, 1.0)) for k in self.lesion_labels}
        region_scores = {k: float(np.clip(mean_val + self.region_bias.get(k, 0.0), 0.0, 1.0)) for k in self.lung_regions}
        return lesion_scores, region_scores


class VisionToolRunner:
    def __init__(self, image_encoder_name, vision_backbone_name, lesion_labels, lung_regions, image_size=224, device='cpu', checkpoint_path=None):
        self.image_encoder_name = image_encoder_name
        self.vision_backbone_name = vision_backbone_name
        self.image_size = image_size
        self.device = device
        self.backbone = SharedVisionBackbone(lesion_labels, lung_regions)
        if checkpoint_path:
            self.backbone.load_checkpoint(checkpoint_path)

    def run(self, image_paths):
        if not image_paths:
            feature = np.zeros(self.backbone.dim, dtype=np.float32)
            lesion_scores, region_scores = self.backbone.predict_heads(feature)
            return VisionOutput(feature_vector=feature.tolist(), lesion_scores=lesion_scores, region_scores=region_scores, quality_flags=['no_image'])

        feats = [self.backbone.encode_image(p) for p in image_paths if Path(p).exists()]
        if not feats:
            feature = np.zeros(self.backbone.dim, dtype=np.float32)
            lesion_scores, region_scores = self.backbone.predict_heads(feature)
            return VisionOutput(feature_vector=feature.tolist(), lesion_scores=lesion_scores, region_scores=region_scores, quality_flags=['missing_files'])

        feature = np.mean(np.stack(feats, axis=0), axis=0)
        lesion_scores, region_scores = self.backbone.predict_heads(feature)
        quality = []
        if float(feature.std()) < 0.01:
            quality.append('low_variance_image_feature')
        return VisionOutput(feature_vector=feature.tolist(), lesion_scores=lesion_scores, region_scores=region_scores, quality_flags=quality)
