from __future__ import annotations

import hashlib
import numpy as np


class TextEncoder:
    def __init__(self, name: str, device: str = 'cpu', dim: int = 64) -> None:
        self.name = name
        self.device = device
        self.dim = dim

    def _encode_one(self, text: str) -> np.ndarray:
        text = text or ''
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vec
        for tok in tokens:
            h = int(hashlib.md5(tok.encode('utf-8')).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode(self, texts):
        return np.stack([self._encode_one(t) for t in texts], axis=0)
