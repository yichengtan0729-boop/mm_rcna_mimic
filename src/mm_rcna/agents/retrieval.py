from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from mm_rcna.config import AppConfig
from mm_rcna.schemas import ConflictReport, EvidenceItem, RetrievalNeighbor, RetrievalSummary
from mm_rcna.models.text_encoder import TextEncoder
from mm_rcna.models.vision import VisionToolRunner
from mm_rcna.utils.io_utils import load_pickle


class RetrievalAgent:
    def __init__(self, config: AppConfig, device: str = 'cpu') -> None:
        self.cfg = config
        idx_dir = Path(config.data.index_dir)
        self.meta = load_pickle(idx_dir / 'meta.pkl', default=[])
        self.text_vecs = np.load(idx_dir / 'text.npy') if (idx_dir / 'text.npy').exists() else np.zeros((0, 64), dtype=np.float32)
        self.image_vecs = np.load(idx_dir / 'image.npy') if (idx_dir / 'image.npy').exists() else np.zeros((0, 64), dtype=np.float32)
        self.text_encoder = TextEncoder(config.models.text_encoder_name, device=device)
        self.vision = VisionToolRunner(
            config.models.image_encoder_name,
            config.models.vision_backbone_name,
            config.models.lesion_labels,
            config.models.lung_regions,
            image_size=config.models.image_size,
            device=device,
            checkpoint_path=config.models.vision_checkpoint if config.models.use_trained_vision_heads else None,
        )

    def _query_text(self, items: List[EvidenceItem], task_name: str) -> str:
        rel = [x for x in items if task_name in x.supports_tasks]
        return ' ; '.join([f'{x.topic}: {x.finding}' for x in rel]) or 'empty evidence'

    def retrieve(self, task_name: str, label_column: str, evidence: List[EvidenceItem], image_paths: List[str], query_study_id: Optional[str] = None, query_subject_id: Optional[str] = None, override_top_k: Optional[int] = None) -> RetrievalSummary:
        text_query = self._query_text(evidence, task_name)
        txt = self.text_encoder.encode([text_query])[0]
        vision_out = self.vision.run(image_paths)
        img = np.asarray(vision_out.feature_vector, dtype=np.float32)
        k = int(override_top_k or self.cfg.retrieval.top_k)

        score_map: Dict[int, float] = {}
        if len(self.text_vecs):
            txt_scores = self.text_vecs @ txt
            for idx, score in enumerate(txt_scores.tolist()):
                score_map[idx] = score_map.get(idx, 0.0) + self.cfg.retrieval.text_weight * float(score)
        if len(self.image_vecs):
            img_scores = self.image_vecs @ img
            for idx, score in enumerate(img_scores.tolist()):
                score_map[idx] = score_map.get(idx, 0.0) + self.cfg.retrieval.image_weight * float(score)

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        neighbors: List[RetrievalNeighbor] = []
        label_values: List[float] = []
        for idx, score in ranked:
            m = self.meta[idx]
            if query_study_id is not None and str(m.get('study_id')) == str(query_study_id):
                continue
            if self.cfg.retrieval.exclude_same_subject and query_subject_id is not None and str(m.get('subject_id')) == str(query_subject_id):
                continue
            lv = float(m.get('labels', {}).get(label_column, m.get('labels', {}).get(task_name, 0.0)))
            label_values.append(lv)
            neighbors.append(RetrievalNeighbor(study_id=str(m.get('study_id')), score=float(score), match_reasons=[text_query[:120]], label_value=lv))
            if len(neighbors) >= k:
                break

        arr = np.asarray(label_values, dtype=np.float32) if label_values else np.asarray([0.0], dtype=np.float32)
        return RetrievalSummary(
            task_name=task_name,
            neighbors=neighbors,
            median=float(np.median(arr)),
            q10=float(np.quantile(arr, 0.1)),
            q90=float(np.quantile(arr, 0.9)),
            std=float(arr.std()),
            effective_n=int(len(label_values)),
            insufficient_support=len(label_values) < self.cfg.retrieval.min_effective_n,
            unstable_distribution=float(arr.std()) > self.cfg.contracts.unstable_dist_std_threshold,
            query_text=text_query,
        )


class ConflictRetrievalAgent:
    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        idx_dir = Path(config.data.index_dir)
        self.meta = load_pickle(idx_dir / 'meta.pkl', default=[])

    def retrieve(self, task_name: str, retrieval_summary: RetrievalSummary, conflict: ConflictReport, label_column: Optional[str] = None, query_study_id: Optional[str] = None, query_subject_id: Optional[str] = None) -> RetrievalSummary:
        vals = []
        neighbors = []
        for m in self.meta:
            if query_study_id is not None and str(m.get('study_id')) == str(query_study_id):
                continue
            if self.cfg.retrieval.exclude_same_subject and query_subject_id is not None and str(m.get('subject_id')) == str(query_subject_id):
                continue
            label_value = float(m.get('labels', {}).get(label_column or task_name, retrieval_summary.median))
            mismatch = abs(label_value - retrieval_summary.median)
            score = max(0.0, 1.0 - abs(mismatch - conflict.conflict_score))
            vals.append(label_value)
            neighbors.append(RetrievalNeighbor(study_id=str(m.get('study_id')), score=float(score), match_reasons=conflict.conflict_sources, label_value=label_value))
        neighbors = sorted(neighbors, key=lambda x: x.score, reverse=True)[: self.cfg.retrieval.conflict_top_k]
        vals = [x.label_value for x in neighbors]
        arr = np.asarray(vals, dtype=np.float32) if vals else np.asarray([retrieval_summary.median], dtype=np.float32)
        return RetrievalSummary(
            task_name=task_name,
            neighbors=neighbors,
            median=float(np.median(arr)),
            q10=float(np.quantile(arr, 0.1)),
            q90=float(np.quantile(arr, 0.9)),
            std=float(arr.std()),
            effective_n=len(vals),
            insufficient_support=len(vals) < max(3, self.cfg.retrieval.min_effective_n // 2),
            unstable_distribution=float(arr.std()) > self.cfg.contracts.unstable_dist_std_threshold,
            query_text='conflict-pattern',
        )
