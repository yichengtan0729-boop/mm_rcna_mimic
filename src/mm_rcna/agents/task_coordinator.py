from __future__ import annotations

import numpy as np

from mm_rcna.schemas import TaskPrediction


class TaskCoordinatorAgent:
    def __init__(self, config, calibrator) -> None:
        self.cfg = config
        self.calibrator = calibrator

    def predict_one(self, task, fused_evidence, coverage, retrieval, conflict, conflict_summary=None):
        task_items = [x for x in fused_evidence if task.name in x.supports_tasks and (not task.evidence_topics or x.topic in task.evidence_topics)]
        evidence_score = float(np.mean([x.score for x in task_items])) if task_items else 0.0
        point = 0.45 * evidence_score + 0.45 * retrieval.median + 0.10 * float(coverage)
        if conflict_summary is not None:
            point = 0.7 * point + 0.3 * conflict_summary.median
        point = float(np.clip(point, 0.0, 1.0))
        lo, hi = self.calibrator.predict_interval(point)
        return TaskPrediction(
            task_name=task.name,
            point=point,
            interval_low=lo,
            interval_high=hi,
            abstain=False,
            rationale=f'evidence={evidence_score:.3f}, retrieval={retrieval.median:.3f}, coverage={coverage:.3f}, conflict={conflict.conflict_score:.3f}',
        )
