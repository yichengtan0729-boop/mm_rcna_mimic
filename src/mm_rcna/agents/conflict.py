from __future__ import annotations

import numpy as np

from mm_rcna.schemas import ConflictReport, RetrievalSummary, VisionOutput, EvidenceItem


class MediatorAgent:
    def run(self, task_name: str, fused_evidence, retrieval: RetrievalSummary, vision_out: VisionOutput | None):
        evidence_score = 0.0 if not fused_evidence else float(np.mean([x.score for x in fused_evidence]))
        retrieval_uncertainty = float(retrieval.std)
        image_noise = 0.0 if vision_out is None else 0.15 * len(vision_out.quality_flags)
        conflict_score = float(np.clip(abs(evidence_score - retrieval.median) + retrieval_uncertainty + image_noise, 0.0, 1.0))
        sources = []
        if retrieval.unstable_distribution:
            sources.append('retrieval_distribution_unstable')
        if image_noise > 0:
            sources.append('vision_quality_issue')
        if not sources:
            sources.append('evidence_retrieval_gap')
        return ConflictReport(
            task_name=task_name,
            conflict_score=conflict_score,
            conflict_sources=sources,
            conflict_vector=[evidence_score, retrieval.median, retrieval.std, image_noise],
            trigger_conflict_retrieval=conflict_score >= 0.5,
        )
