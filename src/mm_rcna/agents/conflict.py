from __future__ import annotations

import numpy as np

from mm_rcna.schemas import ConflictReport, RetrievalSummary, VisionOutput


class MediatorAgent:
    def __init__(self, llm_client=None, model: str | None = None):
        self.llm_client = llm_client
        self.model = model

    def run(self, task_name: str, fused_evidence, retrieval: RetrievalSummary, vision_out: VisionOutput | None):
        evidence_score = 0.0 if not fused_evidence else float(np.mean([x.score for x in fused_evidence]))
        retrieval_uncertainty = float(retrieval.std)
        image_noise = 0.0 if vision_out is None else 0.15 * len(vision_out.quality_flags)

        conflict_score = float(
            np.clip(abs(evidence_score - retrieval.median) + retrieval_uncertainty + image_noise, 0.0, 1.0)
        )

        sources = []
        if retrieval.unstable_distribution:
            sources.append("retrieval_distribution_unstable")
        if image_noise > 0:
            sources.append("vision_quality_issue")
        if not sources:
            sources.append("evidence_retrieval_gap")

        if self.llm_client is not None and getattr(self.llm_client, "ready", False) and self.model:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You classify conflict sources conservatively. "
                            "Output strict JSON with key conflict_sources as a list of short snake_case labels."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"task={task_name}\n"
                            f"evidence_score={evidence_score:.4f}\n"
                            f"retrieval_median={retrieval.median:.4f}\n"
                            f"retrieval_std={retrieval.std:.4f}\n"
                            f"image_noise={image_noise:.4f}\n"
                            f"default_sources={sources}"
                        ),
                    },
                ]
                obj = self.llm_client.json_chat(self.model, messages, max_completion_tokens=300)
                llm_sources = obj.get("conflict_sources", [])
                if isinstance(llm_sources, list) and llm_sources:
                    sources = [str(x) for x in llm_sources[:4]]
            except Exception:
                pass

        return ConflictReport(
            task_name=task_name,
            conflict_score=conflict_score,
            conflict_sources=sources,
            conflict_vector=[evidence_score, retrieval.median, retrieval.std, image_noise],
            trigger_conflict_retrieval=conflict_score >= 0.5,
        )