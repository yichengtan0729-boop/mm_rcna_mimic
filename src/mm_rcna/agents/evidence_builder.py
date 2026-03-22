from __future__ import annotations

from typing import List

from mm_rcna.config import AppConfig
from mm_rcna.schemas import EvidenceItem, StudyRecord, GovernanceOutput, VisionOutput


TOPIC_KEYWORDS = {
    "respiratory_status": ["opacity", "effusion", "pneumothorax", "respiratory", "oxygen", "ventilation"],
    "criticality": ["critical", "unstable", "shock", "icu"],
    "hemodynamics": ["hypotension", "vasopressor", "tachycardia"],
    "support_devices": ["tube", "line", "catheter", "intubated", "ventilator"],
}


class MultimodalEvidenceBuilder:
    def __init__(self, config: AppConfig, llm_client=None, model: str | None = None) -> None:
        self.cfg = config
        self.llm_client = llm_client
        self.model = model

    def _text_evidence_rule(self, text: str) -> List[EvidenceItem]:
        text_low = (text or "").lower()
        items = []
        for topic, kws in TOPIC_KEYWORDS.items():
            hits = [kw for kw in kws if kw in text_low]
            if hits:
                items.append(
                    EvidenceItem(
                        topic=topic,
                        finding=", ".join(hits[:3]),
                        score=min(1.0, 0.2 * len(hits) + 0.2),
                        modality="text",
                        supports_tasks=["mortality_risk", "icu_risk", "ventilation_risk"],
                        source="notes/report",
                    )
                )
        return items

    def _text_evidence_llm(self, study: StudyRecord, gov: GovernanceOutput) -> List[EvidenceItem]:
        if self.llm_client is None or not getattr(self.llm_client, "ready", False) or not self.model:
            return []

        merged_text = (gov.cleaned_notes or "") + "\n\n" + (gov.cleaned_report or "")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical evidence extraction module. "
                    "Extract concise evidence items for topics among: respiratory_status, criticality, "
                    "hemodynamics, support_devices. Output strict JSON with key evidence_items, "
                    "where each item has: topic, finding, score, modality, supports_tasks, source."
                ),
            },
            {
                "role": "user",
                "content": f"STUDY_ID: {study.study_id}\nTEXT:\n{merged_text}",
            },
        ]
        try:
            obj = self.llm_client.json_chat(self.model, messages, max_completion_tokens=900)
            out = []
            for x in obj.get("evidence_items", []):
                out.append(
                    EvidenceItem(
                        topic=str(x.get("topic", "respiratory_status")),
                        finding=str(x.get("finding", ""))[:200],
                        score=float(x.get("score", 0.5)),
                        modality=str(x.get("modality", "text")),
                        supports_tasks=list(x.get("supports_tasks", ["mortality_risk", "icu_risk", "ventilation_risk"])),
                        source=str(x.get("source", "llm_text")),
                    )
                )
            return out
        except Exception:
            return []

    def _vision_evidence(self, vision_out: VisionOutput) -> List[EvidenceItem]:
        items = []
        for lesion, score in vision_out.lesion_scores.items():
            if score >= 0.35:
                items.append(
                    EvidenceItem(
                        topic="respiratory_status",
                        finding=lesion,
                        score=float(score),
                        modality="vision",
                        supports_tasks=["mortality_risk", "icu_risk", "ventilation_risk"],
                        source="vision_tool",
                    )
                )
        return items

    def run(self, study: StudyRecord, gov: GovernanceOutput, vision_out: VisionOutput) -> List[EvidenceItem]:
        merged_text = (gov.cleaned_notes or "") + "\n\n" + (gov.cleaned_report or "")
        text_items = self._text_evidence_llm(study, gov)
        if not text_items:
            text_items = self._text_evidence_rule(merged_text)
        vision_items = self._vision_evidence(vision_out)
        return text_items + vision_items