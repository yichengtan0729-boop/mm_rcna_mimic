from __future__ import annotations

import json
from typing import List

import numpy as np

from mm_rcna.config import AppConfig
from mm_rcna.schemas import EvidenceItem, GovernanceOutput, StudyRecord, VisionOutput

TOPIC_KEYWORDS = {
    "respiratory_status": [
        "opacity",
        "effusion",
        "pneumothorax",
        "respiratory",
        "oxygen",
        "ventilation",
        "edema",
        "atelectasis",
        "infiltrate",
        "airspace",
        "consolidation",
        "dyspnea",
        "sob",
        "hypoxia",
    ],
    "criticality": [
        "critical",
        "unstable",
        "shock",
        "icu",
        "decompensation",
        "severe",
        "cardiomegaly",
        "failure",
        "sepsis",
    ],
    "hemodynamics": [
        "hypotension",
        "vasopressor",
        "tachycardia",
        "hemodynamic",
        "pulmonary vascular",
        "vascular congestion",
    ],
    "support_devices": [
        "tube",
        "line",
        "catheter",
        "intubated",
        "ventilator",
        "pacer",
        "device",
        "ett",
        "endotracheal",
        "trach",
        "cpap",
        "bipap",
    ],
}

ALLOWED_TOPICS = [
    "respiratory_status",
    "criticality",
    "hemodynamics",
    "support_devices",
]

DEFAULT_TASKS = [
    "mortality_risk",
    "icu_risk",
    "ventilation_risk",
]


class MultimodalEvidenceBuilder:
    def __init__(self, config: AppConfig, llm_client=None, model: str | None = None) -> None:
        self.cfg = config
        self.llm_client = llm_client
        self.model = model

    def _tasks_for_topic(self, topic: str, finding: str = "") -> List[str]:
        topic = (topic or "").strip().lower()
        finding_low = (finding or "").lower()

        if topic == "support_devices":
            tasks = ["ventilation_risk", "icu_risk"]
            if any(x in finding_low for x in ["ett", "endotracheal", "intubat", "ventilator", "trach", "bipap", "cpap"]):
                tasks.append("mortality_risk")
            return tasks

        if topic == "criticality":
            return ["mortality_risk", "icu_risk"]

        if topic == "hemodynamics":
            return ["mortality_risk", "icu_risk"]

        if topic == "respiratory_status":
            if any(x in finding_low for x in ["edema", "effusion", "opacity", "consolidation", "infiltrate", "airspace", "hypoxia", "ventilation"]):
                return ["mortality_risk", "icu_risk", "ventilation_risk"]
            return ["icu_risk", "ventilation_risk"]

        return DEFAULT_TASKS

    def _text_evidence_rule(self, text: str) -> List[EvidenceItem]:
        text_low = (text or "").lower()
        items: List[EvidenceItem] = []
        for topic, kws in TOPIC_KEYWORDS.items():
            hits = [kw for kw in kws if kw in text_low]
            if not hits:
                continue
            finding = ", ".join(hits[:4])
            items.append(
                EvidenceItem(
                    topic=topic,
                    finding=finding,
                    score=min(1.0, 0.16 * len(hits) + 0.24),
                    modality="text",
                    supports_tasks=self._tasks_for_topic(topic, finding),
                    source="rule_text",
                )
            )
        return items

    def _vision_topic_from_lesion(self, lesion: str) -> str:
        lesion_low = (lesion or "").lower()
        if any(x in lesion_low for x in ["tube", "line", "catheter", "device", "pacer", "ett", "trach"]):
            return "support_devices"
        if any(x in lesion_low for x in ["shock", "cardiomegaly", "edema"]):
            return "criticality"
        return "respiratory_status"

    def _vision_evidence_rule(self, vision_out: VisionOutput) -> List[EvidenceItem]:
        items: List[EvidenceItem] = []

        if not vision_out or not vision_out.lesion_scores:
            return items

        best_region = None
        best_region_score = -1.0
        for r, s in (vision_out.region_scores or {}).items():
            s = float(s)
            if s > best_region_score:
                best_region = str(r)
                best_region_score = s

        flags = getattr(vision_out, "quality_flags", []) or []
        quality_penalty = 0.03 * len([x for x in flags if x in {"low_contrast", "underexposed", "overexposed", "low_detail"}])
        multi_view_bonus = 0.03 if ("frontal_lateral_pair" in flags or "multi_view_fused" in flags) else 0.0

        for lesion, score in vision_out.lesion_scores.items():
            score = float(score) - quality_penalty + multi_view_bonus
            if score < 0.42:
                continue

            finding = str(lesion)
            if best_region is not None and best_region_score >= 0.28 and lesion not in {"cardiomegaly", "device"}:
                finding = f"{finding}@{best_region}"

            topic = self._vision_topic_from_lesion(lesion)
            items.append(
                EvidenceItem(
                    topic=topic,
                    finding=finding,
                    score=float(np.clip(score, 0.0, 1.0)),
                    modality="vision",
                    supports_tasks=self._tasks_for_topic(topic, finding),
                    source="vision_tool",
                )
            )

        return items

    def _build_llm_payload(
        self,
        study: StudyRecord,
        gov: GovernanceOutput,
        vision_out: VisionOutput,
    ) -> dict:
        return {
            "study_id": study.study_id,
            "subject_id": getattr(study, "subject_id", None),
            "cleaned_notes": gov.cleaned_notes or "",
            "cleaned_report": gov.cleaned_report or "",
            "vision_summary": {
                "lesion_scores": vision_out.lesion_scores,
                "region_scores": vision_out.region_scores,
                "quality_flags": vision_out.quality_flags,
            },
            "allowed_topics": ALLOWED_TOPICS,
            "allowed_tasks": DEFAULT_TASKS,
            "requirements": {
                "max_items": 8,
                "score_range": [0.0, 1.0],
                "be_conservative": True,
                "no_final_risk_prediction": True,
            },
        }

    def _llm_multimodal_evidence(
        self,
        study: StudyRecord,
        gov: GovernanceOutput,
        vision_out: VisionOutput,
    ) -> List[EvidenceItem]:
        if self.llm_client is None or not getattr(self.llm_client, "ready", False) or not self.model:
            return []

        payload = self._build_llm_payload(study, gov, vision_out)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical multimodal evidence extraction module.\n"
                    "Your job is to extract evidence items from cleaned text and vision summary.\n"
                    "Do NOT output final risk predictions.\n"
                    "Output strict JSON with top-level key `evidence_items`.\n"
                    "Each evidence item must contain:\n"
                    "- topic: one of respiratory_status, criticality, hemodynamics, support_devices\n"
                    "- finding: short text\n"
                    "- score: float in [0,1]\n"
                    "- modality: one of text, vision, text+vision\n"
                    "- supports_tasks: subset of mortality_risk, icu_risk, ventilation_risk\n"
                    "- source: short snake_case string\n"
                    "Be conservative. Prefer fewer, higher-quality items."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            obj = self.llm_client.json_chat(
                self.model,
                messages,
                max_completion_tokens=self.cfg.models.api.max_completion_tokens,
            )
        except Exception:
            return []

        out: List[EvidenceItem] = []
        raw_items = obj.get("evidence_items", [])
        if not isinstance(raw_items, list):
            return []

        for x in raw_items[:8]:
            try:
                topic = str(x.get("topic", "respiratory_status")).strip()
                if topic not in ALLOWED_TOPICS:
                    continue
                supports_tasks = x.get("supports_tasks", DEFAULT_TASKS)
                if not isinstance(supports_tasks, list):
                    supports_tasks = DEFAULT_TASKS
                supports_tasks = [str(t) for t in supports_tasks if str(t) in DEFAULT_TASKS]
                if not supports_tasks:
                    supports_tasks = DEFAULT_TASKS
                score = float(x.get("score", 0.5))
                score = max(0.0, min(1.0, score))
                modality = str(x.get("modality", "text")).strip().lower()
                if modality not in {"text", "vision", "text+vision"}:
                    modality = "text"
                out.append(
                    EvidenceItem(
                        topic=topic,
                        finding=str(x.get("finding", ""))[:240],
                        score=score,
                        modality=modality,
                        supports_tasks=supports_tasks,
                        source=str(x.get("source", "llm_multimodal")),
                    )
                )
            except Exception:
                continue
        return out

    @staticmethod
    def _dedup(items: List[EvidenceItem]) -> List[EvidenceItem]:
        seen = set()
        out = []
        for item in items:
            key = (
                item.topic.strip().lower(),
                item.finding.strip().lower(),
                item.modality.strip().lower(),
                tuple(sorted(item.supports_tasks)),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def run(self, study: StudyRecord, gov: GovernanceOutput, vision_out: VisionOutput) -> List[EvidenceItem]:
        merged_text = (gov.cleaned_notes or "") + "\n\n" + (gov.cleaned_report or "")

        llm_items = self._llm_multimodal_evidence(study, gov, vision_out)
        if llm_items:
            rule_vision = self._vision_evidence_rule(vision_out)
            return self._dedup(llm_items + rule_vision[:2])

        text_items = self._text_evidence_rule(merged_text)
        vision_items = self._vision_evidence_rule(vision_out)
        return self._dedup(text_items + vision_items)