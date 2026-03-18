from __future__ import annotations

from typing import List

from mm_rcna.config import AppConfig
from mm_rcna.schemas import EvidenceItem, StudyRecord, GovernanceOutput, VisionOutput


TOPIC_KEYWORDS = {
    'respiratory_status': ['opacity', 'effusion', 'pneumothorax', 'respiratory', 'oxygen', 'ventilation'],
    'criticality': ['critical', 'unstable', 'shock', 'icu'],
    'hemodynamics': ['hypotension', 'vasopressor', 'tachycardia'],
    'support_devices': ['tube', 'line', 'catheter', 'intubated', 'ventilator'],
}


class MultimodalEvidenceBuilder:
    def __init__(self, config: AppConfig) -> None:
        self.cfg = config

    def _text_evidence(self, text: str) -> List[EvidenceItem]:
        text_low = (text or '').lower()
        items = []
        for topic, kws in TOPIC_KEYWORDS.items():
            hits = [kw for kw in kws if kw in text_low]
            if hits:
                items.append(EvidenceItem(
                    topic=topic,
                    finding=', '.join(hits[:3]),
                    score=min(1.0, 0.2 * len(hits) + 0.2),
                    modality='text',
                    supports_tasks=['mortality_risk', 'icu_risk', 'ventilation_risk'],
                    source='notes/report',
                ))
        return items

    def _vision_evidence(self, vision_out: VisionOutput) -> List[EvidenceItem]:
        items = []
        for lesion, score in vision_out.lesion_scores.items():
            if score >= 0.35:
                topic = 'respiratory_status' if lesion in {'opacity', 'effusion', 'pneumothorax', 'edema', 'atelectasis'} else 'criticality'
                items.append(EvidenceItem(
                    topic=topic,
                    finding=lesion,
                    score=float(score),
                    modality='image',
                    supports_tasks=['mortality_risk', 'icu_risk', 'ventilation_risk'],
                    source='vision-head',
                ))
        return items

    def run(self, study: StudyRecord, gov: GovernanceOutput, vision_out: VisionOutput) -> List[EvidenceItem]:
        items = []
        items.extend(self._text_evidence(gov.cleaned_notes + '\n' + gov.cleaned_report))
        items.extend(self._vision_evidence(vision_out))
        return items
