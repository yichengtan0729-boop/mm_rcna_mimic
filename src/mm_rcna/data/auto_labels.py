from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


KEYWORDS = {
    'mortality_risk': ['expired', 'death', 'critical', 'code blue'],
    'icu_risk': ['icu', 'intensive care', 'vasopressor', 'unstable'],
    'ventilation_risk': ['intubated', 'ventilator', 'mechanical ventilation', 'respiratory failure'],
}


@dataclass
class AutoLabelBuilder:
    text_weight: float = 0.7
    table_weight: float = 0.3

    def _score_text(self, text: str, task: str) -> float:
        low = (text or '').lower()
        hits = sum(1 for kw in KEYWORDS[task] if kw in low)
        return min(1.0, hits / max(1, len(KEYWORDS[task]) // 2))

    def build(self, df: pd.DataFrame, note_col: str = 'notes', report_col: str = 'report') -> pd.DataFrame:
        out = df.copy()
        merged = out.get(note_col, '').fillna('') + ' ' + out.get(report_col, '').fillna('')
        for task in KEYWORDS:
            out[task] = [self._score_text(txt, task) for txt in merged]
        return out
