from __future__ import annotations

import re
from mm_rcna.schemas import AuditEvent, GovernanceOutput


class TextGovernanceAgent:
    def run(self, notes: str, report: str) -> GovernanceOutput:
        audit = []
        cleaned_notes = notes or ''
        cleaned_report = report or ''
        for token in ['name:', 'mrn:', 'dob:', 'phone:']:
            if token in cleaned_notes.lower() or token in cleaned_report.lower():
                audit.append(AuditEvent(name='pii_redaction_notice', detail=f'Possible token detected: {token}', severity='warning'))
        cleaned_notes = re.sub(r'\d{7,}', '[REDACTED_ID]', cleaned_notes)
        cleaned_report = re.sub(r'\d{7,}', '[REDACTED_ID]', cleaned_report)
        if not cleaned_report.strip():
            audit.append(AuditEvent(name='missing_report', detail='No report text found', severity='warning'))
        return GovernanceOutput(cleaned_notes=cleaned_notes, cleaned_report=cleaned_report, audit_events=audit)
