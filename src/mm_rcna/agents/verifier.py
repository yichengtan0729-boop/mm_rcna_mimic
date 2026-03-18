from __future__ import annotations

from mm_rcna.schemas import VerificationResult


class ContractVerifierAgent:
    def __init__(self, config) -> None:
        self.cfg = config

    def verify_one(self, pred, retrieval, conflict, gov, vision_out):
        violations = []
        width = float(pred.interval_high - pred.interval_low)
        if retrieval.insufficient_support:
            violations.append('insufficient_support')
        if retrieval.unstable_distribution:
            violations.append('unstable_distribution')
        if conflict.conflict_score >= self.cfg.contracts.high_conflict_threshold:
            violations.append('high_conflict')
        if width < self.cfg.contracts.high_conflict_min_width and conflict.conflict_score >= self.cfg.contracts.high_conflict_threshold:
            violations.append('interval_too_narrow')
        if any(e.name == 'missing_report' for e in gov.audit_events):
            violations.append('low_coverage')
        passed = len(violations) == 0 and not pred.abstain
        confidence = max(0.0, 1.0 - conflict.conflict_score - retrieval.std)
        return VerificationResult(task_name=pred.task_name, passed=passed, violations=violations, confidence=float(confidence))
