from __future__ import annotations

from typing import Optional

from mm_rcna.policy.repair_policy import RuleRepairPolicy, LearnedRepairPolicy
from mm_rcna.schemas import RepairTraceStep


class OrchestratorRepairLoop:
    def __init__(self, config) -> None:
        self.cfg = config
        self.rule_policy = RuleRepairPolicy()
        self.learned_policy = LearnedRepairPolicy(config.repair.learned_policy_path) if config.repair.use_learned_policy else None

    def apply(self, round_idx, task, pred, verification, retrieval, conflict, conflict_summary, budget, fused_evidence, image_paths, study_id, subject_id, retrieval_agent, conflict_retrieval, mediator, verifier):
        if self.learned_policy and self.learned_policy.ready:
            action, reason = self.learned_policy.choose_action([
                conflict.conflict_score,
                retrieval.std,
                retrieval.effective_n,
                float(pred.interval_high - pred.interval_low),
            ])
        else:
            action, reason = self.rule_policy.choose_action(verification.violations)

        updated = []
        if action == 'widen_interval':
            mid = pred.point
            half = max((pred.interval_high - pred.interval_low) / 2.0, self.cfg.contracts.high_conflict_min_width / 2.0)
            pred.interval_low = max(0.0, mid - 1.5 * half)
            pred.interval_high = min(1.0, mid + 1.5 * half)
            updated = ['task_prediction']
        elif action == 'expand_standard_retrieval':
            old_k = int(self.cfg.retrieval.top_k)
            new_k = min(max(old_k + 5, old_k * 2), old_k + 20)
            retrieval = retrieval_agent.retrieve(task.name, task.label_column, fused_evidence, image_paths, query_study_id=study_id, query_subject_id=subject_id, override_top_k=new_k)
            pred.point = float((pred.point + retrieval.median) / 2.0)
            pred.interval_low = max(0.0, min(pred.interval_low, retrieval.q10))
            pred.interval_high = min(1.0, max(pred.interval_high, retrieval.q90))
            conflict = mediator.run(task.name, fused_evidence, retrieval, None)
            conflict_summary = conflict_retrieval.retrieve(task.name, retrieval, conflict, label_column=task.label_column, query_study_id=study_id, query_subject_id=subject_id) if conflict.trigger_conflict_retrieval else None
            updated = ['retrieval', 'task_prediction', 'conflict']
        elif action == 'trigger_conflict_retrieval':
            if conflict_summary is None:
                conflict_summary = conflict_retrieval.retrieve(task.name, retrieval, conflict, label_column=task.label_column, query_study_id=study_id, query_subject_id=subject_id)
            pred.point = float((pred.point + conflict_summary.median) / 2.0)
            pred.interval_low = max(0.0, min(pred.interval_low, conflict_summary.q10))
            pred.interval_high = min(1.0, max(pred.interval_high, conflict_summary.q90))
            updated = ['task_prediction', 'conflict_retrieval']
        elif action == 'rebuild_evidence':
            retrieval = retrieval_agent.retrieve(task.name, task.label_column, fused_evidence, image_paths, query_study_id=study_id, query_subject_id=subject_id)
            pred.point = float((pred.point + retrieval.median) / 2.0)
            pred.interval_low = max(0.0, min(pred.interval_low, retrieval.q10))
            pred.interval_high = min(1.0, max(pred.interval_high, retrieval.q90))
            conflict = mediator.run(task.name, fused_evidence, retrieval, None)
            conflict_summary = conflict_retrieval.retrieve(task.name, retrieval, conflict, label_column=task.label_column, query_study_id=study_id, query_subject_id=subject_id) if conflict.trigger_conflict_retrieval else None
            updated = ['retrieval', 'conflict', 'task_prediction']
        else:
            pred.abstain = True
            updated = ['task_prediction']

        budget.rounds_left -= 1
        trace = RepairTraceStep(round_idx=round_idx, action=action, reason=reason, updated_modules=updated)
        return pred, trace, budget, retrieval, conflict, conflict_summary
