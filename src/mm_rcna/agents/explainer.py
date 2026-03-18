from __future__ import annotations


class ExplainerAgent:
    def explain(self, study_id, predictions, fused_evidence, all_retrievals, all_conflicts, audit_trace):
        lines = [f'Study {study_id} summary:']
        lines.append(f'- fused evidence count: {len(fused_evidence)}')
        for pred in predictions:
            r = all_retrievals[pred.task_name]
            c = all_conflicts[pred.task_name]
            lines.append(
                f'- {pred.task_name}: point={pred.point:.3f}, interval=[{pred.interval_low:.3f}, {pred.interval_high:.3f}], '
                f'abstain={pred.abstain}, retrieval_n={r.effective_n}, conflict={c.conflict_score:.3f}'
            )
        if audit_trace:
            lines.append(f'- audit events: {len(audit_trace)}')
        return '\n'.join(lines)
