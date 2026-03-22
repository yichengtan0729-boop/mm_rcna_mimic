from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from mm_rcna.config import load_config
from mm_rcna.schemas import RepairBudget
from mm_rcna.utils.io_utils import save_json, load_json
from mm_rcna.data.mimic_cxr_dataset import MIMICCXRStudyBuilder
from mm_rcna.data.text_governance import TextGovernanceAgent
from mm_rcna.models.vision import VisionToolRunner
from mm_rcna.models.api_clients import OpenAICompatibleClient, NullAPIClient
from mm_rcna.agents.evidence_builder import MultimodalEvidenceBuilder
from mm_rcna.agents.evidence_fusion import EvidenceFusion
from mm_rcna.agents.retrieval import RetrievalAgent, ConflictRetrievalAgent
from mm_rcna.agents.conflict import MediatorAgent
from mm_rcna.agents.task_coordinator import TaskCoordinatorAgent
from mm_rcna.agents.verifier import ContractVerifierAgent
from mm_rcna.agents.repair import OrchestratorRepairLoop
from mm_rcna.agents.explainer import ExplainerAgent
from mm_rcna.calibrate.conformal import SplitConformalCalibrator


def _load_saved_qhat(default_qhat: float, conformal_json: str) -> float:
    payload = load_json(conformal_json, default=None)
    if not payload:
        return float(default_qhat)
    return float(payload.get("qhat", default_qhat))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", default="./artifacts/pipeline_output.json")
    parser.add_argument("--conformal-json", default="./artifacts/conformal.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    qhat = _load_saved_qhat(cfg.conformal.qhat, args.conformal_json)

    llm_client = OpenAICompatibleClient(cfg.models.api) if cfg.models.api.enabled else NullAPIClient()
    llm_model = cfg.models.api.llm_model if cfg.models.api.enabled else None

    study_builder = MIMICCXRStudyBuilder(cfg)
    gov_agent = TextGovernanceAgent(llm_client=llm_client, model=llm_model)
    vision = VisionToolRunner(
        cfg.models.image_encoder_name,
        cfg.models.vision_backbone_name,
        cfg.models.lesion_labels,
        cfg.models.lung_regions,
        cfg.models.image_size,
        device=args.device,
        checkpoint_path=cfg.models.vision_checkpoint if cfg.models.use_trained_vision_heads else None,
    )
    evidence_builder = MultimodalEvidenceBuilder(cfg, llm_client=llm_client, model=llm_model)
    fusion = EvidenceFusion(cfg)
    retrieval_agent = RetrievalAgent(cfg, device=args.device)
    conflict_retrieval = ConflictRetrievalAgent(cfg)
    mediator = MediatorAgent(llm_client=llm_client, model=llm_model)
    coordinator = TaskCoordinatorAgent(
        cfg,
        calibrator=SplitConformalCalibrator(alpha=cfg.conformal.alpha, qhat=qhat),
    )
    verifier = ContractVerifierAgent(cfg, llm_client=llm_client, model=llm_model)
    repairer = OrchestratorRepairLoop(cfg, llm_client=llm_client, model=llm_model)
    explainer = ExplainerAgent(llm_client=llm_client, model=llm_model)

    study = study_builder.build_study(args.study_id)
    gov = gov_agent.run(study.notes, study.report)

    image_paths = [img.path for img in study.images if Path(img.path).exists()]
    vision_out = vision.run(image_paths)

    evidence = evidence_builder.run(study, gov, vision_out)
    fused_evidence, coverage = fusion.run(evidence)

    predictions = []
    all_retrievals = {}
    all_conflicts = {}
    all_verifications = {}
    audit_trace = [e.model_dump() for e in gov.audit_events]
    subject_id = getattr(study, "subject_id", None)

    for task in cfg.prediction.tasks:
        retrieval = retrieval_agent.retrieve(
            task.name,
            task.label_column,
            fused_evidence,
            image_paths,
            query_study_id=study.study_id,
            query_subject_id=subject_id,
        )

        conflict = mediator.run(task.name, fused_evidence, retrieval, vision_out)

        conflict_summary = (
            conflict_retrieval.retrieve(
                task_name=task.name,
                retrieval_summary=retrieval,
                conflict=conflict,
                label_column=task.label_column,
                query_study_id=study.study_id,
                query_subject_id=subject_id,
            )
            if conflict.trigger_conflict_retrieval
            else None
        )

        pred = coordinator.predict_one(task, fused_evidence, coverage, retrieval, conflict, conflict_summary)
        verification = verifier.verify_one(pred, retrieval, conflict, gov, vision_out)

        budget = RepairBudget(
            rounds_left=cfg.repair.max_rounds,
            api_budget_left=cfg.repair.api_budget,
            token_budget_left=cfg.repair.token_budget,
        )

        round_idx = 0
        while (not verification.passed) and budget.rounds_left > 0:
            pred, trace_step, budget, retrieval, conflict, conflict_summary = repairer.apply(
                round_idx=round_idx,
                task=task,
                pred=pred,
                verification=verification,
                retrieval=retrieval,
                conflict=conflict,
                conflict_summary=conflict_summary,
                budget=budget,
                fused_evidence=fused_evidence,
                image_paths=image_paths,
                study_id=study.study_id,
                subject_id=subject_id,
                retrieval_agent=retrieval_agent,
                conflict_retrieval=conflict_retrieval,
                mediator=mediator,
                verifier=verifier,
            )
            audit_trace.append(trace_step.model_dump())
            verification = verifier.verify_one(pred, retrieval, conflict, gov, vision_out)
            round_idx += 1

        predictions.append(pred)
        all_retrievals[task.name] = retrieval
        all_conflicts[task.name] = conflict
        all_verifications[task.name] = verification

    explanation = explainer.explain(
        study.study_id,
        predictions,
        fused_evidence,
        all_retrievals,
        all_conflicts,
        audit_trace,
    )

    out = {
        "study_id": study.study_id,
        "predictions": [p.model_dump() for p in predictions],
        "verifications": {k: v.model_dump() for k, v in all_verifications.items()},
        "retrievals": {k: v.model_dump() for k, v in all_retrievals.items()},
        "conflicts": {k: v.model_dump() for k, v in all_conflicts.items()},
        "coverage": coverage,
        "explanation_text": explanation,
        "audit_trace": audit_trace,
        "conformal_qhat_used": float(qhat),
    }
    save_json(out, args.output_json)
    print(explanation)
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()