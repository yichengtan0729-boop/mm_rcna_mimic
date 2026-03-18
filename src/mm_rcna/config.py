from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List
import yaml


@dataclass
class PathsConfig:
    artifacts_dir: str = "./artifacts"


@dataclass
class ModelsConfig:
    text_encoder_name: str
    image_encoder_name: str
    vision_backbone_name: str
    image_size: int
    lesion_labels: List[str]
    lung_regions: List[str]
    use_trained_vision_heads: bool = False
    vision_checkpoint: str = "./artifacts/vision_heads.json"


@dataclass
class RetrievalConfig:
    index_dir: str
    top_k: int = 8
    conflict_top_k: int = 6
    min_effective_n: int = 4
    image_weight: float = 0.5
    text_weight: float = 0.5
    exclude_same_subject: bool = True


@dataclass
class ConformalConfig:
    alpha: float = 0.1
    qhat: float = 0.15
    qhat_json: str = "./artifacts/conformal.json"


@dataclass
class ContractsConfig:
    unstable_dist_std_threshold: float = 0.2
    high_conflict_threshold: float = 0.55
    high_conflict_min_width: float = 0.2
    low_coverage_threshold: float = 0.35
    min_interval_low: float = 0.0
    max_interval_high: float = 1.0


@dataclass
class RepairConfig:
    max_rounds: int = 3
    api_budget: int = 0
    token_budget: int = 0
    use_learned_policy: bool = False
    learned_policy_path: str = "./artifacts/repair_policy.pkl"


@dataclass
class TaskConfig:
    name: str
    label_column: str
    evidence_topics: List[str] = field(default_factory=list)


@dataclass
class PredictionConfig:
    tasks: List[TaskConfig]


@dataclass
class DataConfig:
    studies_csv: str
    labels_csv: str
    notes_dir: str
    reports_dir: str
    images_dir: str
    index_dir: str


@dataclass
class AppConfig:
    project_name: str
    paths: PathsConfig
    models: ModelsConfig
    retrieval: RetrievalConfig
    conformal: ConformalConfig
    contracts: ContractsConfig
    repair: RepairConfig
    prediction: PredictionConfig
    data: DataConfig


def _task_from_dict(obj: dict[str, Any]) -> TaskConfig:
    return TaskConfig(**obj)


def load_config(path: str) -> AppConfig:
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    return AppConfig(
        project_name=raw.get('project_name', 'mm_rcna_mimic'),
        paths=PathsConfig(**raw['paths']),
        models=ModelsConfig(**raw['models']),
        retrieval=RetrievalConfig(**raw['retrieval']),
        conformal=ConformalConfig(**raw['conformal']),
        contracts=ContractsConfig(**raw['contracts']),
        repair=RepairConfig(**raw['repair']),
        prediction=PredictionConfig(tasks=[_task_from_dict(x) for x in raw['prediction']['tasks']]),
        data=DataConfig(**raw['data']),
    )
