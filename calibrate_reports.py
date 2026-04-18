from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TASKS = ["mortality_risk", "icu_risk", "ventilation_risk"]


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def get_pred_item(data: dict, task: str) -> dict:
    preds = data.get("predictions", [])
    if isinstance(preds, list):
        for x in preds:
            if isinstance(x, dict) and x.get("task_name") == task:
                return x
    return {}


def parse_rationale(rationale: str) -> dict:
    out = {}
    rationale = str(rationale or "")
    for chunk in rationale.split(","):
        if "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        out[k.strip()] = safe_float(v.strip(), 0.0)
    return out


def extract_task_features(data: dict, task: str) -> dict:
    pred = get_pred_item(data, task)

    point = safe_float(pred.get("point"), 0.0)
    interval_low = safe_float(pred.get("interval_low"), point)
    interval_high = safe_float(pred.get("interval_high"), point)
    interval_width = max(0.0, interval_high - interval_low)
    abstain = 1.0 if bool(pred.get("abstain", False)) else 0.0

    retrieval = data.get("retrievals", {}).get(task, {})
    if not isinstance(retrieval, dict):
        retrieval = {}

    neighbors = retrieval.get("neighbors", [])
    if not isinstance(neighbors, list):
        neighbors = []

    labels = []
    scores = []
    for n in neighbors:
        if not isinstance(n, dict):
            continue
        labels.append(safe_float(n.get("label_value"), 0.0))
        scores.append(safe_float(n.get("score"), 0.0))

    if labels:
        labels_arr = np.asarray(labels, dtype=float)
        scores_arr = np.asarray(scores, dtype=float)

        top1_label = float(labels_arr[0])
        top1_score = float(scores_arr[0])

        top2_label = float(labels_arr[:2].mean())
        top3_label = float(labels_arr[:3].mean())
        top5_label = float(labels_arr[:5].mean())
        top10_label = float(labels_arr[:10].mean())
        topk_label = float(labels_arr.mean())

        top3_pos_count = float(labels_arr[:3].sum())
        top5_pos_count = float(labels_arr[:5].sum())
        top10_pos_count = float(labels_arr[:10].sum())

        label_std = float(labels_arr.std())
        label_min = float(labels_arr.min())
        label_max = float(labels_arr.max())

        score_mean = float(scores_arr.mean())
        score_std = float(scores_arr.std())
        score_max = float(scores_arr.max())
        score_min = float(scores_arr.min())
        score_range = float(score_max - score_min)

        if len(scores_arr) >= 2:
            score_gap_1_2 = float(scores_arr[0] - scores_arr[1])
        else:
            score_gap_1_2 = 0.0

        if len(scores_arr) >= 5:
            score_gap_1_5 = float(scores_arr[0] - scores_arr[4])
        else:
            score_gap_1_5 = 0.0

        score_gap_1_mean = float(scores_arr[0] - score_mean)

        def weighted_label(tau: float) -> float:
            s = scores_arr - scores_arr.max()
            w = np.exp(s / max(tau, 1e-6))
            z = w.sum()
            if z <= 0:
                return float(labels_arr.mean())
            w = w / z
            return float((w * labels_arr).sum())

        weighted_label_001 = weighted_label(0.01)
        weighted_label_003 = weighted_label(0.03)
        weighted_label_005 = weighted_label(0.05)
        weighted_label_010 = weighted_label(0.10)
    else:
        top1_label = top1_score = 0.0
        top2_label = top3_label = top5_label = top10_label = topk_label = 0.0
        top3_pos_count = top5_pos_count = top10_pos_count = 0.0
        label_std = label_min = label_max = 0.0
        score_mean = score_std = score_max = score_min = score_range = 0.0
        score_gap_1_2 = score_gap_1_5 = score_gap_1_mean = 0.0
        weighted_label_001 = weighted_label_003 = weighted_label_005 = weighted_label_010 = 0.0

    conflict = data.get("conflicts", {}).get(task, {})
    if not isinstance(conflict, dict):
        conflict = {}
    conflict_score = safe_float(conflict.get("conflict_score"), 0.0)

    ver = data.get("verifications", {}).get(task, {})
    if not isinstance(ver, dict):
        ver = {}
    ver_conf = safe_float(ver.get("confidence"), point)
    ver_passed = 1.0 if bool(ver.get("passed", True)) else 0.0

    rat = parse_rationale(pred.get("rationale", ""))

    return {
        "point": point,
        "interval_low": interval_low,
        "interval_high": interval_high,
        "interval_width": interval_width,
        "abstain": abstain,

        "top1_label": top1_label,
        "top1_score": top1_score,
        "top2_label_mean": top2_label,
        "top3_label_mean": top3_label,
        "top5_label_mean": top5_label,
        "top10_label_mean": top10_label,
        "topk_label_mean": topk_label,

        "top3_pos_count": top3_pos_count,
        "top5_pos_count": top5_pos_count,
        "top10_pos_count": top10_pos_count,

        "weighted_label_tau_001": weighted_label_001,
        "weighted_label_tau_003": weighted_label_003,
        "weighted_label_tau_005": weighted_label_005,
        "weighted_label_tau_010": weighted_label_010,

        "label_std": label_std,
        "label_min": label_min,
        "label_max": label_max,

        "score_mean": score_mean,
        "score_std": score_std,
        "score_max": score_max,
        "score_min": score_min,
        "score_range": score_range,
        "score_gap_1_2": score_gap_1_2,
        "score_gap_1_5": score_gap_1_5,
        "score_gap_1_mean": score_gap_1_mean,

        "conflict_score": conflict_score,
        "verification_confidence": ver_conf,
        "verification_passed": ver_passed,

        "rat_evidence": rat.get("evidence", 0.0),
        "rat_evidence_top": rat.get("evidence_top", 0.0),
        "rat_vision": rat.get("vision", 0.0),
        "rat_retrieval": rat.get("retrieval", 0.0),
        "rat_top1_label": rat.get("top1_label", 0.0),
        "rat_top1_score": rat.get("top1_score", 0.0),
        "rat_coverage": rat.get("coverage", 0.0),
        "rat_conflict": rat.get("conflict", conflict_score),
        "rat_support_n": rat.get("support_n", len(labels)),
    }


def extract_all_task_points(data: dict) -> dict:
    out = {}
    for task in TASKS:
        pred = get_pred_item(data, task)
        out[task] = safe_float(pred.get("point"), 0.0)
    return out


def build_table(reports_dir: Path, labels_csv: Path) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_csv)
    labels_df["study_id"] = labels_df["study_id"].astype(str)

    rows = []

    for fp in sorted(reports_dir.glob("*.json")):
        try:
            data = load_json(fp)
        except Exception:
            continue

        sid = str(data.get("study_id", fp.stem))
        row = {"study_id": sid, "_path": str(fp)}

        task_points = extract_all_task_points(data)

        for task in TASKS:
            feats = extract_task_features(data, task)
            for k, v in feats.items():
                row[f"{task}__{k}"] = v

            # 任务间交叉特征
            other_tasks = [t for t in TASKS if t != task]
            row[f"{task}__other_point_mean"] = float(np.mean([task_points[t] for t in other_tasks]))
            row[f"{task}__point_minus_other_mean"] = float(task_points[task] - row[f"{task}__other_point_mean"])

            row[f"{task}__mortality_point"] = float(task_points["mortality_risk"])
            row[f"{task}__icu_point"] = float(task_points["icu_risk"])
            row[f"{task}__ventilation_point"] = float(task_points["ventilation_risk"])

            row[f"{task}__mortality_minus_icu"] = float(task_points["mortality_risk"] - task_points["icu_risk"])
            row[f"{task}__icu_minus_ventilation"] = float(task_points["icu_risk"] - task_points["ventilation_risk"])
            row[f"{task}__mortality_minus_ventilation"] = float(task_points["mortality_risk"] - task_points["ventilation_risk"])

        rows.append(row)

    pred_df = pd.DataFrame(rows)
    merged = labels_df.merge(pred_df, on="study_id", how="inner")
    return merged


def make_model(kind: str):
    if kind == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs"),
        )

    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=7,
            min_samples_leaf=6,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

    if kind == "gb":
        return GradientBoostingClassifier(
            n_estimators=260,
            learning_rate=0.025,
            max_depth=2,
            subsample=0.90,
            random_state=42,
        )

    raise ValueError(f"unknown model kind: {kind}")


def metrics_at_threshold(y_true, y_score, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= thr).astype(int)

    out = {
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Acc": accuracy_score(y_true, y_pred),
    }

    if len(np.unique(y_true)) >= 2:
        out["AUROC"] = roc_auc_score(y_true, y_score)
        out["AUPRC"] = average_precision_score(y_true, y_score)
    else:
        out["AUROC"] = np.nan
        out["AUPRC"] = np.nan

    return out


def scan_best_f1(y_true, y_score):
    best = {"thr": 0.5, "F1": -1.0, "Acc": -1.0}
    for thr in np.arange(0.05, 0.96, 0.01):
        m = metrics_at_threshold(y_true, y_score, thr)
        if (m["F1"] > best["F1"]) or (abs(m["F1"] - best["F1"]) < 1e-12 and m["Acc"] > best["Acc"]):
            best = {"thr": float(thr), "F1": float(m["F1"]), "Acc": float(m["Acc"])}
    return best


def get_feature_matrix(df: pd.DataFrame, task: str):
    feat_cols = [c for c in df.columns if c.startswith(f"{task}__")]
    X = df[feat_cols].astype(float).fillna(0.0).to_numpy()
    y = df[task].astype(int).to_numpy()
    return X, y, feat_cols


def oof_single_model(df: pd.DataFrame, task: str, model_kind: str, n_splits: int = 5) -> np.ndarray:
    X, y, _ = get_feature_matrix(df, task)
    oof = np.zeros(len(df), dtype=float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr, va in skf.split(X, y):
        model = make_model(model_kind)
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]

    return oof


def evaluate_score(task: str, y: np.ndarray, score: np.ndarray, model_name: str) -> dict:
    m05 = metrics_at_threshold(y, score, 0.5)
    best = scan_best_f1(y, score)

    return {
        "task": task,
        "n": len(y),
        "AUROC": m05["AUROC"],
        "AUPRC": m05["AUPRC"],
        "F1_0.5": m05["F1"],
        "Acc_0.5": m05["Acc"],
        "best_threshold": best["thr"],
        "best_F1": best["F1"],
        "best_Acc_at_best_F1": best["Acc"],
        "positive_rate": float(y.mean()),
        "model": model_name,
        "eval_mode": "oof_cv",
    }


def add_summary_rows(out_df: pd.DataFrame) -> pd.DataFrame:
    rows = [out_df]

    summary_rows = []
    for model_name, g in out_df.groupby("model"):
        summary_rows.append({
            "task": "MEAN",
            "n": int(g["n"].mean()),
            "AUROC": float(g["AUROC"].mean()),
            "AUPRC": float(g["AUPRC"].mean()),
            "F1_0.5": float(g["F1_0.5"].mean()),
            "Acc_0.5": float(g["Acc_0.5"].mean()),
            "best_threshold": np.nan,
            "best_F1": float(g["best_F1"].mean()),
            "best_Acc_at_best_F1": float(g["best_Acc_at_best_F1"].mean()),
            "positive_rate": float(g["positive_rate"].mean()),
            "model": model_name,
            "eval_mode": "oof_cv_mean",
        })

    rows.append(pd.DataFrame(summary_rows))
    return pd.concat(rows, ignore_index=True)


def run_oof_eval(df: pd.DataFrame, mode: str = "all", n_splits: int = 5) -> pd.DataFrame:
    rows = []

    for task in TASKS:
        _, y, _ = get_feature_matrix(df, task)

        oof_lr = oof_single_model(df, task, "lr", n_splits=n_splits)
        oof_rf = oof_single_model(df, task, "rf", n_splits=n_splits)
        oof_gb = oof_single_model(df, task, "gb", n_splits=n_splits)

        scores = {
            "lr": oof_lr,
            "rf": oof_rf,
            "gb": oof_gb,
            "ens_55gb_35rf_10lr": 0.55 * oof_gb + 0.35 * oof_rf + 0.10 * oof_lr,
            "ens_45gb_45rf_10lr": 0.45 * oof_gb + 0.45 * oof_rf + 0.10 * oof_lr,
            "ens_40gb_50rf_10lr": 0.40 * oof_gb + 0.50 * oof_rf + 0.10 * oof_lr,
            "ens_60gb_30rf_10lr": 0.60 * oof_gb + 0.30 * oof_rf + 0.10 * oof_lr,
        }

        for name, score in scores.items():
            rows.append(evaluate_score(task, y, score, name))

    return add_summary_rows(pd.DataFrame(rows))


def fit_model_full(df: pd.DataFrame, task: str, model_kind: str):
    X, y, feat_cols = get_feature_matrix(df, task)
    model = make_model(model_kind)
    model.fit(X, y)
    pred = model.predict_proba(X)[:, 1]
    return model, feat_cols, pred


def fit_all_and_write(df: pd.DataFrame, reports_dir: Path, out_dir: Path, ensemble_name: str):
    """
    写 calibrated reports。注意：这是 fit-all 写回版，适合生成最终文件；
    严格汇报结果优先看 OOF。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    all_scores: Dict[str, Dict[str, float]] = {}

    for task in TASKS:
        _, _, pred_lr = fit_model_full(df, task, "lr")
        _, _, pred_rf = fit_model_full(df, task, "rf")
        _, _, pred_gb = fit_model_full(df, task, "gb")

        if ensemble_name == "gb":
            pred = pred_gb
        elif ensemble_name == "rf":
            pred = pred_rf
        elif ensemble_name == "lr":
            pred = pred_lr
        elif ensemble_name == "ens_55gb_35rf_10lr":
            pred = 0.55 * pred_gb + 0.35 * pred_rf + 0.10 * pred_lr
        elif ensemble_name == "ens_45gb_45rf_10lr":
            pred = 0.45 * pred_gb + 0.45 * pred_rf + 0.10 * pred_lr
        elif ensemble_name == "ens_40gb_50rf_10lr":
            pred = 0.40 * pred_gb + 0.50 * pred_rf + 0.10 * pred_lr
        elif ensemble_name == "ens_60gb_30rf_10lr":
            pred = 0.60 * pred_gb + 0.30 * pred_rf + 0.10 * pred_lr
        else:
            raise ValueError(f"unknown ensemble_name: {ensemble_name}")

        for sid, score in zip(df["study_id"].astype(str).tolist(), pred.tolist()):
            all_scores.setdefault(sid, {})[task] = float(score)

    for fp in sorted(reports_dir.glob("*.json")):
        data = load_json(fp)
        sid = str(data.get("study_id", fp.stem))
        if sid not in all_scores:
            continue

        preds = data.get("predictions", [])
        if isinstance(preds, list):
            for item in preds:
                if not isinstance(item, dict):
                    continue
                task = item.get("task_name")
                if task not in TASKS:
                    continue

                old = safe_float(item.get("point"), 0.0)
                new = all_scores[sid][task]

                item["raw_point_before_calibration"] = old
                item["point"] = float(new)
                item["interval_low"] = max(0.0, float(new) - 0.12)
                item["interval_high"] = min(1.0, float(new) + 0.12)
                item["rationale"] = str(item.get("rationale", "")) + f", calibrated_{ensemble_name}={new:.3f}"

        save_json(data, out_dir / fp.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--save-csv", default="artifacts/calibrator_ensemble_metrics.csv")
    parser.add_argument("--out-dir", default="artifacts/reports_calibrated")
    parser.add_argument("--write-calibrated", action="store_true")
    parser.add_argument(
        "--write-model",
        default="ens_45gb_45rf_10lr",
        choices=[
            "lr",
            "rf",
            "gb",
            "ens_55gb_35rf_10lr",
            "ens_45gb_45rf_10lr",
            "ens_40gb_50rf_10lr",
            "ens_60gb_30rf_10lr",
        ],
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    labels_csv = Path(args.labels_csv)

    df = build_table(reports_dir, labels_csv)
    print("matched samples:", len(df))

    metrics = run_oof_eval(df, mode="all", n_splits=5)
    metrics.to_csv(args.save_csv, index=False)

    print(metrics.to_string(index=False))
    print(f"Saved metrics to {args.save_csv}")

    if args.write_calibrated:
        fit_all_and_write(
            df=df,
            reports_dir=reports_dir,
            out_dir=Path(args.out_dir),
            ensemble_name=args.write_model,
        )
        print(f"Saved calibrated reports to {args.out_dir} using {args.write_model}")


if __name__ == "__main__":
    main()