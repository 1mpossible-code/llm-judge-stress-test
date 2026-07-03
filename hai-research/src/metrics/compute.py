"""Metrics computation (flip rate, ASR, JSD, bias amplification)"""

import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from typing import Any

from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from src.schemas import Judgment
from src.utils.io import read_jsonl, read_json, write_json, write_jsonl
from src.metrics.bootstrap import compute_bootstrap_ci


def load_judgments(judgments_file: Path) -> list[Judgment]:
    """Load all judgments from JSONL."""
    return [Judgment(**item) for item in read_jsonl(judgments_file)]


def compute_label_distribution(judgments: list[Judgment]) -> dict[str, float]:
    """Compute label distribution from judgments."""
    labels = [j.label for j in judgments if j.label is not None]
    if not labels:
        return {}
    counts = Counter(labels)
    total = len(labels)
    return {label: count / total for label, count in counts.items()}


def compute_entropy(dist: dict[str, float]) -> float:
    """Compute Shannon entropy of a label distribution."""
    if not dist:
        return 0.0
    probs = np.array(list(dist.values()))
    probs = probs[probs > 0]  # Remove zeros
    if len(probs) == 0:
        return 0.0
    return float(entropy(probs, base=2))


def get_entropy_bucket(entropy_val: float) -> str:
    """Bucket entropy into categories."""
    if np.isnan(entropy_val) or entropy_val == 0.0:
        return "low"  # No disagreement
    elif entropy_val < 0.5:
        return "low"
    elif entropy_val < 1.0:
        return "medium"
    else:
        return "high"


def compute_flip_rate(control_judgments: list[Judgment], perturbed_judgments: list[Judgment]) -> float:
    """Compute flip rate: fraction of items that changed label from control to perturbed.
    
    Requires trial-aligned judgments (same trial_idx).
    """
    if len(control_judgments) != len(perturbed_judgments):
        return np.nan
    
    flips = 0
    valid_pairs = 0
    
    # Group by base_id and trial_idx
    control_map = {(j.base_id, j.trial_idx): j.label for j in control_judgments if j.label}
    perturbed_map = {(j.base_id, j.trial_idx): j.label for j in perturbed_judgments if j.label}
    
    for key in control_map:
        if key in perturbed_map:
            valid_pairs += 1
            if control_map[key] != perturbed_map[key]:
                flips += 1
    
    return flips / valid_pairs if valid_pairs > 0 else 0.0


def compute_asr(control_judgments: list[Judgment], perturbed_judgments: list[Judgment]) -> float:
    """Attack Success Rate: same as flip rate."""
    return compute_flip_rate(control_judgments, perturbed_judgments)


def compute_jsd(model_dist: dict[str, float], human_dist: dict[str, float]) -> float:
    """Compute Jensen-Shannon divergence between model and human distributions.
    
    SciPy's ``jensenshannon`` returns the Jensen-Shannon *distance*, i.e. the
    square root of the divergence. We square it and use base 2 so the binary
    divergence is bounded in [0, 1].
    """
    # Get union of labels
    all_labels = sorted(set(list(model_dist.keys()) + list(human_dist.keys())))
    if not all_labels:
        return 0.0
    
    model_vec = np.array([model_dist.get(label, 0.0) for label in all_labels])
    human_vec = np.array([human_dist.get(label, 0.0) for label in all_labels])
    
    # Normalize
    if model_vec.sum() > 0:
        model_vec = model_vec / model_vec.sum()
    if human_vec.sum() > 0:
        human_vec = human_vec / human_vec.sum()
    
    return float(jensenshannon(model_vec, human_vec, base=2.0) ** 2)


def compute_self_consistency(judgments: list[Judgment]) -> float:
    """Compute self-consistency: agreement across trials for same item.
    
    Returns fraction of items where all trials agree.
    """
    # Group by (base_id, perturb_id)
    groups: dict[tuple[str, str], list[str]] = {}
    
    for j in judgments:
        if j.label:
            key = (j.base_id, j.perturb_id)
            if key not in groups:
                groups[key] = []
            groups[key].append(j.label)
    
    if not groups:
        return 0.0
    
    consistent = 0
    for labels in groups.values():
        if len(set(labels)) == 1:  # All same label
            consistent += 1
    
    return consistent / len(groups)


def compute_bias_amplification(
    control_judgments: list[Judgment],
    perturbed_judgments: list[Judgment],
    binary_label_positive: str,
) -> float:
    """Compute bias amplification for binary labels.
    
    Measures how much the positive label rate increases from control to perturbed.
    """
    control_pos = sum(1 for j in control_judgments if j.label == binary_label_positive)
    perturbed_pos = sum(1 for j in perturbed_judgments if j.label == binary_label_positive)
    
    control_rate = control_pos / len(control_judgments) if control_judgments else 0.0
    perturbed_rate = perturbed_pos / len(perturbed_judgments) if perturbed_judgments else 0.0
    
    return perturbed_rate - control_rate


def compute_all_metrics(
    run_dir: Path,
    human_label_dists: dict[str, dict[str, float]],
    bootstrap_samples: int = 200,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Compute all metrics for a run.
    
    Args:
        run_dir: Path to run directory
        human_label_dists: Dict mapping base_id to human label distribution
        bootstrap_samples: Number of bootstrap samples
        confidence_level: Confidence level for CIs
    """
    judgments_file = run_dir / "judgments.jsonl"
    judgments = load_judgments(judgments_file)
    
    # Filter to successful judgments
    valid_judgments = [j for j in judgments if j.status == "ok" and j.label]
    
    if not valid_judgments:
        return pd.DataFrame()
    
    # Build data structures
    df = pd.DataFrame([j.to_dict() for j in valid_judgments])
    
    # Get language and task info if available
    # Load judgments to get language from base_id mapping
    language_map = {}
    task_name = None
    if human_label_dists:
        # Try to get language from dataset metadata
        from src.utils.io import read_jsonl
        judgments_file = run_dir / "judgments.jsonl"
        if judgments_file.exists():
            for j_dict in read_jsonl(judgments_file):
                if "base_id" in j_dict:
                    # We'll need to track this better, but for now use a simple approach
                    pass
    
    # Control judgments have factor == "control"
    # No need for complex mask, we'll filter by factor field
    
    # For each model, language, factor, level combination
    results = []
    
    # Get available grouping dimensions
    languages = df.get("language", pd.Series(["unknown"] * len(df))).unique() if "language" in df.columns else ["unknown"]
    
    for model in df["model"].unique():
        for backend in df[df["model"] == model]["backend"].unique():
            model_df = df[(df["model"] == model) & (df["backend"] == backend)]
            
            # Group by language if available
            for language in languages:
                if "language" in model_df.columns:
                    lang_df = model_df[model_df["language"] == language]
                else:
                    lang_df = model_df
                    language = "unknown"
                
                if lang_df.empty:
                    continue
            
                # Get control judgments for this model/language (factor == "control")
                control_judgments_df = lang_df[lang_df.get("factor") == "control"]
                control_judgments = [
                    Judgment(**row.to_dict()) 
                    for _, row in control_judgments_df.iterrows()
                ]
                
                # Get control label dist
                control_dist = compute_label_distribution(control_judgments)
                
                # For each perturbation (group by factor/level combination)
                # Group by factor and level (not perturb_id, since each base_id has different perturb_id)
                pert_groups = lang_df.groupby(["factor", "level"]).first().reset_index()
                
                for _, pert_row in pert_groups.iterrows():
                    factor_name_val = pert_row["factor"]
                    level_name_val = pert_row["level"]
                    
                    if factor_name_val == "control":
                        continue  # Skip control itself
                    
                    # Get all judgments for this factor/level combination
                    pert_judgments_df = lang_df[
                        (lang_df["factor"] == factor_name_val) & 
                        (lang_df["level"] == level_name_val)
                    ]
                
                    perturb_judgments = [
                        Judgment(**row.to_dict())
                        for _, row in pert_judgments_df.iterrows()
                    ]
                    
                    if not perturb_judgments:
                        continue
                    
                    factor_name = factor_name_val
                    level_name = level_name_val
                    
                    # Use a representative perturb_id for grouping (first one)
                    factor_id = pert_judgments_df.iloc[0]["perturb_id"] if len(pert_judgments_df) > 0 else "unknown"
                    
                    # Compute metrics
                    perturb_dist = compute_label_distribution(perturb_judgments)
                    
                    # Get human dist for base items in this perturbation
                    base_ids = set(j.base_id for j in perturb_judgments)
                    human_dist = {}
                    human_entropy = np.nan
                    if base_ids:
                        # Aggregate human dists
                        all_human_dists = [human_label_dists.get(bid, {}) for bid in base_ids]
                        if all_human_dists:
                            all_labels = set()
                            for hd in all_human_dists:
                                all_labels.update(hd.keys())
                            human_dist = {
                                label: np.mean([hd.get(label, 0.0) for hd in all_human_dists])
                                for label in all_labels
                            }
                            # Compute entropy
                            human_entropy = compute_entropy(human_dist)
                    
                    entropy_bucket = get_entropy_bucket(human_entropy)
                    
                    # Get perturbation category from factor name
                    perturbation_category = factor_name if factor_name != "control" else "control"
                    
                    jsd_model_human = compute_jsd(perturb_dist, human_dist) if human_dist else np.nan
                    jsd_control_human = compute_jsd(control_dist, human_dist) if human_dist else np.nan
                    delta_jsd = jsd_model_human - jsd_control_human
                    
                    flip_rate = compute_flip_rate(control_judgments, perturb_judgments)
                    self_consistency = compute_self_consistency(perturb_judgments)
                    
                    # Bootstrap CIs for flip rate
                    # Align by (base_id, trial_idx) for proper comparison
                    control_map = {(j.base_id, j.trial_idx): j.label for j in control_judgments if j.label}
                    perturb_map = {(j.base_id, j.trial_idx): j.label for j in perturb_judgments if j.label}
                    
                    aligned_keys = set(control_map.keys()) & set(perturb_map.keys())
                    if aligned_keys and len(aligned_keys) > 0:
                        flips = np.array([
                            1 if control_map[key] != perturb_map[key] else 0
                            for key in aligned_keys
                        ])
                        
                        if len(flips) > 0:
                            if flips.sum() > 0:
                                _, flip_lower, flip_upper = compute_bootstrap_ci(
                                    flips,
                                    lambda x: x.mean(),
                                    bootstrap_samples,
                                    confidence_level,
                                )
                            else:
                                flip_lower = flip_upper = 0.0
                        else:
                            flip_lower = flip_upper = np.nan
                    else:
                        flip_lower = flip_upper = np.nan
                    
                    results.append({
                        "model": model,
                        "backend": backend,
                        "language": language,
                        "perturbation_category": perturbation_category,
                        "factor": factor_name,
                        "level": level_name,
                        "perturb_id": factor_id,
                        "entropy_bucket": entropy_bucket,
                        "human_entropy": float(human_entropy) if not np.isnan(human_entropy) else None,
                        "flip_rate": flip_rate,
                        "flip_rate_lower_ci": flip_lower,
                        "flip_rate_upper_ci": flip_upper,
                        "jsd_model_human": jsd_model_human,
                        "jsd_control_human": jsd_control_human,
                        "delta_jsd": delta_jsd,
                        "self_consistency": self_consistency,
                        "bias_amplification": np.nan,  # Would need to compute if binary labels
                        "n_judgments": len(perturb_judgments),
                    })
    
    return pd.DataFrame(results)


def compute_metrics_for_run(run_dir: Path) -> None:
    """Compute and save metrics for a run."""
    # Load metadata to get config
    meta_file = run_dir / "meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")
    
    meta = read_json(meta_file)
    config = meta["config"]
    
    # Load base dataset to get human label distributions
    dataset_path = Path(config["dataset"]["path"])
    from src.dataset.load_base import load_base_dataset
    from src.dataset.normalize import normalize_example
    
    human_label_dists = {}
    for base_ex in load_base_dataset(dataset_path, config["dataset"]["languages"]):
        norm_ex = normalize_example(base_ex, config["dataset"]["allowed_labels"])
        human_label_dists[base_ex.id] = norm_ex.human_label_dist
    
    # Compute metrics
    metrics_df = compute_all_metrics(
        run_dir,
        human_label_dists,
        bootstrap_samples=config["metrics"]["bootstrap_samples"],
        confidence_level=config["metrics"]["confidence_level"],
    )
    
    # Save metrics
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    if metrics_df.empty:
        print("Warning: No metrics computed (likely because only control perturbations were used)")
        # Create empty CSV with headers for consistency
        empty_df = pd.DataFrame(columns=[
            "model", "backend", "language", "perturbation_category", "factor", "level", "perturb_id",
            "entropy_bucket", "human_entropy",
            "flip_rate", "flip_rate_lower_ci", "flip_rate_upper_ci",
            "jsd_model_human", "jsd_control_human", "delta_jsd",
            "self_consistency", "bias_amplification", "n_judgments"
        ])
        empty_df.to_csv(metrics_dir / "metrics.csv", index=False)
        empty_df.to_csv(metrics_dir / "metrics_with_ci.csv", index=False)
    else:
        metrics_df.to_csv(metrics_dir / "metrics.csv", index=False)
        # Save with CIs - grouped by model×language×perturbation_category×perturb_id (+ entropy_bucket)
        metrics_with_ci = metrics_df.copy()
        # Sort by grouping dimensions
        metrics_with_ci = metrics_with_ci.sort_values([
            "model", "language", "perturbation_category", "perturb_id", "entropy_bucket"
        ])
        metrics_with_ci.to_csv(metrics_dir / "metrics_with_ci.csv", index=False)
    
    # Per-item metrics (simplified for now)
    judgments_file = run_dir / "judgments.jsonl"
    judgments = load_judgments(judgments_file)
    per_item = [j.to_dict() for j in judgments]
    write_jsonl(metrics_dir / "per_item_metrics.jsonl", per_item)
    
    # Generate paired_results.jsonl
    print("Generating paired_results.jsonl...")
    from src.metrics.paired_results import generate_paired_results
    generate_paired_results(run_dir)
    
    # Generate exhibits.jsonl (by delta_jsd and confidence_swing)
    print("Generating exhibits.jsonl...")
    from src.metrics.exhibits import generate_exhibits
    generate_exhibits(run_dir, top_n=20, sort_by="delta_jsd")
    generate_exhibits(run_dir, top_n=20, sort_by="confidence_swing")

