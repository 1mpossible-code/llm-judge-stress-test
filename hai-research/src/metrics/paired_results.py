"""Generate paired_results.jsonl with control vs perturbed comparisons"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Any
import numpy as np
from scipy.stats import entropy

from src.utils.io import read_jsonl, write_jsonl, read_json
from src.schemas import Judgment
from src.metrics.compute import compute_jsd


def compute_entropy(dist: dict[str, float]) -> float:
    """Compute Shannon entropy of a label distribution."""
    if not dist:
        return 0.0
    probs = np.array(list(dist.values()))
    probs = probs[probs > 0]  # Remove zeros
    if len(probs) == 0:
        return 0.0
    return float(entropy(probs, base=2))


def generate_paired_results(run_dir: Path) -> None:
    """Generate paired_results.jsonl with control vs perturbed comparisons."""
    judgments_file = run_dir / "judgments.jsonl"
    meta_file = run_dir / "meta.json"
    
    if not judgments_file.exists():
        raise FileNotFoundError(f"Judgments file not found: {judgments_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")
    
    # Load metadata
    meta = read_json(meta_file)
    config = meta["config"]
    
    # Load human label distributions
    dataset_path = Path(config["dataset"]["path"])
    from src.dataset.load_base import load_base_dataset
    from src.dataset.normalize import normalize_example
    
    human_label_dists = {}
    for base_ex in load_base_dataset(dataset_path, config["dataset"]["languages"]):
        norm_ex = normalize_example(base_ex, config["dataset"]["allowed_labels"])
        human_label_dists[base_ex.id] = norm_ex.human_label_dist
    
    # Load judgments
    judgments = []
    for j_dict in read_jsonl(judgments_file):
        try:
            judgments.append(Judgment(**j_dict))
        except Exception as e:
            # Skip invalid judgments
            continue
    
    # Filter to successful judgments
    valid_judgments = [j for j in judgments if j.status == "ok" and j.label is not None]
    
    # Group by (base_id, model, trial_idx)
    grouped = defaultdict(lambda: {"control": None, "perturbed": []})
    
    for j in valid_judgments:
        key = (j.base_id, j.model, j.trial_idx)
        if j.factor == "control" and j.level == "original":
            grouped[key]["control"] = j
        else:
            grouped[key]["perturbed"].append(j)
    
    # Generate paired results
    paired_results = []
    
    for (base_id, model, trial_idx), group in grouped.items():
        control = group["control"]
        if control is None:
            continue  # Skip if no control judgment
        
        human_dist = human_label_dists.get(base_id, {})
        human_entropy = compute_entropy(human_dist)
        
        for perturbed in group["perturbed"]:
            # Compute label distributions
            control_dist = {control.label: 1.0} if control.label else {}
            perturbed_dist = {perturbed.label: 1.0} if perturbed.label else {}
            
            # Compute JSDs
            jsd_control = compute_jsd(control_dist, human_dist) if human_dist else np.nan
            jsd_perturbed = compute_jsd(perturbed_dist, human_dist) if human_dist else np.nan
            delta_jsd = jsd_perturbed - jsd_control
            
            # Determine if flipped
            flipped = control.label != perturbed.label if (control.label and perturbed.label) else False
            
            # Determine flip type
            flip_type = None
            if flipped:
                if control.label == "offensive" and perturbed.label == "not_offensive":
                    flip_type = "offensive_to_not"
                elif control.label == "not_offensive" and perturbed.label == "offensive":
                    flip_type = "not_to_offensive"
                else:
                    flip_type = "other"
            
            # Compute confidence swing
            control_conf = control.confidence if control.confidence is not None else 0.0
            perturbed_conf = perturbed.confidence if perturbed.confidence is not None else 0.0
            confidence_swing = perturbed_conf - control_conf
            
            # Create paired result
            paired_result = {
                "base_id": base_id,
                "model": model,
                "trial_idx": trial_idx,
                "language": control.language,
                "perturbation_id": perturbed.perturb_id,
                "perturbation_factor": perturbed.factor,
                "perturbation_level": perturbed.level,
                "control_perturbation_id": control.perturb_id,
                # Control judgment
                "control_label": control.label,
                "control_confidence": control.confidence,
                "control_rationale": control.rationale,
                "control_input_text": control.input_text,
                "control_perturbation_text": control.perturbation_text_applied,
                # Perturbed judgment
                "perturbed_label": perturbed.label,
                "perturbed_confidence": perturbed.confidence,
                "perturbed_rationale": perturbed.rationale,
                "perturbed_input_text": perturbed.input_text,
                "perturbed_perturbation_text": perturbed.perturbation_text_applied,
                # Comparison metrics
                "flipped": flipped,
                "flip_type": flip_type,
                "confidence_swing": confidence_swing,
                "jsd_control": float(jsd_control) if not np.isnan(jsd_control) else None,
                "jsd_perturbed": float(jsd_perturbed) if not np.isnan(jsd_perturbed) else None,
                "delta_jsd": float(delta_jsd) if not np.isnan(delta_jsd) else None,
                # Human annotations
                "human_label_distribution": human_dist,
                "human_entropy": float(human_entropy) if not np.isnan(human_entropy) else None,
            }
            
            paired_results.append(paired_result)
    
    # Save paired results
    output_file = run_dir / "paired_results.jsonl"
    write_jsonl(output_file, paired_results)
    
    print(f"Generated {len(paired_results)} paired results in {output_file}")

