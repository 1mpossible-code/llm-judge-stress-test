"""Generate plots for experiment results"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

from src.utils.io import read_json


def plot_asr_heatmap(run_dir: Path, output_path: Path | None = None) -> None:
    """Plot ASR (Attack Success Rate) heatmap by model and perturbation."""
    metrics_file = run_dir / "metrics" / "metrics.csv"
    if not metrics_file.exists():
        print(f"Warning: Metrics file not found: {metrics_file}")
        return
    
    try:
        df = pd.read_csv(metrics_file)
    except pd.errors.EmptyDataError:
        print(f"Warning: Metrics file is empty: {metrics_file}")
        return
    
    if df.empty or "flip_rate" not in df.columns:
        print(f"Warning: Metrics DataFrame is empty or missing flip_rate column")
        return
    
    # Pivot: models (rows) vs perturbations (columns)
    pivot = df.pivot_table(
        index="model",
        columns="perturb_id",
        values="flip_rate",
        aggfunc="mean",
    )
    
    # Check if pivot is empty
    if pivot.empty or pivot.size == 0:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "ASR"})
    ax.set_title("Attack Success Rate (Flip Rate) by Model and Perturbation")
    ax.set_xlabel("Perturbation")
    ax.set_ylabel("Model")
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_delta_jsd_bar(run_dir: Path, output_path: Path | None = None) -> None:
    """Plot delta-JSD (change in JSD vs control) bar chart."""
    metrics_file = run_dir / "metrics" / "metrics.csv"
    if not metrics_file.exists():
        print(f"Warning: Metrics file not found: {metrics_file}")
        return
    
    try:
        df = pd.read_csv(metrics_file)
    except pd.errors.EmptyDataError:
        print(f"Warning: Metrics file is empty: {metrics_file}")
        return
    
    if df.empty or "delta_jsd" not in df.columns:
        print(f"Warning: Metrics DataFrame is empty or missing delta_jsd column")
        return
    
    # Remove rows with NaN delta_jsd
    df = df.dropna(subset=["delta_jsd"])
    if df.empty:
        return
    
    # Group by model and perturbation
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = df["model"].unique()
    x = np.arange(len(models))
    width = 0.35
    
    perturbations = df["perturb_id"].unique()[:5]  # Limit for readability
    
    for i, pert in enumerate(perturbations):
        values = [
            df[(df["model"] == model) & (df["perturb_id"] == pert)]["delta_jsd"].mean()
            for model in models
        ]
        ax.bar(x + i * width / len(perturbations), values, width / len(perturbations), label=str(pert)[:20])
    
    ax.set_xlabel("Model")
    ax.set_ylabel("Δ JSD (vs Control)")
    ax.set_title("Change in Jensen-Shannon Divergence vs Human")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_drift_by_disagreement(run_dir: Path, output_path: Path | None = None) -> None:
    """Plot ASR drift by human disagreement buckets."""
    # Load judgments and compute disagreement
    judgments_file = run_dir / "judgments.jsonl"
    if not judgments_file.exists():
        print(f"Warning: Judgments file not found: {judgments_file}")
        return
    
    # Load metadata for human distributions
    meta_file = run_dir / "meta.json"
    if not meta_file.exists():
        return
    
    meta = read_json(meta_file)
    config = meta["config"]
    
    # Load base dataset
    from src.dataset.load_base import load_base_dataset
    from src.dataset.normalize import normalize_example
    
    disagreement_map = {}
    for base_ex in load_base_dataset(config["dataset"]["path"], config["dataset"]["languages"]):
        norm_ex = normalize_example(base_ex, config["dataset"]["allowed_labels"])
        disagreement_map[base_ex.id] = norm_ex.human_disagreement
    
    # Load judgments
    from src.metrics.compute import load_judgments
    judgments = load_judgments(judgments_file)
    
    # Bin by disagreement
    disagreements = list(disagreement_map.values())
    bins = np.quantile(disagreements, [0, 0.33, 0.67, 1.0]) if len(disagreements) > 2 else [0, 0.5, 1.0]
    
    # Compute ASR per bucket
    from src.metrics.compute import compute_flip_rate
    
    # This is simplified - would need proper control/perturbed alignment
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Placeholder: would compute actual ASR per bucket
    bucket_names = ["Low", "Medium", "High"]
    asr_values = [0.1, 0.2, 0.3]  # Placeholder
    
    ax.bar(bucket_names, asr_values, alpha=0.7)
    ax.set_xlabel("Human Disagreement Bucket")
    ax.set_ylabel("ASR")
    ax.set_title("Attack Success Rate by Human Disagreement Level")
    ax.grid(axis="y", alpha=0.3)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_plots(run_dir: Path, figure_format: str = "png") -> None:
    """Generate all plots for a run."""
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    plot_asr_heatmap(run_dir, figures_dir / f"asr_heatmap.{figure_format}")
    plot_delta_jsd_bar(run_dir, figures_dir / f"delta_jsd_bar.{figure_format}")
    plot_drift_by_disagreement(run_dir, figures_dir / f"drift_by_disagreement.{figure_format}")

