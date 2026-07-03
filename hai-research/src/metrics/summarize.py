"""Create paper-ready aggregate tables from one or more run directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import read_json, read_jsonl


DIRECTION_LABELS = {
    "not_to_offensive": "not$\\rightarrow$off",
    "offensive_to_not": "off$\\rightarrow$not",
    "other": "other",
}


def _display_model(model: str) -> str:
    if model == "llama-3.3-70b-versatile":
        return "Llama 3.3 70B"
    if model == "llama-3.1-8b-instant":
        return "Llama 3.1 8B"
    if model == "qwen/qwen3-32b":
        return "Qwen3 32B"
    if model == "gpt-4o-mini":
        return "GPT-4o-mini"
    if model.lower() in {"gpt-5.4", "openai-gpt-5.4"}:
        return "GPT-5.4"
    if model.lower() in {"haiku", "claude-haiku"}:
        return "Claude Haiku"
    if model.lower() == "claude-haiku-4-5-20251001":
        return "Claude Haiku 4.5"
    if "claude" in model.lower() and "haiku" in model.lower():
        return "Claude Haiku"
    if model.lower() == "gemini 3.5 flash (low)":
        return "Gemini 3.5 Flash"
    if "gemini" in model.lower():
        return "Gemini"
    return model


def _read_paired(run_dir: Path) -> pd.DataFrame:
    paired_file = run_dir / "paired_results.jsonl"
    if not paired_file.exists():
        raise FileNotFoundError(f"Missing paired results: {paired_file}")
    rows = list(read_jsonl(paired_file))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    meta = read_json(run_dir / "meta.json")
    df["run_id"] = meta.get("run_id", run_dir.name)
    df["backend"] = meta.get("config", {}).get("judging", {}).get("models", [{}])[0].get("backend")
    return df


def _read_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_file = run_dir / "metrics" / "metrics.csv"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Missing metrics: {metrics_file}")
    df = pd.read_csv(metrics_file)
    if not df.empty:
        df["run_dir"] = str(run_dir)
    return df


def summarize_runs(run_dirs: list[str | Path], out_dir: str | Path) -> dict[str, Path]:
    """Summarize run directories and write CSV/LaTeX tables."""
    run_paths = [Path(p) for p in run_dirs]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paired = pd.concat([_read_paired(p) for p in run_paths], ignore_index=True)
    metrics = pd.concat([_read_metrics(p) for p in run_paths], ignore_index=True)

    if paired.empty:
        raise ValueError("No paired results available to summarize")

    # Overall stability table.
    overall_rows: list[dict[str, Any]] = []
    for model, group in paired.groupby("model"):
        flips = group[group["flipped"] == True]
        flip_rate = float(group["flipped"].mean()) if len(group) else 0.0
        mean_delta_jsd = float(group["delta_jsd"].dropna().mean()) if "delta_jsd" in group else float("nan")

        if flips.empty:
            directionality = "none"
        else:
            direction_counts = flips["flip_type"].value_counts(normalize=True)
            top_direction = direction_counts.index[0]
            directionality = f"{direction_counts.iloc[0] * 100:.1f}\\% {DIRECTION_LABELS.get(top_direction, top_direction)}"

        overall_rows.append({
            "Model": _display_model(model),
            "Pairs": len(group),
            "Flip Rate": flip_rate,
            "Mean $\\Delta$JSD": mean_delta_jsd,
            "Flip Directionality": directionality,
        })

    overall = pd.DataFrame(overall_rows).sort_values("Model")

    # Category/factor effects table from metrics.
    if metrics.empty:
        category = pd.DataFrame()
    else:
        category = (
            metrics.groupby(["model", "perturbation_category", "factor", "level"], dropna=False)
            .agg(
                asr=("flip_rate", "mean"),
                delta_jsd=("delta_jsd", "mean"),
                n_judgments=("n_judgments", "sum"),
            )
            .reset_index()
        )
        category["Model"] = category["model"].map(_display_model)
        category = category.sort_values(["Model", "asr"], ascending=[True, False])

    # Top individual perturbations.
    top = category.sort_values("asr", ascending=False).head(20) if not category.empty else pd.DataFrame()

    outputs = {
        "overall_csv": out / "overall_stability.csv",
        "category_csv": out / "category_effects.csv",
        "top_csv": out / "top_perturbations.csv",
        "overall_tex": out / "overall_stability.tex",
        "category_tex": out / "category_effects.tex",
        "top_tex": out / "top_perturbations.tex",
    }

    overall.to_csv(outputs["overall_csv"], index=False)
    if not category.empty:
        category.to_csv(outputs["category_csv"], index=False)
        top.to_csv(outputs["top_csv"], index=False)
    else:
        category.to_csv(outputs["category_csv"], index=False)
        top.to_csv(outputs["top_csv"], index=False)

    # LaTeX formatting.
    latex_overall = overall.copy()
    latex_overall["Flip Rate"] = latex_overall["Flip Rate"].map(lambda x: f"{x * 100:.2f}\\%")
    latex_overall["Mean $\\Delta$JSD"] = latex_overall["Mean $\\Delta$JSD"].map(lambda x: f"{x:.4f}")
    outputs["overall_tex"].write_text(
        latex_overall.to_latex(index=False, escape=False), encoding="utf-8"
    )

    if not category.empty:
        latex_category = category[["Model", "perturbation_category", "level", "asr", "delta_jsd"]].copy()
        latex_category["asr"] = latex_category["asr"].map(lambda x: f"{x * 100:.2f}\\%")
        latex_category["delta_jsd"] = latex_category["delta_jsd"].map(lambda x: f"{x:.4f}")
        latex_category = latex_category.rename(columns={
            "perturbation_category": "Category",
            "level": "Perturbation",
            "asr": "ASR",
            "delta_jsd": "Avg $\\Delta$JSD",
        })
        outputs["category_tex"].write_text(
            latex_category.to_latex(index=False, escape=False), encoding="utf-8"
        )

        latex_top = top[["Model", "perturbation_category", "level", "asr", "delta_jsd"]].copy()
        latex_top["asr"] = latex_top["asr"].map(lambda x: f"{x * 100:.2f}\\%")
        latex_top["delta_jsd"] = latex_top["delta_jsd"].map(lambda x: f"{x:.4f}")
        latex_top = latex_top.rename(columns={
            "perturbation_category": "Category",
            "level": "Perturbation",
            "asr": "ASR",
            "delta_jsd": "$\\Delta$JSD",
        })
        outputs["top_tex"].write_text(latex_top.to_latex(index=False, escape=False), encoding="utf-8")
    else:
        outputs["category_tex"].write_text("", encoding="utf-8")
        outputs["top_tex"].write_text("", encoding="utf-8")

    return outputs
