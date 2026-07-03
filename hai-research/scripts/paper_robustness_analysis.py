#!/usr/bin/env python3
"""Generate uncertainty, paired-test, and item-level analyses for the paper.

The main analysis uses majority labels over the three repeated calls for each
(model, item, condition). Raw per-call analyses are also emitted as robustness
checks. The script intentionally avoids provider calls; it only reads completed
run artifacts.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_artifacts" / "robustness"
FIG = OUT / "figures"

RUNS = [
    ROOT / "runs/2026-06-26_023709_exp_core_claude_haiku",
    ROOT / "runs/2026-06-25_220140_exp_core_gpt54",
    ROOT / "runs/2026-06-25_212106_exp_core_antigravity_gemini_flash",
    ROOT / "runs/2026-06-25_204534_exp_core_groq_llama31_8b",
    ROOT / "runs/2026-06-25_212409_exp_core_groq_qwen3_32b",
]

DISPLAY = {
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "gpt-5.4": "GPT-5.4",
    "Gemini 3.5 Flash (Low)": "Gemini 3.5 Flash",
    "llama-3.1-8b-instant": "Llama 3.1 8B",
    "qwen/qwen3-32b": "Qwen3 32B",
}
MODEL_ORDER = [
    "Claude Haiku 4.5",
    "GPT-5.4",
    "Gemini 3.5 Flash",
    "Llama 3.1 8B",
    "Qwen3 32B",
]
PERT_ORDER = [
    "west_comparison",
    "ingroup_pride",
    "west_bias_medium",
    "ingroup_medium",
    "modern_decay",
    "expert_consensus",
    "padded_fluent",
    "affirmative_medium",
    "negative_medium",
]
PERT_LABEL = {
    "west_comparison": "West comparison",
    "ingroup_pride": "In-group pride",
    "west_bias_medium": "West-bias framing",
    "ingroup_medium": "In-group framing",
    "modern_decay": "Traditionalism",
    "expert_consensus": "Authority",
    "padded_fluent": "Verbosity",
    "affirmative_medium": "Affirmative framing",
    "negative_medium": "Negative framing",
}
LABEL_TO_INT = {"not_offensive": 0, "offensive": 1}


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_pct(x: float) -> str:
    return f"{100*x:.1f}\\%"


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{100*lo:.1f}, {100*hi:.1f}]"


def tex_escape(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%")


def jsd_one_hot(label: str, human_off_rate: float) -> float:
    p = np.array([1.0 if label == "offensive" else 0.0, 1.0 if label == "not_offensive" else 0.0])
    q = np.array([human_off_rate, 1.0 - human_off_rate])
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def bh_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        true_rank = m - rank + 1
        val = min(prev, pvals[idx] * m / true_rank)
        adj[idx] = min(1.0, val)
        prev = val
    return adj


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_sample():
    sample = {}
    for row in read_jsonl(ROOT / "data/lewidi/paper_sample_30_balanced.jsonl"):
        labels = row["human_labels"]
        off = sum(1 for x in labels if x == "offensive")
        sample[row["id"]] = {
            "off_rate": off / len(labels),
            "votes": f"{off}/{len(labels)}",
            "text_hash_id": row["id"],
        }
    return sample


def majority_label(labels: list[str]) -> str:
    counts = Counter(labels)
    if counts["offensive"] > counts["not_offensive"]:
        return "offensive"
    return "not_offensive"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    sample = load_sample()

    raw_pairs = []
    judgments = []
    for run in RUNS:
        for row in read_jsonl(run / "paired_results.jsonl"):
            row["display_model"] = DISPLAY.get(row["model"], row["model"])
            raw_pairs.append(row)
        for row in read_jsonl(run / "judgments.jsonl"):
            row["display_model"] = DISPLAY.get(row["model"], row["model"])
            judgments.append(row)

    # Majority labels by model-item-condition.
    groups = defaultdict(list)
    for j in judgments:
        key = (j["display_model"], j["base_id"], j["factor"], j["level"])
        groups[key].append(j)

    majority = {}
    self_rows = []
    for key, vals in groups.items():
        labels = [v["label"] for v in vals if v.get("status") == "ok"]
        confs = [float(v["confidence"]) for v in vals if v.get("confidence") is not None]
        if len(labels) != 3:
            continue
        model, base_id, factor, level = key
        pairwise = sum(1 for a in range(3) for b in range(a + 1, 3) if labels[a] == labels[b]) / 3
        majority[key] = {
            "label": majority_label(labels),
            "mean_confidence": mean(confs),
            "unanimous": len(set(labels)) == 1,
            "pairwise_agreement": pairwise,
            "labels": labels,
        }

    for model in MODEL_ORDER:
        model_groups = [(k, v) for k, v in majority.items() if k[0] == model]
        all_n = len(model_groups)
        ctrl_groups = [(k, v) for k, v in model_groups if k[2] == "control"]
        self_rows.append(
            {
                "model": model,
                "all_conditions": all_n,
                "unanimity_rate": mean([v["unanimous"] for _, v in model_groups]),
                "pairwise_agreement": mean([v["pairwise_agreement"] for _, v in model_groups]),
                "control_conditions": len(ctrl_groups),
                "control_unanimity": mean([v["unanimous"] for _, v in ctrl_groups]),
                "control_pairwise_agreement": mean([v["pairwise_agreement"] for _, v in ctrl_groups]),
            }
        )

    # Majority paired rows.
    maj_pairs = []
    for model in MODEL_ORDER:
        for base_id, meta in sample.items():
            ctrl = majority.get((model, base_id, "control", "original"))
            if not ctrl:
                continue
            for level in PERT_ORDER:
                # Find the factor for this level.
                candidates = [k for k in majority if k[0] == model and k[1] == base_id and k[3] == level]
                if not candidates:
                    continue
                k = candidates[0]
                pert = majority[k]
                flipped = ctrl["label"] != pert["label"]
                c_int = LABEL_TO_INT[ctrl["label"]]
                p_int = LABEL_TO_INT[pert["label"]]
                maj_pairs.append(
                    {
                        "model": model,
                        "base_id": base_id,
                        "factor": k[2],
                        "level": level,
                        "human_off_rate": meta["off_rate"],
                        "human_votes": meta["votes"],
                        "control_label": ctrl["label"],
                        "perturbed_label": pert["label"],
                        "flipped": flipped,
                        "flip_type": "not_to_offensive" if c_int == 0 and p_int == 1 else "offensive_to_not" if c_int == 1 and p_int == 0 else "none",
                        "confidence_delta": pert["mean_confidence"] - ctrl["mean_confidence"],
                        "abs_confidence_delta": abs(pert["mean_confidence"] - ctrl["mean_confidence"]),
                        "delta_jsd": jsd_one_hot(pert["label"], meta["off_rate"]) - jsd_one_hot(ctrl["label"], meta["off_rate"]),
                    }
                )

    # Overall majority ASR.
    overall_rows = []
    for model in MODEL_ORDER:
        rows = [r for r in maj_pairs if r["model"] == model]
        k = sum(r["flipped"] for r in rows)
        n = len(rows)
        lo, hi = wilson_ci(k, n)
        dirs = Counter(r["flip_type"] for r in rows if r["flipped"])
        dom = "--"
        if k:
            if dirs["not_to_offensive"] >= dirs["offensive_to_not"]:
                dom = f"{100*dirs['not_to_offensive']/k:.1f}\\% not$\\rightarrow$off"
            else:
                dom = f"{100*dirs['offensive_to_not']/k:.1f}\\% off$\\rightarrow$not"
        overall_rows.append(
            {
                "model": model,
                "n": n,
                "flips": k,
                "asr": k / n,
                "ci_low": lo,
                "ci_high": hi,
                "mean_delta_jsd": mean(r["delta_jsd"] for r in rows),
                "dominant_direction": dom,
            }
        )

    # Raw ASR robustness.
    raw_rows = []
    for model in MODEL_ORDER:
        rows = [r for r in raw_pairs if r["display_model"] == model]
        k = sum(r["flipped"] for r in rows)
        n = len(rows)
        lo, hi = wilson_ci(k, n)
        raw_rows.append({"model": model, "n": n, "flips": k, "asr": k / n, "ci_low": lo, "ci_high": hi})

    # Perturbation ASR across models.
    pert_rows = []
    for level in PERT_ORDER:
        rows = [r for r in maj_pairs if r["level"] == level]
        k = sum(r["flipped"] for r in rows)
        n = len(rows)
        lo, hi = wilson_ci(k, n)
        pert_rows.append(
            {
                "level": level,
                "perturbation": PERT_LABEL[level],
                "n": n,
                "flips": k,
                "asr": k / n,
                "ci_low": lo,
                "ci_high": hi,
                "mean_delta_jsd": mean(r["delta_jsd"] for r in rows),
            }
        )

    # Model x perturbation ASR and McNemar tests.
    mxp_rows = []
    for model in MODEL_ORDER:
        for level in PERT_ORDER:
            rows = [r for r in maj_pairs if r["model"] == model and r["level"] == level]
            k = sum(r["flipped"] for r in rows)
            n = len(rows)
            lo, hi = wilson_ci(k, n)
            b = sum(1 for r in rows if r["control_label"] == "not_offensive" and r["perturbed_label"] == "offensive")
            c = sum(1 for r in rows if r["control_label"] == "offensive" and r["perturbed_label"] == "not_offensive")
            p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
            mxp_rows.append(
                {
                    "model": model,
                    "level": level,
                    "perturbation": PERT_LABEL[level],
                    "n": n,
                    "flips": k,
                    "asr": k / n if n else 0.0,
                    "ci_low": lo,
                    "ci_high": hi,
                    "not_to_off": b,
                    "off_to_not": c,
                    "mean_delta_jsd": mean(r["delta_jsd"] for r in rows) if rows else 0.0,
                    "mcnemar_p": p,
                }
            )
    qvals = bh_adjust([r["mcnemar_p"] for r in mxp_rows])
    for r, q in zip(mxp_rows, qvals):
        r["mcnemar_q_bh"] = q

    # Human agreement level analysis.
    agree_rows = []
    for rate in sorted({r["human_off_rate"] for r in maj_pairs}):
        rows = [r for r in maj_pairs if r["human_off_rate"] == rate]
        k = sum(r["flipped"] for r in rows)
        n = len(rows)
        lo, hi = wilson_ci(k, n)
        agree_rows.append({"human_off_rate": rate, "n": n, "flips": k, "asr": k / n, "ci_low": lo, "ci_high": hi})

    # Confidence instability by model.
    conf_rows = []
    for model in MODEL_ORDER:
        rows = [r for r in maj_pairs if r["model"] == model]
        signed = [r["confidence_delta"] for r in rows]
        absd = [r["abs_confidence_delta"] for r in rows]
        try:
            p = wilcoxon(signed, zero_method="zsplit").pvalue
        except ValueError:
            p = 1.0
        conf_rows.append(
            {
                "model": model,
                "n": len(rows),
                "mean_signed_delta": mean(signed),
                "mean_abs_delta": mean(absd),
                "wilcoxon_p": p,
            }
        )
    conf_q = bh_adjust([r["wilcoxon_p"] for r in conf_rows])
    for r, q in zip(conf_rows, conf_q):
        r["wilcoxon_q_bh"] = q

    # Per-item instability across all models and perturbations.
    item_rows = []
    for base_id, meta in sample.items():
        rows = [r for r in maj_pairs if r["base_id"] == base_id]
        counts_by_level = Counter(r["level"] for r in rows if r["flipped"])
        top = counts_by_level.most_common(1)[0][0] if counts_by_level else "none"
        item_rows.append(
            {
                "base_id": base_id,
                "human_votes": meta["votes"],
                "human_off_rate": meta["off_rate"],
                "n": len(rows),
                "flips": sum(r["flipped"] for r in rows),
                "asr": sum(r["flipped"] for r in rows) / len(rows),
                "top_perturbation": PERT_LABEL.get(top, top),
                "mean_abs_conf_delta": mean(r["abs_confidence_delta"] for r in rows),
            }
        )
    item_rows.sort(key=lambda r: (-r["flips"], r["base_id"]))

    # Leave-one-out overall range by model.
    loo_rows = []
    for model in MODEL_ORDER:
        rows = [r for r in maj_pairs if r["model"] == model]
        vals = []
        for base_id in sample:
            sub = [r for r in rows if r["base_id"] != base_id]
            vals.append(sum(r["flipped"] for r in sub) / len(sub))
        loo_rows.append({"model": model, "loo_min_asr": min(vals), "loo_max_asr": max(vals), "range": max(vals) - min(vals)})

    # Write CSVs.
    write_csv(OUT / "majority_overall_asr.csv", overall_rows)
    write_csv(OUT / "raw_overall_asr.csv", raw_rows)
    write_csv(OUT / "perturbation_asr.csv", pert_rows)
    write_csv(OUT / "model_perturbation_tests.csv", mxp_rows)
    write_csv(OUT / "agreement_asr.csv", agree_rows)
    write_csv(OUT / "self_consistency.csv", self_rows)
    write_csv(OUT / "confidence_instability.csv", conf_rows)
    write_csv(OUT / "per_item_instability.csv", item_rows)
    write_csv(OUT / "leave_one_out.csv", loo_rows)

    # LaTeX tables.
    with (OUT / "majority_overall_asr.tex").open("w") as f:
        f.write("\\begin{tabular}{lrrrrl}\n\\toprule\n")
        f.write("Model & $n$ & Flips & ASR & 95\\% CI & Direction \\\\\n\\midrule\n")
        for r in overall_rows:
            f.write(f"{r['model']} & {r['n']} & {r['flips']} & {fmt_pct(r['asr'])} & {fmt_ci(r['ci_low'], r['ci_high'])} & {r['dominant_direction']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with (OUT / "perturbation_asr.tex").open("w") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Perturbation & $n$ & Flips & ASR & 95\\% CI \\\\\n\\midrule\n")
        for r in sorted(pert_rows, key=lambda x: -x["asr"]):
            f.write(f"{r['perturbation']} & {r['n']} & {r['flips']} & {fmt_pct(r['asr'])} & {fmt_ci(r['ci_low'], r['ci_high'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with (OUT / "self_consistency.tex").open("w") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Model & Conditions & Unanimity & Control unanimity & Pairwise agreement \\\\\n\\midrule\n")
        for r in self_rows:
            f.write(f"{r['model']} & {r['all_conditions']} & {fmt_pct(r['unanimity_rate'])} & {fmt_pct(r['control_unanimity'])} & {fmt_pct(r['pairwise_agreement'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with (OUT / "agreement_asr.tex").open("w") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Human offensive rate & $n$ & Flips & ASR & 95\\% CI \\\\\n\\midrule\n")
        for r in agree_rows:
            f.write(f"{int(100*r['human_off_rate'])}\\% & {r['n']} & {r['flips']} & {fmt_pct(r['asr'])} & {fmt_ci(r['ci_low'], r['ci_high'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with (OUT / "confidence_instability.tex").open("w") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Model & $n$ & Mean $\\Delta c$ & Mean $|\\Delta c|$ & BH-adjusted $p$ \\\\\n\\midrule\n")
        for r in conf_rows:
            f.write(f"{r['model']} & {r['n']} & {r['mean_signed_delta']:.3f} & {r['mean_abs_delta']:.3f} & {r['wilcoxon_q_bh']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    top_tests = sorted(mxp_rows, key=lambda r: (-r["asr"], r["mcnemar_q_bh"]))[:10]
    with (OUT / "top_model_perturbation_tests.tex").open("w") as f:
        f.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
        f.write("Model & Perturbation & $n$ & ASR & 95\\% CI & $p$ & BH $q$ \\\\\n\\midrule\n")
        for r in top_tests:
            f.write(f"{r['model']} & {r['perturbation']} & {r['n']} & {fmt_pct(r['asr'])} & {fmt_ci(r['ci_low'], r['ci_high'])} & {r['mcnemar_p']:.3f} & {r['mcnemar_q_bh']:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with (OUT / "per_item_instability.tex").open("w") as f:
        f.write("\\begin{tabular}{llrrrl}\n\\toprule\n")
        f.write("Item ID & Human votes & $n$ & Flips & ASR & Most frequent flip perturbation \\\\\n\\midrule\n")
        for r in item_rows:
            f.write(f"\\texttt{{{r['base_id']}}} & {r['human_votes']} & {r['n']} & {r['flips']} & {fmt_pct(r['asr'])} & {tex_escape(r['top_perturbation'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # Figures.
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    xs = np.arange(len(overall_rows))
    vals = np.array([r["asr"] for r in overall_rows])
    lo = np.array([r["ci_low"] for r in overall_rows])
    hi = np.array([r["ci_high"] for r in overall_rows])
    ax.bar(xs, vals, color="#4C78A8")
    ax.errorbar(xs, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(xs, [r["model"] for r in overall_rows], rotation=25, ha="right")
    ax.set_ylabel("Majority-vote ASR")
    ax.set_ylim(0, max(hi) * 1.25)
    ax.set_title("Label instability by model with Wilson 95% intervals")
    fig.tight_layout()
    fig.savefig(FIG / "majority_asr_by_model.pdf")
    plt.close(fig)

    heat = np.zeros((len(MODEL_ORDER), len(PERT_ORDER)))
    for i, model in enumerate(MODEL_ORDER):
        for j, level in enumerate(PERT_ORDER):
            row = next(r for r in mxp_rows if r["model"] == model and r["level"] == level)
            heat[i, j] = row["asr"]
    fig, ax = plt.subplots(figsize=(8.5, 3.7))
    im = ax.imshow(heat, cmap="YlOrRd", vmin=0, vmax=max(0.35, heat.max()))
    ax.set_xticks(np.arange(len(PERT_ORDER)), [PERT_LABEL[p] for p in PERT_ORDER], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
    for i in range(len(MODEL_ORDER)):
        for j in range(len(PERT_ORDER)):
            ax.text(j, i, f"{100*heat[i,j]:.0f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ASR")
    ax.set_title("Majority-vote ASR heatmap: model $\\times$ perturbation")
    fig.tight_layout()
    fig.savefig(FIG / "model_perturbation_heatmap.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    xs = np.arange(len(agree_rows))
    vals = np.array([r["asr"] for r in agree_rows])
    lo = np.array([r["ci_low"] for r in agree_rows])
    hi = np.array([r["ci_high"] for r in agree_rows])
    ax.plot(xs, vals, marker="o", color="#F58518")
    ax.fill_between(xs, lo, hi, color="#F58518", alpha=0.2)
    ax.set_xticks(xs, [f"{int(100*r['human_off_rate'])}%" for r in agree_rows])
    ax.set_xlabel("Human offensive vote rate")
    ax.set_ylabel("Majority-vote ASR")
    ax.set_title("Instability by human agreement level")
    fig.tight_layout()
    fig.savefig(FIG / "asr_by_human_agreement.pdf")
    plt.close(fig)

    print(f"Wrote robustness artifacts to {OUT}")


if __name__ == "__main__":
    main()
