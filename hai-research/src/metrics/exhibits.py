"""Generate exhibits.jsonl with top N flips for analysis"""

from pathlib import Path
from typing import Any
import numpy as np

from src.utils.io import read_jsonl, write_jsonl


def generate_exhibits(
    run_dir: Path,
    top_n: int = 20,
    sort_by: str = "delta_jsd",  # "delta_jsd" or "confidence_swing"
) -> None:
    """Generate exhibits.jsonl with top N flips.
    
    Args:
        run_dir: Path to run directory
        top_n: Number of top examples to include
        sort_by: Sort by "delta_jsd" or "confidence_swing"
    """
    paired_results_file = run_dir / "paired_results.jsonl"
    
    if not paired_results_file.exists():
        print(f"Warning: paired_results.jsonl not found. Run generate_paired_results first.")
        return
    
    # Load paired results
    paired_results = list(read_jsonl(paired_results_file))
    
    if not paired_results:
        print("No paired results found.")
        return
    
    # Filter to only flipped examples
    flipped_results = [r for r in paired_results if r.get("flipped", False)]
    
    if not flipped_results:
        print("No flipped examples found.")
        return
    
    # Sort by specified metric
    if sort_by == "delta_jsd":
        # Sort by absolute delta_jsd (descending)
        flipped_results.sort(
            key=lambda x: abs(x.get("delta_jsd", 0.0)) if x.get("delta_jsd") is not None else 0.0,
            reverse=True
        )
    elif sort_by == "confidence_swing":
        # Sort by absolute confidence_swing (descending)
        flipped_results.sort(
            key=lambda x: abs(x.get("confidence_swing", 0.0)) if x.get("confidence_swing") is not None else 0.0,
            reverse=True
        )
    else:
        raise ValueError(f"Invalid sort_by: {sort_by}. Must be 'delta_jsd' or 'confidence_swing'")
    
    # Take top N
    top_results = flipped_results[:top_n]
    
    # Add metadata
    exhibits = []
    for i, result in enumerate(top_results, 1):
        exhibit = {
            "rank": i,
            "sort_metric": sort_by,
            "sort_value": (
                abs(result.get("delta_jsd", 0.0)) if sort_by == "delta_jsd"
                else abs(result.get("confidence_swing", 0.0))
            ),
            **result,  # Include all paired result fields
        }
        exhibits.append(exhibit)
    
    # Save exhibits with sort metric in filename
    output_file = run_dir / f"exhibits_{sort_by}.jsonl"
    write_jsonl(output_file, exhibits)
    
    print(f"Generated {len(exhibits)} exhibits in {output_file} (sorted by {sort_by})")

