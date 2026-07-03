"""LeWiDi/SemEval dataset ingestion"""

import json
import hashlib
from pathlib import Path
from typing import Any

from src.schemas import BaseExample
from src.utils.io import write_jsonl


def stable_id(dataset: str, task: str, text: str) -> str:
    """Generate stable SHA256 ID for example."""
    key = f"{dataset}:{task}:{text}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def extract_text_from_example(example: dict[str, Any], task: str) -> str:
    """Extract text field based on task type.
    
    Task-specific text extraction rules documented here.
    """
    # For offensive language detection tasks
    if task in ["offensiveness", "hate_speech", "toxic"]:
        # Usually in 'text' or 'sentence' field
        return example.get("text") or example.get("sentence") or example.get("comment", "")
    
    # For sentiment tasks
    if task in ["sentiment", "emotion"]:
        return example.get("text") or example.get("sentence") or example.get("tweet", "")
    
    # Default: try common field names
    return (
        example.get("text") or 
        example.get("sentence") or 
        example.get("comment") or 
        example.get("tweet") or
        example.get("content") or
        str(example)  # Last resort
    )


def extract_labels(example: dict[str, Any], task: str) -> list[str]:
    """Extract all human annotation labels from example."""
    labels = []
    
    # LeWiDi format: annotations field contains comma-separated string like "0,0,0,0,0"
    if "annotations" in example:
        annotations_val = example["annotations"]
        if isinstance(annotations_val, str):
            # Parse comma-separated string
            labels = [ann.strip() for ann in annotations_val.split(",") if ann.strip()]
        elif isinstance(annotations_val, list):
            labels = [str(ann) for ann in annotations_val]
        else:
            labels.append(str(annotations_val))
    
    # Fallback to other formats
    if not labels and "labels" in example:
        if isinstance(example["labels"], list):
            labels.extend([str(l) for l in example["labels"]])
        else:
            labels.append(str(example["labels"]))
    
    if not labels and "label" in example:
        labels.append(str(example["label"]))
    
    # Try per-annotator fields (common in LeWiDi)
    if not labels:
        for key in example.keys():
            if key.startswith("annotator_") or key.startswith("rater_"):
                label_val = example[key]
                if label_val:
                    labels.append(str(label_val))
    
    return labels


def ingest_lewidi(
    input_dir: Path,
    task: str,
    out_path: Path,
    splits: list[str] | None = None,
    limit: int | None = None,
    label_mapping: dict[str, str] | None = None,
    allowed_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Ingest LeWiDi/SemEval dataset.
    
    Args:
        input_dir: Directory containing dataset JSON files
        task: Task name (e.g., "offensiveness", "sentiment")
        out_path: Output JSONL file path
        splits: List of splits to ingest (e.g., ["train", "dev"])
        limit: Optional limit on number of examples per split
        label_mapping: Dict mapping raw labels to normalized labels
        allowed_labels: List of allowed normalized labels (for validation)
    
    Returns:
        Metadata dict with ingestion stats and label mapping info
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if splits is None:
        splits = ["train", "dev", "test"]
    
    examples = []
    all_raw_labels = set()
    label_mapping_used = label_mapping or {}
    stats = {"total_examples": 0, "skipped": 0, "by_split": {}}
    
    # Find dataset JSON files (exclude metadata files)
    json_files = [f for f in input_dir.glob("*.json") if not f.name.endswith("_meta.json")]
    
    if not json_files:
        raise ValueError(f"No JSON files found in {input_dir}")
    
    print(f"Found {len(json_files)} JSON files: {[f.name for f in json_files]}")
    
    # Load all JSON files
    # LeWiDi format: files like train.json, dev.json contain dicts where keys are example IDs
    all_examples_by_file = {}  # filename -> {example_id -> example}
    
    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle LeWiDi format: dict where keys are example IDs, values are examples
                if isinstance(data, dict) and len(data) > 0:
                    first_val = next(iter(data.values()))
                    if isinstance(first_val, dict) and "text" in first_val:
                        # This is LeWiDi format: dict of examples
                        all_examples_by_file[json_file.stem] = data
                        print(f"Loaded {len(data)} examples from {json_file.name}")
                    else:
                        print(f"Warning: {json_file} doesn't appear to be in LeWiDi format")
                else:
                    print(f"Warning: {json_file} is not a dict or is empty")
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue
    
    # Process examples by split
    for split in splits:
        split_examples_list = []  # List of (ex_id, example) tuples
        
        # Look for examples in files that match split name OR examples with matching split field
        for filename, examples_dict in all_examples_by_file.items():
            # Check if filename matches split (e.g., "train" in "train.json")
            filename_matches = split.lower() in filename.lower()
            
            for ex_id, example in examples_dict.items():
                example_split_val = example.get("split")
                example_split = str(example_split_val).lower() if example_split_val is not None else ""
                
                # Include if filename matches OR split field matches
                if filename_matches or example_split == split.lower():
                    split_examples_list.append((ex_id, example))
        
        print(f"Split '{split}': found {len(split_examples_list)} candidate examples (filename match or split field match)")
        
        if not split_examples_list:
            print(f"Warning: No examples found for split '{split}'")
            continue
        
        processed_count = 0
        split_examples_list_processed = []  # Initialize list for processed examples
        
        for ex_id, example in split_examples_list:
            if limit and processed_count >= limit:
                break
            
            # Extract text
            text = extract_text_from_example(example, task)
            if not text or not text.strip():
                stats["skipped"] += 1
                continue
            
            # Extract labels
            raw_labels = extract_labels(example, task)
            
            if len(raw_labels) < 2:
                stats["skipped"] += 1
                continue
            
            # Map labels
            if label_mapping_used:
                normalized_labels = [
                    label_mapping_used.get(l, l) for l in raw_labels
                ]
            else:
                normalized_labels = raw_labels
            
            # Validate labels
            if allowed_labels:
                normalized_labels = [l for l in normalized_labels if l in allowed_labels]
                if not normalized_labels:
                    stats["skipped"] += 1
                    continue
            
            # Track raw labels
            all_raw_labels.update(raw_labels)
            
            # Generate ID
            ex_id = stable_id("lewidi", task, text)
            
            # Extract language
            language = example.get("language") or example.get("lang") or "en"
            
            # Extract annotator IDs (LeWiDi format: comma-separated string)
            annotator_ids = []
            if "annotators" in example:
                annotator_str = example["annotators"]
                if isinstance(annotator_str, str):
                    annotator_ids = [a.strip() for a in annotator_str.split(",") if a.strip()]
                elif isinstance(annotator_str, list):
                    annotator_ids = annotator_str
            
            # Extract metadata
            metadata = {
                "dataset": "lewidi",
                "task": task,
                "split": split,
                "raw_labels": raw_labels,
                "annotator_ids": annotator_ids,
                "soft_label": example.get("soft_label"),
                "hard_label": example.get("hard_label"),
                "annotation_task": example.get("annotation task"),
                "number_of_annotations": example.get("number of annotations"),
                **{k: v for k, v in example.items() 
                   if k not in ["text", "sentence", "comment", "labels", "label", "annotations", "language", "lang", 
                               "annotators", "hard_label", "soft_label", "annotation task", "number of annotations", "split"]}
            }
            
            # Create example
            base_example = BaseExample(
                id=ex_id,
                language=language,
                text=text,
                human_labels=normalized_labels,
            )
            
            # Store metadata separately (not in BaseExample schema, but we'll extend it)
            example_dict = base_example.model_dump()
            example_dict["metadata"] = metadata
            
            split_examples_list_processed.append(example_dict)
            examples.append(example_dict)
            processed_count += 1
        
        stats["by_split"][split] = len(split_examples_list_processed)
        stats["total_examples"] += len(split_examples_list_processed)
    
    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, examples)
    
    # Build label mapping if not provided
    if not label_mapping_used and all_raw_labels:
        # Create identity mapping by default
        label_mapping_used = {label: label for label in all_raw_labels}
    
    metadata_result = {
        "ingestion_stats": stats,
        "label_mapping": label_mapping_used,
        "raw_labels_seen": sorted(list(all_raw_labels)),
        "normalized_labels_seen": sorted(list(set(
            label_mapping_used.get(l, l) for l in all_raw_labels
        ))),
        "task": task,
        "dataset": "lewidi",
    }
    
    return metadata_result

