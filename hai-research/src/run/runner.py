"""End-to-end experiment runner"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.config import ExperimentConfig, load_config
from src.dataset.load_base import load_base_dataset
from src.dataset.normalize import normalize_example
from src.perturb.generators import generate_all_perturbations
from src.registry import get_backend
from src.judging.parser import parse_judgment
from src.run.cache import make_cache_key, load_cache, save_cache
from src.schemas import JudgeRequest, Judgment
from src.utils.io import append_jsonl, read_json, write_json, read_jsonl


def make_run_id(experiment_name: str) -> str:
    """Generate deterministic run ID."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{timestamp}_{experiment_name}"


def load_prompt(prompt_file: str) -> str:
    """Load prompt template."""
    prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def format_prompt(
    template: str, 
    text: str, 
    allowed_labels: list[str], 
    language: str = "en",
) -> str:
    """Format prompt template with text and labels.
    
    Adds language-aware instructions for non-English text.
    """
    labels_str = ", ".join(allowed_labels)
    
    # Add language instruction if non-English
    language_instruction = ""
    if language != "en":
        language_instruction = f"\n\nNote: The text to classify is in {language}. Please evaluate it in its original language context, considering cultural and linguistic nuances appropriate to that language."
    
    try:
        formatted = template.format(text=text, allowed_labels=labels_str)
        if language_instruction:
            # Insert language instruction after the text field if present
            if "{text}" in template or "{allowed_labels}" in template:
                formatted += language_instruction
        return formatted
    except KeyError:
        # If template doesn't have placeholders, add text and labels anyway
        base_prompt = template
        if "{text}" not in template:
            base_prompt += f"\n\nText to classify:\n{text}"
        if "{allowed_labels}" not in template:
            base_prompt += f"\n\nAllowed labels: {labels_str}"
        return base_prompt + language_instruction


def compute_prompt_hash(prompt_template: str) -> str:
    """Compute SHA256 hash of prompt template."""
    return hashlib.sha256(prompt_template.encode()).hexdigest()


def run_experiment(config_path: str | Path) -> Path:
    """Run complete experiment pipeline.
    
    Returns:
        Path to run directory
    """
    # Load config
    config = load_config(config_path)
    
    # Create run directory
    run_id = make_run_id(config.experiment_name)
    runs_dir = Path(config.output.runs_dir)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and hash prompt
    prompt_template = load_prompt(config.judging.prompt_file)
    prompt_hash = compute_prompt_hash(prompt_template)
    
    # Prepare factor/level pairs
    factor_levels = []
    for factor in config.perturbations.factors:
        for level in factor.levels:
            factor_levels.append((factor.name, level))
    
    # Load base dataset
    base_examples = list(load_base_dataset(config.dataset.path, config.dataset.languages))
    
    # Apply limit if specified (for testing)
    original_count = len(base_examples)
    if config.dataset.limit:
        base_examples = base_examples[:config.dataset.limit]
        if len(base_examples) < original_count:
            print(f"Limiting dataset to {len(base_examples)} examples (requested limit={config.dataset.limit}, dataset has {original_count})")
        else:
            print(f"Using all {len(base_examples)} examples from dataset (requested limit={config.dataset.limit})")
    else:
        print(f"Using all {len(base_examples)} examples from dataset")
    
    normalized_examples = [
        normalize_example(ex, config.dataset.allowed_labels) for ex in base_examples
    ]
    
    # Create mapping from base_id to original text for logging
    base_id_to_text = {ex.id: ex.text for ex in base_examples}
    
    # Generate all perturbations
    all_perturbations = []
    for norm_ex in normalized_examples:
        perts = generate_all_perturbations(
            norm_ex.id, norm_ex.text, norm_ex.language, factor_levels
        )
        all_perturbations.extend(perts)
    
    # Initialize backends (cache by backend name)
    backends = {}
    for model_spec in config.judging.models:
        if model_spec.backend not in backends:
            backends[model_spec.backend] = get_backend(model_spec.backend)
    
    # Open judgment log file
    judgments_file = run_dir / "judgments.jsonl"
    
    # Main loop: models -> perturbations -> trials
    total_judgments = len(config.judging.models) * len(all_perturbations) * config.judging.repeats
    
    # Prepare all judgment tasks
    judgment_tasks = []
    for model_spec in config.judging.models:
        backend = backends[model_spec.backend]
        for perturb in all_perturbations:
            for trial_idx in range(config.judging.repeats):
                judgment_tasks.append((
                    model_spec, backend, perturb, trial_idx, run_id, run_dir,
                    prompt_template, prompt_hash, config, judgments_file
                ))
    
    # Process judgments with configurable parallelization.
    # Free-tier providers often need max_workers=1 to avoid rate limits.
    max_workers = min(config.judging.max_workers, len(judgment_tasks))
    
    def process_judgment(task):
        """Process a single judgment task."""
        (model_spec, backend, perturb, trial_idx, run_id, run_dir,
         prompt_template, prompt_hash, config, judgments_file) = task
        
        # Check cache
        cache_key = make_cache_key(
            model_spec,
            prompt_hash,
            perturb.perturb_id,
            trial_idx,
            config.judging.temperature,
            config.judging.max_tokens,
        )
        
        response = None
        if config.cache.enabled:
            response = load_cache(run_dir, cache_key)
        
        # Call backend if not cached. Convert backend exceptions into error
        # judgments so failed provider/dependency setup still leaves an auditable log.
        if response is None:
            formatted_prompt = format_prompt(
                prompt_template, 
                perturb.text, 
                config.dataset.allowed_labels,
                language=perturb.language,
            )
            
            req = JudgeRequest(
                text=perturb.text,
                allowed_labels=config.dataset.allowed_labels,
                prompt=formatted_prompt,
                temperature=config.judging.temperature,
                max_tokens=config.judging.max_tokens,
            )
            
            try:
                response = backend.judge(model_spec, req)
            except Exception as e:
                from src.schemas import JudgeResponse

                response = JudgeResponse(
                    raw_output="",
                    usage=None,
                    status="error",
                    error=f"{type(e).__name__}: {e}",
                )
            
            # Save to cache
            if config.cache.enabled:
                save_cache(run_dir, cache_key, response)
        
        # Parse judgment (unless backend returned an error)
        if response.status == "error":
            # Preserve backend error instead of parsing
            judgment = Judgment(
                run_id=run_id,
                model=model_spec.name,
                backend=model_spec.backend,
                prompt_version=config.judging.prompt_version,
                base_id=perturb.base_id,
                perturb_id=perturb.perturb_id,
                factor=perturb.factor,
                level=perturb.level,
                language=perturb.language,
                trial_idx=trial_idx,
                label=None,
                confidence=None,
                rationale=None,
                raw_output=response.raw_output,
                status="error",
                error=response.error or "Backend returned error status",
                usage=response.usage,
                input_text=base_id_to_text.get(perturb.base_id),
                prompt_hash=prompt_hash,
                perturbation_text_applied=perturb.text,
            )
        else:
            judgment = parse_judgment(
                raw_output=response.raw_output,
                allowed_labels=config.dataset.allowed_labels,
                run_id=run_id,
                model=model_spec.name,
                backend=model_spec.backend,
                prompt_version=config.judging.prompt_version,
                base_id=perturb.base_id,
                perturb_id=perturb.perturb_id,
                factor=perturb.factor,
                level=perturb.level,
                trial_idx=trial_idx,
                usage=response.usage,
                input_text=base_id_to_text.get(perturb.base_id),
                prompt_hash=prompt_hash,
                perturbation_text_applied=perturb.text,
            )
            # Add language field
            judgment_dict = judgment.to_dict()
            judgment_dict["language"] = perturb.language
            judgment = Judgment(**judgment_dict)
        
        return judgment.to_dict()
    
    # Execute with thread pool for I/O-bound API calls
    import threading
    write_lock = threading.Lock()
    
    with tqdm(total=total_judgments, desc="Judgments") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(process_judgment, task): task 
                for task in judgment_tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                try:
                    judgment_dict = future.result()
                    # Append to JSONL with lock for thread safety
                    with write_lock:
                        append_jsonl(judgments_file, judgment_dict)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing judgment: {e}")
                    pbar.update(1)
    
    # Collect perturbation metadata
    from src.perturb.registry import get_perturbation_info
    perturbation_metadata = []
    for factor in config.perturbations.factors:
        for level in factor.levels:
            info = get_perturbation_info(factor.name, level)
            perturbation_metadata.append({
                **info,
                "description": f"Perturbation factor {factor.name} at level {level}",
            })
    
    # Load dataset metadata if available
    dataset_meta = {}
    dataset_path = Path(config.dataset.path)
    if dataset_path.exists():
        meta_path = dataset_path.parent / f"{dataset_path.stem}_meta.json"
        if meta_path.exists():
            from src.utils.io import read_json
            dataset_meta = read_json(meta_path)
    
    # Save metadata
    meta = {
        "run_id": run_id,
        "config": config.model_dump(),
        "prompt_hash": prompt_hash,
        "package_versions": _get_package_versions(),
        "dataset": {
            "path": str(config.dataset.path),
            "allowed_labels": config.dataset.allowed_labels,
            "languages": config.dataset.languages,
            "task": dataset_meta.get("task", "unknown"),
            "label_mapping": dataset_meta.get("label_mapping", {}),
        },
        "perturbations": perturbation_metadata,
        "random_seed": config.perturbations.seed,
        "model_versions": {
            model_spec.name: {
                "backend": model_spec.backend,
                "params": model_spec.params,
            }
            for model_spec in config.judging.models
        },
    }
    write_json(run_dir / "meta.json", meta)
    
    return run_dir


def _get_package_versions() -> dict[str, str]:
    """Get versions of key packages."""
    versions = {}
    packages = ["pydantic", "numpy", "pandas", "matplotlib"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions

