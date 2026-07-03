"""Tests for paper table summarization."""

from pathlib import Path

from src.metrics.summarize import summarize_runs
from src.run.runner import run_experiment


def test_summarize_toy_run(tmp_path):
    config_content = Path("configs/exp_toy_mock.yaml").read_text()
    config_content = config_content.replace('runs_dir: "runs"', f'runs_dir: "{tmp_path}/runs"')
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)

    run_dir = run_experiment(config_path)

    from src.metrics.compute import compute_metrics_for_run

    compute_metrics_for_run(run_dir)
    outputs = summarize_runs([run_dir], tmp_path / "tables")

    assert outputs["overall_csv"].exists()
    assert outputs["overall_tex"].exists()
