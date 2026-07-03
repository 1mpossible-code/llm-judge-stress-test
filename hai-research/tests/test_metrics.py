"""Tests for metrics computation"""

import numpy as np
import pytest
from src.metrics.compute import (
    compute_flip_rate,
    compute_jsd,
    compute_self_consistency,
    compute_label_distribution,
)
from src.schemas import Judgment


def test_compute_label_distribution():
    """Test label distribution computation."""
    judgments = [
        Judgment(
            run_id="test", model="m", backend="b", prompt_version="v1",
            base_id="b1", perturb_id="p1", factor="test", level="test", trial_idx=0,
            label="offensive", raw_output="", status="ok"
        ),
        Judgment(
            run_id="test", model="m", backend="b", prompt_version="v1",
            base_id="b1", perturb_id="p1", factor="test", level="test", trial_idx=1,
            label="offensive", raw_output="", status="ok"
        ),
        Judgment(
            run_id="test", model="m", backend="b", prompt_version="v1",
            base_id="b2", perturb_id="p1", factor="test", level="test", trial_idx=0,
            label="not_offensive", raw_output="", status="ok"
        ),
    ]
    
    dist = compute_label_distribution(judgments)
    
    assert abs(dist["offensive"] - 2/3) < 0.01
    assert abs(dist["not_offensive"] - 1/3) < 0.01


def test_compute_flip_rate():
    """Test flip rate computation."""
    control = [
        Judgment(
            run_id="test", model="m", backend="b", prompt_version="v1",
            base_id="b1", perturb_id="p_control", factor="control", level="original", trial_idx=0,
            label="offensive", raw_output="", status="ok"
        ),
        Judgment(
            run_id="test", model="m", backend="b", prompt_version="v1",
            base_id="b2", perturb_id="p_control", factor="control", level="original", trial_idx=0,
            label="not_offensive", raw_output="", status="ok"
        ),
    ]
    
    perturbed = [
        Judgment(
            run_id="test", model="m", backend="b", prompt_version="v1",
            base_id="b1", perturb_id="p_pert", factor="test", level="test", trial_idx=0,
            label="not_offensive", raw_output="", status="ok"  # Flipped
        ),
        Judgment(
            run_id="test", model="m", backend="b", prompt_version="v1",
            base_id="b2", perturb_id="p_pert", factor="test", level="test", trial_idx=0,
            label="not_offensive", raw_output="", status="ok"  # Same
        ),
    ]
    
    flip_rate = compute_flip_rate(control, perturbed)
    assert abs(flip_rate - 0.5) < 0.01  # One flip out of two


def test_compute_jsd():
    """Test Jensen-Shannon divergence computation."""
    model_dist = {"offensive": 0.8, "not_offensive": 0.2}
    human_dist = {"offensive": 0.5, "not_offensive": 0.5}
    
    jsd = compute_jsd(model_dist, human_dist)
    
    assert 0 <= jsd <= 1
    assert jsd > 0  # Should be non-zero for different distributions
    
    # Same distributions should have JSD = 0
    jsd_same = compute_jsd(model_dist, model_dist)
    assert abs(jsd_same) < 0.001


def test_compute_self_consistency():
    """Test self-consistency computation."""
    # All agree
    consistent = [
        Judgment(
            run_id="test", model="m", backend="b", prompt_version="v1",
            base_id="b1", perturb_id="p1", factor="test", level="test", trial_idx=0,
            label="offensive", raw_output="", status="ok"
        ),
        Judgment(
            run_id="test", model="m", backend="b", prompt_version="v1",
            base_id="b1", perturb_id="p1", factor="test", level="test", trial_idx=1,
            label="offensive", raw_output="", status="ok"
        ),
        Judgment(
            run_id="test", model="m", backend="b", prompt_version="v1",
            base_id="b1", perturb_id="p1", factor="test", level="test", trial_idx=2,
            label="offensive", raw_output="", status="ok"
        ),
    ]
    
    consistency = compute_self_consistency(consistent)
    assert abs(consistency - 1.0) < 0.01
    
    # Mixed
    mixed = consistent + [
        Judgment(
            run_id="test", model="m", backend="b", prompt_version="v1",
            base_id="b2", perturb_id="p1", factor="test", level="test", trial_idx=0,
            label="offensive", raw_output="", status="ok"
        ),
        Judgment(
            run_id="test", model="m", backend="b", prompt_version="v1",
            base_id="b2", perturb_id="p1", factor="test", level="test", trial_idx=1,
            label="not_offensive", raw_output="", status="ok"
        ),
    ]
    
    consistency_mixed = compute_self_consistency(mixed)
    assert 0 < consistency_mixed < 1

