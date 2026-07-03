"""Tests for output parser"""

import pytest
from src.judging.parser import parse_judgment


def test_parse_ideal_format():
    """Test parsing ideal format."""
    raw = "LABEL: offensive\nCONFIDENCE: 0.85\nRATIONALE: This text contains offensive language."
    judgment = parse_judgment(
        raw_output=raw,
        allowed_labels=["offensive", "not_offensive"],
        run_id="test_run",
        model="test_model",
        backend="mock",
        prompt_version="v1",
        base_id="base_001",
        perturb_id="pert_001",
        factor="test_factor",
        level="test_level",
        trial_idx=0,
    )
    
    assert judgment.status == "ok"
    assert judgment.label == "offensive"
    assert judgment.confidence == 0.85
    assert judgment.rationale == "This text contains offensive language."


def test_parse_with_percentage():
    """Test parsing confidence as percentage."""
    raw = "LABEL: not_offensive\nCONFIDENCE: 75%\nRATIONALE: Harmless text."
    judgment = parse_judgment(
        raw_output=raw,
        allowed_labels=["offensive", "not_offensive"],
        run_id="test_run",
        model="test_model",
        backend="mock",
        prompt_version="v1",
        base_id="base_001",
        perturb_id="pert_001",
        factor="test_factor",
        level="test_level",
        trial_idx=0,
    )
    
    assert judgment.status == "ok"
    assert judgment.label == "not_offensive"
    assert judgment.confidence == 0.75


def test_parse_case_insensitive_labels():
    """Test case-insensitive label matching."""
    raw = "LABEL: NOT_OFFENSIVE\nCONFIDENCE: 0.9\nRATIONALE: Fine."
    judgment = parse_judgment(
        raw_output=raw,
        allowed_labels=["offensive", "not_offensive"],
        run_id="test_run",
        model="test_model",
        backend="mock",
        prompt_version="v1",
        base_id="base_001",
        perturb_id="pert_001",
        factor="test_factor",
        level="test_level",
        trial_idx=0,
    )
    
    assert judgment.status == "ok"
    assert judgment.label == "not_offensive"


def test_parse_label_colon_fallback():
    """Test parsing provider output like 'not_offensive: 0.8'."""
    raw = "not_offensive: 0.8\nRATIONALE: Harmless text."
    judgment = parse_judgment(
        raw_output=raw,
        allowed_labels=["offensive", "not_offensive"],
        run_id="test_run",
        model="test_model",
        backend="mock",
        prompt_version="v1",
        base_id="base_001",
        perturb_id="pert_001",
        factor="test_factor",
        level="test_level",
        trial_idx=0,
    )

    assert judgment.status == "ok"
    assert judgment.label == "not_offensive"
    assert judgment.confidence == 0.8


def test_parse_missing_label():
    """Test parse error when label is missing."""
    raw = "CONFIDENCE: 0.8\nRATIONALE: Some text."
    judgment = parse_judgment(
        raw_output=raw,
        allowed_labels=["offensive", "not_offensive"],
        run_id="test_run",
        model="test_model",
        backend="mock",
        prompt_version="v1",
        base_id="base_001",
        perturb_id="pert_001",
        factor="test_factor",
        level="test_level",
        trial_idx=0,
    )
    
    assert judgment.status == "parse_error"
    assert judgment.label is None


def test_parse_missing_confidence():
    """Test parse error when confidence is missing."""
    raw = "LABEL: offensive\nRATIONALE: Some text."
    judgment = parse_judgment(
        raw_output=raw,
        allowed_labels=["offensive", "not_offensive"],
        run_id="test_run",
        model="test_model",
        backend="mock",
        prompt_version="v1",
        base_id="base_001",
        perturb_id="pert_001",
        factor="test_factor",
        level="test_level",
        trial_idx=0,
    )
    
    assert judgment.status == "parse_error"
    assert judgment.confidence is None


def test_parse_invalid_label():
    """Test parse error with invalid label."""
    raw = "LABEL: invalid_label\nCONFIDENCE: 0.5\nRATIONALE: Text."
    judgment = parse_judgment(
        raw_output=raw,
        allowed_labels=["offensive", "not_offensive"],
        run_id="test_run",
        model="test_model",
        backend="mock",
        prompt_version="v1",
        base_id="base_001",
        perturb_id="pert_001",
        factor="test_factor",
        level="test_level",
        trial_idx=0,
    )
    
    assert judgment.status == "parse_error"


def test_parse_confidence_clipping():
    """Test confidence clipping to [0, 1]."""
    raw = "LABEL: offensive\nCONFIDENCE: 1.5\nRATIONALE: Text."
    judgment = parse_judgment(
        raw_output=raw,
        allowed_labels=["offensive", "not_offensive"],
        run_id="test_run",
        model="test_model",
        backend="mock",
        prompt_version="v1",
        base_id="base_001",
        perturb_id="pert_001",
        factor="test_factor",
        level="test_level",
        trial_idx=0,
    )
    
    assert judgment.confidence == 1.0
    
    raw2 = "LABEL: offensive\nCONFIDENCE: -0.1\nRATIONALE: Text."
    judgment2 = parse_judgment(
        raw_output=raw2,
        allowed_labels=["offensive", "not_offensive"],
        run_id="test_run",
        model="test_model",
        backend="mock",
        prompt_version="v1",
        base_id="base_001",
        perturb_id="pert_001",
        trial_idx=0,
    )
    
    assert judgment2.confidence == 0.0


def test_parse_multiline_rationale():
    """Test parsing multi-line rationale."""
    raw = "LABEL: offensive\nCONFIDENCE: 0.7\nRATIONALE: This is a longer\nrationale that spans\nmultiple lines."
    judgment = parse_judgment(
        raw_output=raw,
        allowed_labels=["offensive", "not_offensive"],
        run_id="test_run",
        model="test_model",
        backend="mock",
        prompt_version="v1",
        base_id="base_001",
        perturb_id="pert_001",
        factor="test_factor",
        level="test_level",
        trial_idx=0,
    )
    
    assert judgment.status == "ok"
    assert "longer rationale" in judgment.rationale.lower()


def test_parse_messy_output():
    """Test parsing messy output with extra text."""
    raw = "Some preamble text.\nLABEL: not_offensive\nCONFIDENCE: 62%\nRATIONALE: This is fine.\nSome trailing text."
    judgment = parse_judgment(
        raw_output=raw,
        allowed_labels=["offensive", "not_offensive"],
        run_id="test_run",
        model="test_model",
        backend="mock",
        prompt_version="v1",
        base_id="base_001",
        perturb_id="pert_001",
        factor="test_factor",
        level="test_level",
        trial_idx=0,
    )
    
    assert judgment.status == "ok"
    assert judgment.label == "not_offensive"
    assert abs(judgment.confidence - 0.62) < 0.01

