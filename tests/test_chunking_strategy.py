from scripts.data_prep.build_parent_child_chunks import split_parent_structured
from scripts.experiment.analyze_chunking_profile import (
    derive_candidates,
    derive_hybrid_candidates,
    derive_refinement_candidates,
    distribution,
)
from scripts.experiment.evaluate_parent_child import load_excluded_qids
from scripts.experiment.evaluate_chunking_candidate_answers import validate_evaluation_scope


def test_candidate_answer_evaluator_enforces_sealed_scope():
    holdout = {"A002", "A057"}
    assert validate_evaluation_scope(holdout, holdout, "holdout") == ["A002", "A057"]

    import pytest

    with pytest.raises(RuntimeError, match="exactly match"):
        validate_evaluation_scope({"A002"}, holdout, "holdout")
    with pytest.raises(RuntimeError, match="overlaps"):
        validate_evaluation_scope({"A002", "A003"}, holdout, "development")


class CharacterTokenizer:
    """Small deterministic tokenizer used to test span logic without model files."""

    def __call__(self, text, **_kwargs):
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


def test_structured_split_prefers_line_boundaries_and_covers_parent():
    text = "第一段说明。\n第二段包含一个更长的解释。\n第三段给出结论。"
    chunks = split_parent_structured(
        CharacterTokenizer(), text, target_tokens=18, min_tokens=8,
        max_tokens=24, overlap_max_tokens=5, overlap_ratio_cap=0.25,
    )
    assert chunks
    assert all(chunk["token_count"] <= 24 for chunk in chunks)
    covered = [False] * len(text)
    for chunk in chunks:
        start, end = chunk["local_char_start"], chunk["local_char_end"]
        assert chunk["text"] == text[start:end]
        for index in range(start, end):
            covered[index] = True
    assert all(hit or char.isspace() for hit, char in zip(covered, text))


def test_structured_split_uses_token_windows_only_as_safety_fallback():
    text = "A" * 65
    chunks = split_parent_structured(
        CharacterTokenizer(), text, target_tokens=20, min_tokens=12,
        max_tokens=25, overlap_max_tokens=5, overlap_ratio_cap=0.25,
    )
    assert len(chunks) == 3
    assert max(chunk["token_count"] for chunk in chunks) <= 25
    assert all(chunk["start_boundary"] == "token-window" for chunk in chunks)


def test_dynamic_overlap_never_exceeds_ratio_cap():
    text = "12345\n67890\nabcde\nfghij\nklmno"
    chunks = split_parent_structured(
        CharacterTokenizer(), text, target_tokens=12, min_tokens=6,
        max_tokens=15, overlap_max_tokens=10, overlap_ratio_cap=0.25,
    )
    for previous, current in zip(chunks, chunks[1:]):
        overlap = max(
            0,
            min(previous["local_char_end"], current["local_char_end"])
            - max(previous["local_char_start"], current["local_char_start"]),
        )
        assert overlap <= 3


def test_profile_candidates_are_derived_from_observed_quantiles():
    profile = {
        "parent_tokens": distribution([300, 500, 700, 900]),
        "paragraph_tokens": distribution([40, 80, 120, 160]),
        "sentence_tokens": distribution([20, 30, 40, 50]),
        "evidence_tokens": distribution([100, 160, 220, 300]),
        "embedding_capacity": {
            "model_max_seq_length": 128,
            "special_tokens": 2,
            "raw_body_tokens": 126,
            "shared_safe_body_tokens": 75,
        },
    }
    candidates = derive_candidates(profile)
    targets = {item["target_tokens"] for item in candidates}
    assert 75 in targets
    assert max(targets) <= profile["embedding_capacity"]["shared_safe_body_tokens"]
    assert all(item["min_tokens"] <= item["target_tokens"] <= item["max_tokens"] for item in candidates)
    assert all(item["overlap_max_tokens"] < item["target_tokens"] for item in candidates)


def test_load_excluded_qids_supports_holdout_manifest(tmp_path):
    manifest = tmp_path / "holdout.json"
    manifest.write_text('{"qids": ["A002", "A057"]}', encoding="utf-8")

    assert load_excluded_qids(manifest) == {"A002", "A057"}


def test_refinement_candidates_use_raw_capacity_and_structural_quantiles():
    profile = {
        "paragraph_tokens": distribution([20, 30, 40, 50]),
        "sentence_tokens": distribution([10, 20, 30, 45]),
        "embedding_capacity": {"raw_body_tokens": 126},
    }

    candidates = derive_refinement_candidates(profile)
    targets = {item["target_tokens"] for item in candidates}
    assert targets == {80, 95, 125}
    assert {item["overlap_max_tokens"] for item in candidates} == {0, 30}
    assert all(item["max_tokens"] <= 126 for item in candidates)


def test_hybrid_candidates_use_soft_system_bounds():
    profile = {
        "paragraph_tokens": distribution([20, 30, 40, 50]),
        "sentence_tokens": distribution([10, 20, 30, 45]),
        "embedding_capacity": {"raw_body_tokens": 126},
    }

    candidates = derive_hybrid_candidates(profile)
    assert {item["target_tokens"] for item in candidates} == {165, 205, 250}
    assert {item["overlap_max_tokens"] for item in candidates} == {30, 50}
    assert max(item["target_tokens"] for item in candidates) <= 2 * 126
