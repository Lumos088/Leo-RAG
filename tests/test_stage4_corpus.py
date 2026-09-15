import csv
import hashlib
import json
from pathlib import Path

from scripts.pipeline.corpus_files import discover_chunk_files
from scripts.pipeline.query_router import detect_subject


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = json.loads((ROOT / "config/parent_child.json").read_text(encoding="utf-8"))
STAGE4 = (ROOT / PRODUCTION["parent_dir"]).parent
BUILD_INFO = json.loads((STAGE4 / "config.json").read_text(encoding="utf-8"))
LEGACY = ROOT / BUILD_INFO["legacy_source_dir"]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage4_preserves_every_legacy_file_and_id():
    for side in ("parents", "children"):
        for filename in ("ds_chunks.json", "os_chunks.json", "cn_chunks.json"):
            assert digest(STAGE4 / side / filename) == digest(LEGACY / side / filename)
        assert discover_chunk_files(STAGE4 / side) == (
            "ds_chunks.json", "os_chunks.json", "cn_chunks.json", "ai_chunks.json"
        )


def test_ai_parent_child_integrity_and_metadata():
    parents = load(STAGE4 / "parents/ai_chunks.json")
    children = load(STAGE4 / "children/ai_chunks.json")
    parent_ids = {item["id"] for item in parents}
    assert len(parents) == len(parent_ids) == 119
    assert len(children) == len({item["id"] for item in children}) == BUILD_INFO["ai_child_count"]
    assert all(item["parent_id"] in parent_ids for item in children)
    assert {item["parent_id"] for item in children} == parent_ids
    required = {
        "source_url", "source_revision", "source_license", "source_license_url",
        "source_slug", "acquired_at", "topics", "chapter_path"
    }
    assert all(required <= item.keys() for item in parents)


def test_ai_eval_has_25_judged_questions():
    with (ROOT / "data/stage4/evaluation/topics_ai.csv").open(encoding="utf-8-sig", newline="") as handle:
        topics = list(csv.DictReader(handle))
    with (ROOT / "data/stage4/evaluation/qrels_ai.csv").open(encoding="utf-8-sig", newline="") as handle:
        qrels = list(csv.DictReader(handle))
    assert len(topics) == 25
    assert {row["qid"] for row in topics} == {row["qid"] for row in qrels}
    assert all(int(row["rel"]) > 0 for row in qrels)


def test_source_registry_matches_downloaded_files():
    registry = load(ROOT / "data/stage4/sources/SOURCE_REGISTRY.json")
    assert len(registry) == 21
    for item in registry:
        raw_path = ROOT / item["raw_file"]
        assert raw_path.is_file()
        assert digest(raw_path) == item["raw_sha256"]
        assert item["source_revision"]
        assert item["source_license"] in {"CC-BY-SA-4.0", "Apache-2.0", "MIT"}
        assert item["source_license_url"].startswith("https://")
        license_path = ROOT / item["source_license_file"]
        assert license_path.is_file()
        assert digest(license_path) == item["source_license_sha256"]


def test_production_config_points_to_verified_stage4_indexes():
    parent_child = load(ROOT / "config/parent_child.json")
    retrieval = load(ROOT / "config/retrieval.json")
    assert parent_child["child_index_dir"].startswith("vector_db/stage4/")
    assert parent_child["parent_dir"].startswith("data/stage4/")
    assert parent_child["parent_top_k"] == 6
    assert "struct205_o50_ai70_o15" in parent_child["variant"]
    assert retrieval["index_dir"].startswith("vector_db/stage4/")
    for relative in (parent_child["child_index_dir"], retrieval["index_dir"]):
        directory = ROOT / relative
        assert (directory / "kb.index").is_file()
        assert (directory / "kb_meta.json").is_file()
        config = load(directory / "kb_config.json")
        metadata = load(directory / "kb_meta.json")
        assert config["ntotal"] == len(metadata)
        chunks_dir = ROOT / config["chunks_dir"]
        assert tuple(config["chunk_files"]) == discover_chunk_files(chunks_dir)
        assert config["chunk_file_sha256"] == {
            filename: digest(chunks_dir / filename)
            for filename in config["chunk_files"]
        }


def test_ai_queries_route_to_ai_without_changing_unknown_queries():
    assert detect_subject("Transformer 为什么需要位置编码？") == "Artificial Intelligence"
    assert detect_subject("LoRA 如何减少训练参数？") == "Artificial Intelligence"
    assert detect_subject("什么是死锁？") is None
