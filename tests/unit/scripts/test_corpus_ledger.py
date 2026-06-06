"""Tests for the Urban RAG corpus ledger generator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "corpus_ledger.py"
spec = importlib.util.spec_from_file_location("corpus_ledger", SCRIPT_PATH)
assert spec is not None
corpus_ledger = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(corpus_ledger)


def test_build_ledger_reconciles_legacy_visual_corpus() -> None:
    ledger = corpus_ledger.build_ledger()

    assert ledger["active_corpus_version"] == "legacy-visual-738"
    assert ledger["summary"]["page_images_total"] == 743
    assert ledger["summary"]["legacy_embedding_rows"] == 738
    assert ledger["summary"]["legacy_active_rows"] == 738

    rows = {row["prefix"]: row for row in ledger["active_legacy_rows"]}
    assert rows["urdpfi_vol1"]["page_images"] == 447
    assert rows["urdpfi_vol1"]["embedding_rows"] == 447
    assert rows["urdpfi_vol2"]["page_images"] == 250
    assert rows["urdpfi_vol2"]["embedding_rows"] == 250
    assert rows["swm_2016"]["page_images"] == 41
    assert rows["swm_2016"]["embedding_rows"] == 41
    assert {row["status"] for row in rows.values()} == {"active_legacy"}


def test_build_ledger_exposes_current_pipeline_mismatch() -> None:
    ledger = corpus_ledger.build_ledger()

    assert ledger["summary"]["current_docs_dirs"] == 8
    assert ledger["summary"]["current_docs_pages_jsonl_total"] == 322
    assert ledger["summary"]["orphan_image_prefixes"] == [
        "2671af3e4e26f088c42e2294185c1a5cafd56c59e825da427c79d4857bf8dda4"
    ]
