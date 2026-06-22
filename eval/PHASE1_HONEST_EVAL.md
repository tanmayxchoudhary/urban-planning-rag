# Phase 1 Honest-Eval Groundwork for Urban RAG v1

## Changes Made
- Removed/deprecated _synthetic_candidates_for_entry (the liar)
- Added _smoke_mock_candidates_for_entry with honest warnings
- Added _detect_synthetic_placeholders for loud failure
- Made _compute_retrieval_metrics_for_entry retrieval-pluggable (accepts provided_candidates or use_smoke_mock)
- Updated run_smoke_eval with use_smoke_mock and provided_candidates_map params
- CLI supports --smoke-mock flag
- Updated tests/eval/test_smoke_gates.py to honest naming (smoke_mock_*)
- Updated eval/ __init__.py and metrics __init__.py
- Added this dataset note file

## Honest Behavior
- Without --smoke-mock or real candidates: raises HONEST-EVAL FAILURE (no pretending)
- With --smoke-mock: logs "eval_smoke_mock_mode_active - recall metrics are from MOCK, not real retrieval"
- Real mode: expects live retrieval candidates

## Tests
- pytest tests/eval/test_smoke_gates.py : all pass
- CLI smoke without flag: fails loudly as designed
- CLI with --smoke-mock: succeeds with honest label

## Files Changed (eval scope only)
- src/urban_rag/eval/__main__.py
- src/urban_rag/eval/__init__.py
- src/urban_rag/eval/metrics/__init__.py
- tests/eval/test_smoke_gates.py
- eval/PHASE1_HONEST_EVAL.md (new)

## Scope Truth
This is **Phase 1 groundwork only**: the code path no longer pretends synthetic recall is real.
It does **not** satisfy the PLAN.md Phase 1 gate by itself. The plan's real Phase 1 exit
criteria still need live measurement and receipts for retrieval quality and behavior, including:
- recall@5 ≥ 80%
- out-of-domain refusal ≥ 90%
- citation faithfulness ≥ 85%

Until those are measured from real retrieval/generation runs, this file should not be read as
"Phase 1 complete" — only as "the eval harness now fails honestly instead of lying."
