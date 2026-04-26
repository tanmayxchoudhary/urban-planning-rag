## Area: Bootstrap, Toolchain, and Hygiene CLI

### VAL-CLI-001: `urban-rag` entrypoint is discoverable
Running `uv run urban-rag --help` from repo root must exit `0` and list command groups for ingestion/corpus plus a query-answer command surface. Pass when help output includes `ingest`, `corpus`, and one retrieval/generation command (`ask` or `query`); fail on non-zero exit or missing command groups.
Tool: Execute (`uv run urban-rag --help`)
Evidence:
- Exit code is `0`
- Help text contains `ingest`
- Help text contains `corpus`
- Help text contains `ask` or `query`

### VAL-CLI-002: Locked environment bootstrap works with `uv`
Running `uv sync --frozen` must succeed using lockfile-resolved dependencies. Pass when command exits `0` and project virtual environment is created; fail on dependency resolution drift or bootstrap failure.
Tool: Execute (`uv sync --frozen`)
Evidence:
- Exit code is `0`
- `.venv/bin/python` exists after command
- Output indicates frozen/lockfile sync completed

### VAL-CLI-003: Lint gate is runnable from CLI
Running `uv run ruff format --check && uv run ruff check` must pass with exit `0`. Fail if formatting/lint diagnostics remain.
Tool: Execute (`uv run ruff format --check && uv run ruff check`)
Evidence:
- Exit code is `0`
- No `error:` diagnostics in output

### VAL-CLI-004: Type-check gate is runnable from CLI
Running `uv run pyright` must pass with exit `0` for the configured strictness. Fail on any reported type error.
Tool: Execute (`uv run pyright`)
Evidence:
- Exit code is `0`
- Output reports `0 errors`

### VAL-CLI-005: Unit/integration test gate is runnable from CLI
Running `uv run pytest -q` must pass with exit `0`. Fail if any test fails, errors, or collection crashes.
Tool: Execute (`uv run pytest -q`)
Evidence:
- Exit code is `0`
- Pytest summary reports all tests passed

### VAL-CLI-006: Smoke eval runner is callable from CLI
Running `uv run python -m src.eval run --dataset smoke --tag <tag>` must produce a results directory and exit `0`. Fail if command crashes or emits incomplete run artifacts.
Tool: Execute (`uv run python -m src.eval run --dataset smoke --tag cli_smoke_contract`)
Evidence:
- Exit code is `0`
- Output includes run identifier/tag
- Result artifact directory for the tag exists

### VAL-CLI-007: Environment-secret hygiene is enforced
`git check-ignore .env` must return success and `.env.example` must exist with placeholder (non-live) values. Fail if `.env` is not ignored or `.env.example` is missing.
Tool: Execute (`git check-ignore .env && test -f .env.example`)
Evidence:
- Exit code is `0`
- `git check-ignore .env` prints `.env`
- `.env.example` file exists

### VAL-CLI-008: Known leaked key signatures are absent from git history scan
Running `git log -p -S '<known-key-fragment>'` for previously leaked signatures must return no matching patch content. Fail if historical matches remain undisclosed.
Tool: Execute (`git log -p -S 'AIzaSyAnrBWsA4'`)
Evidence:
- Exit code is `0`
- Output contains no matching key-bearing patch hunks

## Area: Ingestion and Corpus CLI

### VAL-CLI-009: Single-PDF ingest succeeds
`urban-rag ingest <pdf_path>` must ingest a valid PDF, exit `0`, and emit a stable document identifier/hash in output. Fail on crash or missing ingest artifact.
Tool: Execute (`uv run urban-rag ingest <valid_pdf_path>`)
Evidence:
- Exit code is `0`
- Output includes document hash/id
- Ingested document appears in `urban-rag corpus list`

### VAL-CLI-010: Directory ingest handles batch input
`urban-rag ingest <directory>` must process all valid PDFs in the directory and complete with a deterministic summary of ingested/skipped/failed files. Fail if valid PDFs are silently dropped.
Tool: Execute (`uv run urban-rag ingest <pdf_directory>`)
Evidence:
- Exit code is `0` (or documented partial code with explicit failures)
- Output includes total scanned and ingested counts
- Corpus totals increase by expected valid-PDF count

### VAL-CLI-011: Invalid ingest path is rejected
`urban-rag ingest <missing_path>` must fail fast with non-zero exit and a clear path-not-found error. Fail if command exits `0` or mutates corpus state.
Tool: Execute (`uv run urban-rag ingest <missing_path>`)
Evidence:
- Exit code is non-zero
- stderr/stdout contains path-not-found message
- `urban-rag corpus stats` unchanged from pre-run snapshot

### VAL-CLI-012: Non-PDF or malformed PDF input is rejected safely
`urban-rag ingest <non_pdf_or_corrupt_file>` must return non-zero and explain validation failure (`not a PDF`, parse failure, or encrypted/invalid). Fail if artifact is ingested anyway.
Tool: Execute (`uv run urban-rag ingest <invalid_file>`)
Evidence:
- Exit code is non-zero
- Error output explains document validation failure
- No new document id appears in corpus listing

### VAL-CLI-013: Re-ingesting the same PDF is idempotent
Running `urban-rag ingest <same_pdf_path>` twice must not duplicate corpus entries. Pass when second run reports no-op/already-ingested for same content hash; fail if doc/page totals grow unexpectedly.
Tool: Execute (`uv run urban-rag ingest <same_pdf_path>` twice + `uv run urban-rag corpus stats`)
Evidence:
- First run ingests document
- Second run reports existing content hash/no-op
- Doc/page totals unchanged after second run

### VAL-CLI-014: `--skip-eval` bypasses eval gate but still ingests
`urban-rag ingest <pdf_path> --skip-eval` must ingest successfully without running eval gate checks. Fail if ingest is blocked solely by unavailable eval path when skip flag is present.
Tool: Execute (`uv run urban-rag ingest <valid_pdf_path> --skip-eval`)
Evidence:
- Exit code is `0`
- Output indicates eval skipped
- Document appears in corpus list

### VAL-CLI-015: `corpus list` renders operator-usable inventory
`urban-rag corpus list` must exit `0` and print each indexed document with id/title and page metadata summary. Fail if output is empty while corpus has documents.
Tool: Execute (`uv run urban-rag corpus list`)
Evidence:
- Exit code is `0`
- Output contains at least one document id/hash
- Output includes title/filename and page count fields

### VAL-CLI-016: `corpus stats` totals are internally consistent
`urban-rag corpus stats` must return totals consistent with `corpus list` aggregation. Fail if totals mismatch document-level sums.
Tool: Execute (`uv run urban-rag corpus list` + `uv run urban-rag corpus stats`)
Evidence:
- Exit code is `0` for both commands
- `stats.documents` equals count of listed documents
- `stats.pages` equals sum of listed page counts

### VAL-CLI-017: `corpus rm` removes existing document
`urban-rag corpus rm <doc_hash>` must remove the targeted document and update corpus totals. Fail if target remains queryable/listed.
Tool: Execute (`uv run urban-rag corpus rm <existing_doc_hash>`)
Evidence:
- Exit code is `0`
- Removed hash absent from `urban-rag corpus list`
- Doc/page totals decrease accordingly

### VAL-CLI-018: `corpus rm` rejects unknown document id
`urban-rag corpus rm <unknown_hash>` must fail with non-zero exit and explicit not-found error, without mutating corpus totals.
Tool: Execute (`uv run urban-rag corpus rm <unknown_hash>`)
Evidence:
- Exit code is non-zero
- Error output contains `not found`/equivalent
- Corpus stats unchanged versus pre-run snapshot

### VAL-CLI-019: `corpus reindex` succeeds for existing document
`urban-rag corpus reindex <doc_hash>` must complete for an existing document and report completion status. Fail on silent partial update or missing completion signal.
Tool: Execute (`uv run urban-rag corpus reindex <existing_doc_hash>`)
Evidence:
- Exit code is `0`
- Output includes reindex start and completion for target hash
- Post-run retrieval on a known query still returns target document pages

### VAL-CLI-020: `corpus reindex` rejects unknown document id
`urban-rag corpus reindex <unknown_hash>` must fail fast with non-zero exit and no corpus mutation. Fail if command exits `0` or creates phantom records.
Tool: Execute (`uv run urban-rag corpus reindex <unknown_hash>`)
Evidence:
- Exit code is non-zero
- Error output contains `not found`/equivalent
- `corpus list` and `corpus stats` unchanged

## Area: Retrieval and Generation CLI (`urban-rag` query surface)

### VAL-CLI-021: Fast-mode ask returns grounded answer with citations
`urban-rag ask "<in-corpus-question>" --mode fast` must return an answer with inline citations and source listing. Pass when output contains citation markers and sources; fail if answer is uncited.
Tool: Execute (`uv run urban-rag ask "What is FAR for residential zones in URDPFI?" --mode fast`)
Evidence:
- Exit code is `0`
- Output contains citation tokens (e.g., `[1]`, `[2]`)
- Output contains `Sources` section

### VAL-CLI-022: Deep mode is accepted and completes end-to-end
`urban-rag ask "<in-corpus-question>" --mode deep` must execute successfully and produce a cited answer. Fail if mode is accepted syntactically but execution path is broken.
Tool: Execute (`uv run urban-rag ask "What are parking norms for commercial use?" --mode deep`)
Evidence:
- Exit code is `0`
- Output includes cited answer
- Run metadata/log indicates deep mode path selected

### VAL-CLI-023: Invalid mode value is rejected with validation error
`urban-rag ask "<question>" --mode turbo` must fail with non-zero exit and a schema/validation error. Fail if invalid mode silently falls back.
Tool: Execute (`uv run urban-rag ask "Test question" --mode turbo`)
Evidence:
- Exit code is non-zero
- Error output contains `validation_error` or invalid-choice message

### VAL-CLI-024: Question length bounds are enforced
CLI query surface must accept `1`-char questions and reject `>1000` chars. Fail if over-limit text is accepted without validation failure.
Tool: Execute (`uv run urban-rag ask "A" --mode fast` and `uv run urban-rag ask "<1001-char-text>" --mode fast`)
Evidence:
- 1-char request exits `0`
- 1001-char request exits non-zero (or returns `validation_error`)
- Over-limit error message includes length constraint

### VAL-CLI-025: Out-of-corpus question triggers explicit refusal path
`urban-rag ask "What is FAR for Mars colonies?" --mode fast` must refuse instead of hallucinating. Pass when refusal reason is explicit (`out_of_corpus`/equivalent) and no fabricated citations are emitted.
Tool: Execute (`uv run urban-rag ask "What is FAR for Mars colonies?" --mode fast`)
Evidence:
- Exit code indicates handled refusal (documented success/error path)
- Output includes explicit refusal reason/message
- Output does not include fake source citations

### VAL-CLI-026: Low-confidence retrieval path is surfaced to operator
For a weak/ambiguous in-corpus query, CLI output must expose low-confidence status and provide candidate pages instead of overconfident uncited claims. Fail if confidence state is hidden and answer is asserted as definitive.
Tool: Execute (`uv run urban-rag ask "<ambiguous_query>" --mode fast`)
Evidence:
- Output contains low-confidence indicator
- Candidate/source pages are displayed for operator review
- Final answer is hedged or refusal-based, not definitive without support

### VAL-CLI-027: Missing generator credentials fail with actionable error
With generator key unset/invalid, `urban-rag ask` must fail with explicit operator action (`set GEMINI_API_KEY` or equivalent). Fail on opaque stack trace-only failure.
Tool: Execute (`GEMINI_API_KEY= uv run urban-rag ask "What is FAR for residential zones?" --mode fast`)
Evidence:
- Exit code is non-zero
- Error output names missing/invalid credential
- Output includes actionable remediation text

### VAL-CLI-028: Retrieval backend outage is surfaced as service-unavailable
When embed/Qdrant backend is unreachable, CLI query command must surface a service-unavailable class error (`corpus_unavailable`/equivalent) with retry guidance. Fail on hang or silent empty response.
Tool: Execute (`uv run urban-rag ask "What is FAR for residential zones?" --mode fast` with backend intentionally unavailable)
Evidence:
- Exit code is non-zero (or documented degraded-mode return)
- Output contains backend-unavailable error code/message
- Retry guidance or retry-after information is shown
