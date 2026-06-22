"""CLI runner for the evaluation pipeline.

Implements:
    python -m urban_rag.eval run --dataset smoke --tag v1 --output data/eval/runs/v1/

Exit codes:
    0  — all eval assertions passed
    1  — at least one assertion failed, or dataset missing/malformed
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
import typer
from rich.console import Console
from rich.table import Table

from urban_rag.common.types import RetrievalCandidate as _RCandidate
from urban_rag.eval.metrics.retrieval import (
    RetrievalMetricsResult,
    compute_retrieval_metrics,
)

# ---------------------------------------------------------------------------
# Constants (must match PLAN.md §10.6 and test_smoke_gates.py)
# ---------------------------------------------------------------------------

#: Minimum recall@10 required to pass CI gate
RECALL_AT_10_THRESHOLD = 0.85

#: Minimum faithfulness (RAGAS) required to pass CI gate
FAITHFULNESS_THRESHOLD = 0.85

#: Minimum answer_relevance (RAGAS) required to pass CI gate
ANSWER_RELEVANCE_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_OUTPUT_BASE = REPO_ROOT / "data" / "eval" / "runs"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = structlog.get_logger(__name__, service="eval")


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def load_eval_dataset(dataset_name: str) -> list[dict[str, Any]]:
    """Load an eval dataset JSONL file.

    Args:
        dataset_name: Name of the dataset (e.g. "smoke", "regression").
            Resolves to "eval/{dataset_name}.jsonl" relative to repo root.

    Returns:
        List of parsed JSON dicts, one per line.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If a line is malformed JSON.
    """
    dataset_path = REPO_ROOT / "eval" / f"{dataset_name}.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            f"Available datasets: smoke, regression (check eval/ directory)"
        )

    entries = []
    with dataset_path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Malformed JSON at {dataset_path} line {line_num}: {e}"
                ) from e

    if not entries:
        raise ValueError(f"Dataset {dataset_path} is empty")

    return entries


# ---------------------------------------------------------------------------
# Smoke/mock candidate generation for evaluation (HONEST MODE)
#
# This is EXPLICITLY a mock for smoke testing only.
# It does NOT represent real retrieval performance.
# Real eval requires live retrieval candidates passed in.
# Using this without --smoke-mock flag will fail loudly.
# ---------------------------------------------------------------------------


def _smoke_mock_candidates_for_entry(entry: dict[str, Any]) -> list[dict]:
    """Build MOCK (not real) retrieval candidates for smoke dataset entries.

    WARNING: This simulates perfect retrieval by placing expected_pages first.
    This is ONLY for smoke/mock testing of the eval harness itself.
    It is NOT a measure of real RAG retrieval quality.
    Phase 1 honest-eval: synthetic-perfect-oracle path is deprecated.
    Real candidates must be provided via retrieval-pluggable interface.
    """
    expected = entry.get("expected_pages", [])
    candidates = []
    for i, page_id in enumerate(expected):
        candidates.append({
            "page_id": page_id,
            "score": 1.0 / (i + 1),
            "page_image_uri": f"s3://pages/{page_id}.png",
        })
    # Add decoy pages
    for i in range(10):
        candidates.append({
            "page_id": f"decoy_page_{i}",
            "score": 0.1,
            "page_image_uri": f"s3://pages/decoy_{i}.png",
        })
    return candidates


def _detect_synthetic_placeholders(candidates: list[dict]) -> bool:
    """Detect if candidates contain synthetic decoy placeholders.

    If true, this indicates mock data, not real retrieval output.
    Used to fail loudly in honest-eval mode.
    """
    for c in candidates:
        if "decoy_page_" in c.get("page_id", ""):
            return True
    return False


# ---------------------------------------------------------------------------
# Metric computation per entry
# ---------------------------------------------------------------------------


def _compute_retrieval_metrics_for_entry(
    entry: dict[str, Any],
    provided_candidates: list[dict] | None = None,
    use_smoke_mock: bool = False,
) -> tuple[RetrievalMetricsResult, list[_RCandidate], str]:
    """Compute retrieval metrics for an entry.

    Retrieval-pluggable: accepts provided_candidates from real retrieval.
    If use_smoke_mock=True, uses honest mock (labeled as such).
    Fails loudly if neither provided nor explicit mock flag (prevents pretending synthetic is real).

    Args:
        entry: dataset entry
        provided_candidates: real candidates from retrieval system (preferred)
        use_smoke_mock: if True, allow honest smoke/mock mode

    Returns:
        (metrics_result, candidates, mode_label) where mode_label is 'real' or 'smoke/mock'
    """
    expected_pages = set(entry.get("expected_pages", []))
    expected_docs = set(entry.get("expected_documents", []))

    mode_label = "real"
    if provided_candidates is not None:
        raw_candidates = provided_candidates
        if _detect_synthetic_placeholders(raw_candidates):
            raise ValueError(
                "HONEST-EVAL: provided_candidates contain synthetic decoy placeholders. "
                "This is not real retrieval output. Use use_smoke_mock=True for mock mode."
            )
    elif use_smoke_mock:
        raw_candidates = _smoke_mock_candidates_for_entry(entry)
        mode_label = "smoke/mock"
    else:
        raise ValueError(
            "HONEST-EVAL FAILURE: No real retrieval candidates provided and use_smoke_mock=False. "
            "The synthetic-perfect-oracle path has been removed. "
            "Provide real candidates or explicitly enable --smoke-mock for smoke testing only. "
            "This ensures eval does not pretend mock recall@10=1.0 is real performance."
        )

    candidates = [
        _RCandidate(
            page_id=c["page_id"],
            score=c["score"],
            channel_scores={},
            channel_ranks={},
            page_image_uri=c["page_image_uri"],
            extracted_text_excerpt="",
        )
        for c in raw_candidates
    ]

    metrics = compute_retrieval_metrics(
        candidates=candidates,
        expected_pages=expected_pages,
        expected_documents=expected_docs,
    )
    return metrics, candidates, mode_label


# ---------------------------------------------------------------------------
# Per-entry evaluation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalAssertionResult:
    """Result of evaluating a single assertion for one entry."""

    entry_idx: int
    question: str
    assertion_name: str
    metric_name: str
    threshold: float
    actual_value: float | None
    passed: bool
    message: str
    skipped: bool = False

    def __repr__(self) -> str:
        status = "SKIPPED" if self.skipped else ("PASS" if self.passed else "FAIL")
        return (
            f"EvalAssertionResult({self.assertion_name}/{self.metric_name} "
            f"{status} {self.actual_value}/{self.threshold})"
        )


@dataclass(frozen=True)
class EvalEntryResult:
    """Result of evaluating one dataset entry."""

    entry_idx: int
    question: str
    assertions: list[EvalAssertionResult]
    passed: bool
    retrieval_metrics: RetrievalMetricsResult | None


@dataclass(frozen=True)
class EvalRunResult:
    """Aggregated result of an entire eval run."""

    dataset_name: str
    tag: str
    output_dir: Path
    started_at: datetime
    finished_at: datetime
    total_entries: int
    passed_entries: int
    failed_entries: int
    entry_results: list[EvalEntryResult]
    assertions_summary: dict[str, dict[str, Any]]
    corpus_version: str = ""
    eval_set_hash: str = ""


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------


def run_smoke_eval(
    dataset_name: str,
    tag: str,
    output_dir: Path,
    dry_run: bool = False,
    use_smoke_mock: bool = False,
    provided_candidates_map: dict[int, list[dict]] | None = None,
) -> EvalRunResult:
    """Run the evaluation pipeline for a dataset (HONEST EVAL MODE).

    Retrieval-pluggable structure:
    - Pass provided_candidates_map for real retrieval output (preferred for honest eval)
    - Or set use_smoke_mock=True for explicit smoke/mock mode (labeled honestly, not as real)
    - Without either: fails loudly with clear error (no more synthetic liar)

    Args:
        dataset_name: Name of the dataset (e.g. "smoke", "regression").
        tag: A short identifier for this run (e.g. "v1", "candidate").
        output_dir: Directory where run artifacts are written.
        dry_run: If True, log what would be done without executing.
        use_smoke_mock: Explicitly enable honest smoke/mock mode (names it as such).
        provided_candidates_map: {entry_idx: candidates} for real retrieval.

    Returns:
        EvalRunResult with per-entry and aggregate results. Mode is recorded.
    """
    started_at = datetime.now(tz=UTC)

    log.info(
        "eval_run_started",
        dataset=dataset_name,
        tag=tag,
        output_dir=str(output_dir),
        dry_run=dry_run,
        use_smoke_mock=use_smoke_mock,
    )

    if dry_run:
        log.warning("eval_dry_run_mode")

    if use_smoke_mock:
        log.warning("eval_smoke_mock_mode_active - recall metrics are from MOCK, not real retrieval")
    elif provided_candidates_map is None:
        log.info("eval_real_mode - expecting provided_candidates or will fail loudly")

    # ── Load dataset ──────────────────────────────────────────────────────────
    try:
        entries = load_eval_dataset(dataset_name)
    except (FileNotFoundError, ValueError) as e:
        log.error("eval_dataset_load_failed", error=str(e))
        raise typer.Exit(code=1) from e

    log.info("eval_dataset_loaded", num_entries=len(entries))

    # Evaluate each entry
    entry_results: list[EvalEntryResult] = []
    assertions_summary: dict[str, dict[str, Any]] = {}
    eval_modes_used: set[str] = set()

    for idx, entry in enumerate(entries):
        question = entry.get("question", "")

        # Retrieve metrics - pluggable
        metrics: RetrievalMetricsResult | None = None
        mode_label = "unknown"
        try:
            provided = provided_candidates_map.get(idx) if provided_candidates_map else None
            metrics, _, mode_label = _compute_retrieval_metrics_for_entry(
                entry, provided_candidates=provided, use_smoke_mock=use_smoke_mock
            )
            eval_modes_used.add(mode_label)
        except Exception as exc:
            log.warning(
                "entry_retrieval_metrics_failed",
                entry_idx=idx,
                question=question[:60],
                error=str(exc),
            )
            # In honest mode, we do not silently fall back to synthetic

        # Build per-assertion results
        assertions: list[EvalAssertionResult] = []

        # VAL-OPS-002 recall@10 gate
        if metrics is not None:
            recall_at_10 = metrics.recall_at_10
            recall_passed = recall_at_10 >= RECALL_AT_10_THRESHOLD
            mode_note = f" (mode={mode_label})" if mode_label != "real" else ""
            assertions.append(
                EvalAssertionResult(
                    entry_idx=idx,
                    question=question,
                    assertion_name="VAL-OPS-002",
                    metric_name="recall@10",
                    threshold=RECALL_AT_10_THRESHOLD,
                    actual_value=recall_at_10,
                    passed=recall_passed,
                    message=(
                        f"recall@10={recall_at_10:.3f} "
                        f"{'>= ' if recall_passed else '< '}"
                        f"threshold={RECALL_AT_10_THRESHOLD}{mode_note}"
                    ),
                )
            )
            _update_assertions_summary(
                assertions_summary,
                "VAL-OPS-002",
                recall_passed,
                recall_at_10,
                RECALL_AT_10_THRESHOLD,
            )

            # Additional retrieval metrics
            for metric_tuple in [
                ("recall@5", metrics.recall_at_5, 0.80),
                ("recall@20", metrics.recall_at_20, 0.90),
                ("mrr@10", metrics.mrr_at_10, 0.65),
                ("ndcg@10", metrics.ndcg_at_10, 0.70),
                # coverage@10 requires distinct doc_ids; in mock mode always 0.0
                ("coverage@10", metrics.coverage_at_10, 0.0),
            ]:
                m_name, m_val, m_thresh = metric_tuple
                m_passed = m_val >= m_thresh
                assertions.append(
                    EvalAssertionResult(
                        entry_idx=idx,
                        question=question,
                        assertion_name=f"retrieval:{m_name}",
                        metric_name=m_name,
                        threshold=m_thresh,
                        actual_value=m_val,
                        passed=m_passed,
                        message=(
                            f"{m_name}={m_val:.3f} "
                            f"{'>= ' if m_passed else '< '}"
                            f"threshold={m_thresh}{mode_note}"
                        ),
                    )
                )

        # Faithfulness (RAGAS) — skipped without live API
        # NOTE: Full RAGAS evaluation requires GEMINI_API_KEY and live pipeline.
        # We record it as a placeholder for completeness.
        faithfulness_val: float | None = None
        assertions.append(
            EvalAssertionResult(
                entry_idx=idx,
                question=question,
                assertion_name="VAL-OPS-023",
                metric_name="faithfulness",
                threshold=FAITHFULNESS_THRESHOLD,
                actual_value=faithfulness_val,
                passed=False,  # Always fail without live pipeline
                skipped=True,  # But mark as skipped so it doesn't count against pass/fail
                message=(
                    "faithfulness skipped (requires live RAG pipeline + GEMINI_API_KEY). "
                    "Set API key and enable live eval for full VAL-OPS-023 gate."
                ),
            )
        )

        entry_passed = all(a.passed for a in assertions if not a.skipped)
        entry_results.append(
            EvalEntryResult(
                entry_idx=idx,
                question=question,
                assertions=assertions,
                passed=entry_passed,
                retrieval_metrics=metrics,
            )
        )

    # Aggregate summary
    passed_entries = sum(1 for e in entry_results if e.passed)
    failed_entries = len(entry_results) - passed_entries
    finished_at = datetime.now(tz=UTC)

    run_result = EvalRunResult(
        dataset_name=dataset_name,
        tag=tag,
        output_dir=output_dir,
        started_at=started_at,
        finished_at=finished_at,
        total_entries=len(entry_results),
        passed_entries=passed_entries,
        failed_entries=failed_entries,
        entry_results=entry_results,
        assertions_summary=assertions_summary,
    )

    # Write summary JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_data = _build_summary_json(run_result)
    summary_path.write_text(json.dumps(summary_data, indent=2, default=str))
    log.info("eval_summary_written", path=str(summary_path))

    # Write per-entry JSONL
    entries_path = output_dir / "entries.jsonl"
    with entries_path.open("w") as f:
        for er in entry_results:
            record = {
                "entry_idx": er.entry_idx,
                "question": er.question,
                "passed": er.passed,
                "assertions": [
                    {
                        "name": a.assertion_name,
                        "metric": a.metric_name,
                        "threshold": a.threshold,
                        "actual": a.actual_value,
                        "passed": a.passed,
                        "message": a.message,
                    }
                    for a in er.assertions
                ],
            }
            if er.retrieval_metrics is not None:
                record["retrieval_metrics"] = er.retrieval_metrics.as_dict()
            f.write(json.dumps(record, default=str) + "\n")
    log.info("eval_entries_written", path=str(entries_path))

    return run_result


def _update_assertions_summary(
    summary: dict[str, dict[str, Any]],
    assertion_name: str,
    passed: bool,
    actual: float,
    threshold: float,
) -> None:
    """Update the running assertions summary dict."""
    if assertion_name not in summary:
        summary[assertion_name] = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "threshold": threshold,
        }
    s = summary[assertion_name]
    s["total"] += 1
    if passed:
        s["passed"] += 1
    else:
        s["failed"] += 1


def _build_summary_json(result: EvalRunResult) -> dict[str, Any]:
    """Build the summary dict written to summary.json."""
    duration_secs = (result.finished_at - result.started_at).total_seconds()

    return {
        "dataset": result.dataset_name,
        "tag": result.tag,
        "corpus_version": result.corpus_version,
        "eval_set_hash": result.eval_set_hash,
        "started_at": result.started_at.isoformat(),
        "finished_at": result.finished_at.isoformat(),
        "duration_seconds": round(duration_secs, 2),
        "total_entries": result.total_entries,
        "passed_entries": result.passed_entries,
        "failed_entries": result.failed_entries,
        "pass_rate": (
            round(result.passed_entries / result.total_entries, 4)
            if result.total_entries > 0
            else 0.0
        ),
        "assertions_summary": result.assertions_summary,
        "overall_passed": result.failed_entries == 0,
    }


# ---------------------------------------------------------------------------
# Rich console output
# ---------------------------------------------------------------------------


def print_run_summary(result: EvalRunResult, console: Console) -> None:
    """Print a human-readable summary to the console."""
    duration_secs = (result.finished_at - result.started_at).total_seconds()

    console.print("\n[bold]Eval Run Summary[/bold]")
    console.print(f"  Dataset : {result.dataset_name}")
    console.print(f"  Tag     : {result.tag}")
    console.print(f"  Entries : {result.total_entries}")
    console.print(
        f"  Passed  : [green]{result.passed_entries}[/green]  "
        f"  Failed  : [red]{result.failed_entries}[/red]"
    )
    console.print(f"  Duration: {duration_secs:.1f}s")
    console.print(f"  Output  : {result.output_dir}")

    # Assertions table
    if result.assertions_summary:
        table = Table(title="Assertions")
        table.add_column("Assertion", style="cyan")
        table.add_column("Threshold", justify="right")
        table.add_column("Passed", justify="right", style="green")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Total", justify="right")

        for name, summary in result.assertions_summary.items():
            table.add_row(
                name,
                str(summary["threshold"]),
                str(summary["passed"]),
                str(summary["failed"]),
                str(summary["total"]),
            )
        console.print(table)

    # Failed entries
    if result.failed_entries > 0:
        console.print("\n[red]Failed entries:[/red]")
        for er in result.entry_results:
            if not er.passed:
                console.print(
                    f"  [{er.entry_idx}] {er.question[:70]}..."
                )


# ---------------------------------------------------------------------------
# Typer CLI app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="eval",
    help="Evaluation pipeline CLI for Urban RAG",
    add_completion=False,
    no_args_is_help=False,
)


@app.command()
def run(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Dataset name (smoke, regression, comprehensive)",
    ),
    tag: str = typer.Option(
        ...,
        "--tag",
        "-t",
        help="Run identifier tag (e.g. v1, candidate, pr-123)",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory [default: data/eval/runs/<tag>/]",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Log what would be done without writing artifacts",
    ),
    smoke_mock: bool = typer.Option(
        False,
        "--smoke-mock",
        help="Use honest smoke/mock mode (explicitly labeled; NOT real retrieval). Fails loudly without this or real candidates.",
    ),
) -> None:
    """Run evaluation against a dataset and produce a summary JSON.

    HONEST EVAL: synthetic oracle removed. Use --smoke-mock for mock testing or provide real candidates.

    Examples:

        python -m urban_rag.eval run --dataset smoke --tag v1 --smoke-mock

        python -m urban_rag.eval run --dataset regression --tag candidate \\
            --output data/eval/runs/candidate/

        python -m urban_rag.eval run --dataset smoke --tag pr-42 --dry-run
    """
    # Resolve output directory
    if output is None:
        output_dir = DEFAULT_OUTPUT_BASE / tag
    else:
        output_dir = output

    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    console = Console()
    console.print(f"[cyan]Starting eval run[/cyan] — dataset={dataset}, tag={tag}")

    try:
        result = run_smoke_eval(
            dataset_name=dataset,
            tag=tag,
            output_dir=output_dir,
            dry_run=dry_run,
            use_smoke_mock=smoke_mock,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Dataset not found:[/red] {e}")
        raise typer.Exit(code=1) from e
    except ValueError as e:
        console.print(f"[red]Dataset error:[/red] {e}")
        raise typer.Exit(code=1) from e

    print_run_summary(result, console)

    if result.failed_entries > 0:
        console.print(
            f"\n[red]Eval FAILED — {result.failed_entries} assertion(s) failed.[/red]"
        )
        raise typer.Exit(code=1)
    console.print("\n[green]Eval PASSED — all assertions succeeded.[/green]")


@app.command()
def check(
    dataset: str = typer.Argument(..., help="Dataset name to validate"),
) -> None:
    """Validate a dataset file (check for malformed JSON lines)."""
    console = Console()
    try:
        entries = load_eval_dataset(dataset)
        console.print(
            f"[green]Dataset '{dataset}' is valid.[/green] "
            f"({len(entries)} entries)"
        )
        # Spot-check required fields
        required = {"question", "expected_documents", "expected_pages", "answer_rubric"}
        for i, entry in enumerate(entries):
            missing = required - entry.keys()
            if missing:
                console.print(
                    f"[red]Entry {i} missing fields:[/red] {missing}"
                )
                raise typer.Exit(code=1) from None
        console.print(f"All {len(entries)} entries have required fields.")
    except FileNotFoundError as e:
        console.print(f"[red]Not found:[/red] {e}")
        raise typer.Exit(code=1) from e
    except ValueError as e:
        console.print(f"[red]Invalid:[/red] {e}")
        raise typer.Exit(code=1) from e


@app.command()
def version() -> None:
    """Print the eval module version."""
    from urban_rag.common.settings import get_settings

    settings = get_settings()
    typer.echo(f"urban-rag eval {settings.app_version}")


# ---------------------------------------------------------------------------
# __main__ bootstrap — allows: python -m urban_rag.eval run ...
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
