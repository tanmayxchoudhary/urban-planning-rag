"""CLI commands (Typer)."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from urban_rag.common.logging import configure_logging, get_logger
from urban_rag.common.settings import get_settings


def _get_version() -> str:
    """Return the application version string."""
    return get_settings().app_version


def main() -> None:
    """Main entry point for the urban-rag CLI."""
    # Handle --version flag before Typer processes it
    import sys

    if "--version" in sys.argv or "-V" in sys.argv:
        typer.echo(f"urban-rag {_get_version()}")
        sys.exit(0)

    app = typer.Typer(
        name="urban-rag",
        help="Visual RAG system for Indian urban planning regulations",
        add_completion=False,
        no_args_is_help=True,
    )

    # Configure logging for CLI service
    configure_logging(service="cli")
    log = get_logger(__name__)

    @app.callback()
    def common_options(
        ctx: typer.Context,
        verbose: Annotated[
            bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
        ] = False,
    ) -> None:
        """Common options for all commands."""
        if verbose:
            import logging

            logging.getLogger().setLevel(logging.DEBUG)
            log.debug("verbose_logging_enabled")

    # Import and register subcommands
    from urban_rag.cli.corpus import register as register_corpus

    register_corpus(app)

    @app.command("ingest")
    def ingest_cmd(
        path: Annotated[str, typer.Argument(help="PDF file or directory to ingest")],
        rebuild: Annotated[
            bool, typer.Option("--rebuild", help="Force re-render and re-embed")
        ] = False,
        skip_eval: Annotated[
            bool, typer.Option("--skip-eval", help="Skip eval validation")
        ] = False,
    ) -> None:
        """Ingest a PDF document (or all PDFs in a directory) into the corpus."""
        from pathlib import Path

        from urban_rag.cli.ingest import ingest_directory, ingest_file
        from urban_rag.common.errors import IngestError, ValidationError

        console = Console()
        p = Path(path)

        try:
            if p.is_dir():
                exit_code, count = ingest_directory(p, rebuild=rebuild, skip_eval=skip_eval)
                if count == 0:
                    console.print(f"[yellow]No PDFs found in:[/yellow] {path}")
                    raise typer.Exit(code=0)
                console.print(
                    f"[green]Ingested {count} PDF(s) from:[/green] {path}"
                )
                raise typer.Exit(code=exit_code)
            ingest_file(p, rebuild=rebuild, skip_eval=skip_eval)
            console.print(f"[green]Successfully ingested:[/green] {path}")
        except ValidationError as e:
            console.print(f"[red]Validation error:[/red] {e.message}")
            raise typer.Exit(code=1) from e
        except IngestError as e:
            console.print(f"[red]Ingest error:[/red] {e.message}")
            log.error("ingest_failed", path=path, error=str(e))
            raise typer.Exit(code=1) from e
        except typer.Exit:
            raise  # Let typer.Exit propagate cleanly
        except Exception as e:
            console.print(f"[red]Ingest error:[/red] {e}")
            log.error("ingest_failed", path=path, error=str(e))
            raise typer.Exit(code=1) from e

    @app.command("query")
    def query_cmd(
        question: Annotated[str, typer.Argument(help="Question to ask")],
        retrieve_only: Annotated[
            bool,
            typer.Option(
                "--retrieve-only", help="Only retrieve, don't generate"
            ),
        ] = False,
        top_k: Annotated[
            int, typer.Option("--top-k", help="Number of candidates", min=1, max=50)
        ] = 5,
        timeout: Annotated[
            int,
            typer.Option(
                "--timeout", help="Query timeout in seconds (default: 60)"
            ),
        ] = 60,
    ) -> None:
        """Query the corpus with a question."""
        import signal

        from urban_rag.retrieve.orchestrator import retrieve

        console = Console()
        log = get_logger(__name__)

        # Set up a timeout handler
        def timeout_handler(signum, frame):
            console.print(f"\n[yellow]Query timed out after {timeout}s[/yellow]")
            raise typer.Exit(code=124)

        # Register the timeout signal if supported
        if hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

        try:
            result = retrieve(
                query=question,
                top_k=top_k,
                use_rerank=not retrieve_only,  # Skip rerank in retrieve-only mode
            )

            console.print(f"\n[bold]Query:[/bold] {question}")
            console.print(f"[bold]Strategy:[/bold] {result.retrieval_strategy}")
            console.print(f"[bold]Latency:[/bold] {result.latency_ms}ms")

            if result.flags:
                flags_str = ", ".join(f"{k}={v}" for k, v in result.flags.items())
                console.print(f"[bold]Flags:[/bold] {flags_str}")

            if result.candidates:
                console.print(
                    f"\n[bold green]Top {len(result.candidates)} candidates:[/bold green]"
                )
                for i, candidate in enumerate(result.candidates, 1):
                    channels = ", ".join(
                        f"{ch}:{score:.3f}"
                        for ch, score in candidate.channel_scores.items()
                    )
                    console.print(
                        f"  [{i}] {candidate.page_id}  "
                        f"score={candidate.score:.4f}  "
                        f"channels={{{channels}}}"
                    )
                    if candidate.extracted_text_excerpt:
                        excerpt = candidate.extracted_text_excerpt[:80].replace("\n", " ")
                        console.print(f"      excerpt: {excerpt}...")
            else:
                console.print("\n[yellow]No candidates found.[/yellow]")

            log.info(
                "query_command_complete",
                question=question[:50],
                candidates=len(result.candidates),
                latency_ms=result.latency_ms,
            )

        except Exception as e:
            log.error("query_command_failed", question=question[:50], error=str(e))
            console.print(f"[red]Query failed:[/red] {e}")
            raise typer.Exit(code=1) from e
        finally:
            # Cancel the alarm if still pending
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    @app.command("version")
    def version_cmd() -> None:
        """Show version information."""
        from urban_rag.common.settings import get_settings

        settings = get_settings()
        typer.echo(f"urban-rag {settings.app_version}")

    app()


if __name__ == "__main__":
    main()
