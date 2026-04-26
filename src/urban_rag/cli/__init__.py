"""CLI commands (Typer)."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from urban_rag.common.logging import configure_logging, get_logger


def main() -> None:
    """Main entry point for the urban-rag CLI."""
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
        """Ingest a PDF document into the corpus."""
        console = Console()
        console.print("[yellow]ingest command not yet implemented[/yellow]")
        log.info("ingest_command_placeholder", path=path, rebuild=rebuild, skip_eval=skip_eval)

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
    ) -> None:
        """Query the corpus with a question."""
        console = Console()
        console.print("[yellow]query command not yet implemented[/yellow]")
        log.info(
            "query_command_placeholder",
            question=question,
            retrieve_only=retrieve_only,
            top_k=top_k,
        )

    @app.command("version")
    def version_cmd() -> None:
        """Show version information."""
        from urban_rag.common.settings import get_settings

        settings = get_settings()
        typer.echo(f"urban-rag {settings.app_version}")

    app()


if __name__ == "__main__":
    main()
