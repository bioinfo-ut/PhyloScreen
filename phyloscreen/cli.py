import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from pathlib import Path
from typing import Optional

from phyloscreen import __version__
from phyloscreen.utils import setup_logging

app = typer.Typer(
    name="phyloscreen",
    help="Predict bacterial phenotypes using mash screen and logistic regression.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold blue]phyloscreen[/] version [green]{__version__}[/]")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """
    PhyloScreen: Predict bacterial phenotypes using mash screen and logistic regression.

    Use 'phyloscreen train' to build a model from reference genomes,
    then 'phyloscreen predict' to classify a query genome.
    """
    pass


@app.command()
def train(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="TSV file with column 'path' plus one or more phenotype columns (1/0, blank = missing).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output directory for trained model bundle.",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    kmer_size: int = typer.Option(
        21,
        "--kmer_size",
        "-k",
        help="K-mer size for mash sketch.",
        min=1,
    ),
    sketch_size: int = typer.Option(
        100000,
        "--sketch_size",
        "-s",
        help="Sketch size for mash sketch.",
        min=100,
    ),
    threads: int = typer.Option(
        4,
        "--threads",
        "-t",
        help="Number of threads for parallel processing.",
        min=1,
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducibility.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging.",
    ),
) -> None:
    """
    Train a phenotype prediction model from reference genomes.

    Builds a mash sketch from reference genomes, extracts features based on
    mash screen results, and trains one logistic regression classifier per
    phenotype column. Isolates with a blank value for a phenotype are left out
    of that phenotype's training and neighbour pool but still used for the others.

    [bold]Example:[/]
        phyloscreen train -i refs.tsv -o model_v1/
    """
    setup_logging(verbose=verbose)

    console.print(
        Panel(
            f"[bold]Training phenotype prediction model[/]\n\n"
            f"Input: [cyan]{input}[/]\n"
            f"Output: [cyan]{output}[/]\n"
            f"Threads: [cyan]{threads}[/]\n",
            title="[blue]phyloscreen train[/]",
        )
    )

    # Import here to avoid circular imports and speed up --help
    from phyloscreen.train import train_model

    try:
        result = train_model(
            references=input,
            output_dir=output,
            kmer_size=kmer_size,
            sketch_size=sketch_size,
            threads=threads,
            seed=seed
        )
        console.print(f"\n[green]✓[/] Model trained successfully ({len(result.results)} phenotype(s))")

        table = Table(title="Cross-validation metrics", show_lines=False)
        table.add_column("Phenotype", style="bold")
        table.add_column("n", justify="right")
        table.add_column("AUC", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1", justify="right")
        for r in result.results.values():
            table.add_row(
                r.phenotype,
                str(r.n_labelled),
                f"{r.auc:.3f}",
                f"{r.accuracy:.3f}",
                f"{r.precision:.3f}",
                f"{r.recall:.3f}",
                f"{r.f1:.3f}",
            )
        console.print(table)
        console.print(f"  Model saved to: [cyan]{output}[/]")
    except Exception as e:
        console.print(f"[red]✗ Error:[/] {e}")
        raise typer.Exit(code=1)


@app.command()
def predict(
    query: Optional[Path] = typer.Option(
        None,
        "--query",
        "-q",
        help="One isolate: a FASTA/FASTQ file, or a directory of read files belonging to that isolate.",
        exists=True,
        resolve_path=True,
    ),
    query_list: Optional[Path] = typer.Option(
        None,
        "--query-list",
        "-l",
        help="TSV listing many isolates: column 'path' (file or directory per isolate) and optional 'id'.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to trained model directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output TSV path (one row per phenotype).",
        resolve_path=True,
    ),
    threads: int = typer.Option(
        4,
        "--threads",
        "-t",
        help="Number of threads for parallel processing.",
        min=1,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging.",
    ),
) -> None:
    """
    Predict all phenotypes in a model bundle for one isolate or a list of isolates.

    Runs mash screen once per isolate against the reference sketch, then extracts
    features and predicts each phenotype with its own classifier. The output TSV
    has one row per isolate and phenotype. Give exactly one of --query / --query-list.

    With --query-list, an isolate that fails (missing file, mash error) is written
    with prediction 'error', the rest are still processed, and the exit code is 1.

    [bold]Examples:[/]
        phyloscreen predict -q query.fasta -m model_v1/ -o result.tsv
        phyloscreen predict -l queries.tsv -m model_v1/ -o results.tsv
    """
    setup_logging(verbose=verbose)

    if (query is None) == (query_list is None):
        console.print("[red]✗ Error:[/] give exactly one of --query/-q or --query-list/-l")
        raise typer.Exit(code=2)

    console.print(
        Panel(
            f"[bold]Predicting phenotypes[/]\n\n"
            + (f"Query: [cyan]{query}[/]\n" if query is not None else f"Query list: [cyan]{query_list}[/]\n")
            + f"Model: [cyan]{model}[/]\n"
            f"Output: [cyan]{output}[/]",
            title="[blue]phyloscreen predict[/]",
        )
    )

    # Import here to avoid circular imports and speed up --help
    from phyloscreen import predict as predict_mod

    if query is not None:
        try:
            results = predict_mod.predict_phenotype(
                query_path=query,
                model_dir=model,
                output_path=output,
                threads=threads,
            )
        except Exception as e:
            console.print(f"[red]✗ Error:[/] {e}")
            raise typer.Exit(code=1)
        console.print(f"\n[green]✓[/] Prediction complete ({len(results)} phenotype(s))")
        for r in results:
            console.print(f"  {r.query_id} {r.phenotype}: [bold]{r.prediction}[/] (p={r.probability:.3f})")
        console.print(f"  Results saved to: [cyan]{output}[/]")
        return

    try:
        queries = predict_mod.read_query_list(query_list)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Predicting", total=len(queries))
            batch = predict_mod.predict_queries(
                queries,
                model_dir=model,
                output_path=output,
                threads=threads,
                on_progress=lambda q: progress.advance(task),
            )
    except Exception as e:
        console.print(f"[red]✗ Error:[/] {e}")
        raise typer.Exit(code=1)

    n_phenotypes = len({r.phenotype for r in batch.results})
    n_ok = len(queries) - len(batch.failures)
    console.print(
        f"\n[green]✓[/] Predicted {n_phenotypes} phenotype(s) for {n_ok}/{len(queries)} isolate(s)"
    )
    console.print(f"  Results saved to: [cyan]{output}[/]")
    if batch.failures:
        console.print(f"\n[red]✗ {len(batch.failures)} isolate(s) failed[/] (rows written with prediction 'error'):")
        for qid, msg in batch.failures.items():
            console.print(f"  {qid}: {msg}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
