import typer
from rich.console import Console
from rich.panel import Panel
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
    query: Path = typer.Option(
        ...,
        "--query",
        "-q",
        help="Query genome FASTA/FASTQ file or directory of FASTAs/FASTQs.",
        exists=True,
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
    Predict all phenotypes in a model bundle for a query genome.

    Runs mash screen once against the reference sketch, then extracts features
    and predicts each phenotype with its own classifier. The output TSV has one
    row per phenotype.

    [bold]Example:[/]
        phyloscreen predict -q query.fasta -m model_v1/ -o result.tsv
    """
    setup_logging(verbose=verbose)

    console.print(
        Panel(
            f"[bold]Predicting phenotype[/]\n\n"
            f"Query: [cyan]{query}[/]\n"
            f"Model: [cyan]{model}[/]\n"
            f"Output: [cyan]{output}[/]",
            title="[blue]phyloscreen predict[/]",
        )
    )

    # Import here to avoid circular imports and speed up --help
    from phyloscreen.predict import predict_phenotype

    try:
        results = predict_phenotype(
            query_path=query,
            model_dir=model,
            output_path=output,
            threads=threads,
        )
        console.print(f"\n[green]✓[/] Prediction complete ({len(results)} phenotype(s))")
        for r in results:
            console.print(f"  {r.phenotype}: [bold]{r.prediction}[/] (p={r.probability:.3f})")
        console.print(f"  Results saved to: [cyan]{output}[/]")
    except Exception as e:
        console.print(f"[red]✗ Error:[/] {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
