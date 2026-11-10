"""Smoke tests for the phyloscreen command-line interface."""

import re

from typer.testing import CliRunner

from phyloscreen import __version__
from phyloscreen.cli import app

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def plain(text: str) -> str:
    """Strip ANSI escape sequences (Typer forces colour on CI runners)."""
    return _ANSI.sub("", text)


def test_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = plain(result.output)
    assert "train" in out
    assert "predict" in out


def test_version_matches_package():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in plain(result.output)


def test_subcommand_help():
    for cmd in ("train", "predict"):
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0, cmd
        assert "--threads" in plain(result.output), cmd


def test_predict_help_lists_query_list():
    result = runner.invoke(app, ["predict", "--help"])
    out = plain(result.output)
    assert "--query-list" in out
    assert "phenotype columns" in plain(runner.invoke(app, ["train", "--help"]).output)


def test_predict_requires_exactly_one_input(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (tmp_path / "q.fa").write_text(">q\nACGT\n")
    (tmp_path / "l.tsv").write_text("path\nq.fa\n")

    neither = runner.invoke(app, ["predict", "-m", str(model_dir), "-o", str(tmp_path / "o.tsv")])
    assert neither.exit_code == 2
    assert "exactly one" in plain(neither.output)

    both = runner.invoke(
        app,
        ["predict", "-q", str(tmp_path / "q.fa"), "-l", str(tmp_path / "l.tsv"),
         "-m", str(model_dir), "-o", str(tmp_path / "o.tsv")],
    )
    assert both.exit_code == 2


def test_predict_list_exits_nonzero_on_failures(tmp_path, monkeypatch):
    from phyloscreen import predict as predict_mod
    from phyloscreen.predict import BatchResult, Query, QueryResult

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    lst = tmp_path / "l.tsv"
    lst.write_text("path\tid\n/x/a.fa\tA\n/x/b.fa\tB\n")

    def fake_predict_queries(queries, model_dir, output_path, threads=4, on_progress=None, raise_on_error=False):
        assert [q.id for q in queries] == ["A", "B"]
        for q in queries:
            on_progress(q)
        rows = [QueryResult("A", "/x/a.fa", "amp", 1, 0.9, 0.9, 0.1),
                QueryResult("B", "/x/b.fa", "amp", "error", 0.0, 0.0, 0.0, error="boom")]
        return BatchResult(results=rows, failures={"B": "FileNotFoundError: boom"})

    monkeypatch.setattr(predict_mod, "predict_queries", fake_predict_queries)

    result = runner.invoke(app, ["predict", "-l", str(lst), "-m", str(model_dir), "-o", str(tmp_path / "o.tsv")])
    out = plain(result.output)
    assert result.exit_code == 1, out
    assert "1/2 isolate(s)" in out
    assert "B: FileNotFoundError: boom" in out
