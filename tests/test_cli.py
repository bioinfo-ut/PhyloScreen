"""Smoke tests for the phyloscreen command-line interface."""

from typer.testing import CliRunner

from phyloscreen import __version__
from phyloscreen.cli import app

runner = CliRunner()


def test_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.output
    assert "predict" in result.output


def test_version_matches_package():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_subcommand_help():
    for cmd in ("train", "predict"):
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0, cmd
        assert "--threads" in result.output
