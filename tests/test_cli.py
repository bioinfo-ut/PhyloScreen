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
