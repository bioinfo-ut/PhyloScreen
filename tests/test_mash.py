"""Tests for the mash wrapper. Parsing tests run everywhere; the live test needs mash on PATH."""

import shutil
import subprocess

import pytest

from phyloscreen import mash


def test_parse_screen_output_sorts_by_shared_hashes():
    raw = (
        "0.95\t300/1000\t1\t0\tref_c.fa\t[comment c]\n"
        "0.99\t900/1000\t1\t0\tref_a.fa\n"
        "0.97\t600/1000\t1\t0\tref_b.fa\n"
    )
    df = mash.parse_screen_output(raw)

    assert list(df["reference_path"]) == ["ref_a.fa", "ref_b.fa", "ref_c.fa"]
    assert list(df["shared_hashes"]) == [900, 600, 300]
    assert df["total_hashes"].eq(1000).all()
    assert df.loc[2, "query_comment"] == "[comment c]"


def test_parse_screen_output_empty():
    df = mash.parse_screen_output("")
    assert df.empty
    assert list(df.columns) == mash.SCREEN_COLUMNS


def test_parse_screen_output_columns_match_constant():
    df = mash.parse_screen_output("0.99\t900/1000\t1\t0\tref_a.fa\n")
    assert list(df.columns) == mash.SCREEN_COLUMNS


def test_check_mash_parses_version(monkeypatch):
    def fake_run(cmd, **kwargs):
        assert cmd == ["mash", "--version"]
        return subprocess.CompletedProcess(cmd, 0, stdout="2.3\n", stderr="")

    monkeypatch.setattr(mash.subprocess, "run", fake_run)
    v = mash.check_mash()
    assert (v.major, v.minor, v.patch) == (2, 3, 0)
    assert str(v) == "2.3"


def test_check_mash_failure_raises(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="boom")

    monkeypatch.setattr(mash.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="boom"):
        mash.check_mash()


@pytest.mark.skipif(shutil.which("mash") is None, reason="mash binary not on PATH")
def test_check_mash_live():
    v = mash.check_mash()
    assert (v.major, v.minor) >= (2, 3)
