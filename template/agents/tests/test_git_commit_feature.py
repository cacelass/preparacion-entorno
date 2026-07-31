from __future__ import annotations

import subprocess

from agents.agents.git_agent import GitAgent


def _write_versioned_files(root):
    (root / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.1.0"\n')
    (root / "README.md").write_text(
        "![Version](https://img.shields.io/badge/Version-0.1.0-green)\n\n**Versión:** 0.1.0\n"
    )


def test_commit_feature_dry_run_no_escribe_nada_y_propone(context):
    _write_versioned_files(context.root)
    subprocess.run(["git", "add", "-A"], cwd=context.root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "chore: setup"], cwd=context.root, check=True)
    (context.root / "mi_paquete" / "modulo.py").write_text("print('feature')\n")

    agent = GitAgent(context=context)
    result = agent.commit_feature(id="DATA-001", title="EDA del dataset", dry_run=True)

    assert result.success
    assert result.data["next_version"] == "0.1.1"
    assert result.data["suggested_message"] == "feat(DATA-001): EDA del dataset"
    assert any("mi_paquete" in f for f in result.data["changed_files"]), result.data["changed_files"]
    assert 'version = "0.1.0"' in (context.root / "pyproject.toml").read_text(), "dry_run no debe escribir"
    assert not (context.root / "CHANGELOG.md").exists(), "dry_run no debe crear el changelog"

    log = subprocess.run(["git", "log", "--oneline"], cwd=context.root, capture_output=True, text=True, check=True).stdout
    assert "feat(DATA-001)" not in log, "dry_run no debe commitear"


def test_commit_feature_cierra_con_bump_y_commit_sin_tag(context):
    _write_versioned_files(context.root)
    subprocess.run(["git", "add", "-A"], cwd=context.root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "chore: setup"], cwd=context.root, check=True)
    (context.root / "mi_paquete" / "modulo.py").write_text("print('feature')\n")

    agent = GitAgent(context=context)
    result = agent.commit_feature(id="DATA-001", title="EDA del dataset")

    assert result.success
    assert 'version = "0.1.1"' in (context.root / "pyproject.toml").read_text()
    readme = (context.root / "README.md").read_text()
    assert "Version-0.1.1-green" in readme and "**Versión:** 0.1.1" in readme

    changelog = (context.root / "CHANGELOG.md").read_text()
    assert "EDA del dataset (DATA-001)" in changelog and "## [Unreleased]" in changelog

    log = subprocess.run(["git", "log", "--oneline"], cwd=context.root, capture_output=True, text=True, check=True).stdout
    assert "feat(DATA-001): EDA del dataset" in log

    tags = subprocess.run(["git", "tag"], cwd=context.root, capture_output=True, text=True, check=True).stdout
    assert not tags.strip(), "commit_feature no debe crear tags"

    show = subprocess.run(
        ["git", "show", "--stat", "HEAD"], cwd=context.root, capture_output=True, text=True, check=True
    ).stdout
    assert "pyproject.toml" in show and "README.md" in show and "CHANGELOG.md" in show


def test_commit_feature_sin_version_en_pyproject_avisa_y_commitea(context):
    (context.root / "pyproject.toml").write_text('[project]\nname = "x"\n')  # sin "version = ..."
    subprocess.run(["git", "add", "-A"], cwd=context.root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "chore: setup"], cwd=context.root, check=True)
    (context.root / "mi_paquete" / "modulo.py").write_text("print('feature')\n")

    agent = GitAgent(context=context)
    result = agent.commit_feature(id="DATA-001", title="EDA del dataset")

    assert result.success
    assert any("versión" in w.lower() for w in result.warnings)
    log = subprocess.run(["git", "log", "--oneline"], cwd=context.root, capture_output=True, text=True, check=True).stdout
    assert "feat(DATA-001): EDA del dataset" in log


def test_commit_feature_exige_id_y_titulo(context):
    agent = GitAgent(context=context)
    result = agent.commit_feature(id="", title="")
    assert not result.success
    assert any("id" in n.lower() for n in result.needs)
    assert any("título" in n.lower() for n in result.needs)
