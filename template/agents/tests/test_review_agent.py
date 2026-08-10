from __future__ import annotations

import pytest

from agents.agents.review_agent import ReviewAgent
from agents.config import ProjectConfig
from agents.context import SharedContext


@pytest.fixture
def review_context(tmp_path):
    (tmp_path / "mi_paquete").mkdir()
    (tmp_path / "mi_paquete" / "__init__.py").write_text("")
    return SharedContext(root=tmp_path, config=ProjectConfig(project_slug="mi_paquete"))


def test_review_file_nonexistent(review_context):
    agent = ReviewAgent(context=review_context)
    result = agent.review_file(relative_path="no_existe.py")
    assert not result.success


def test_review_file_short(review_context):
    f = review_context.root / "mi_paquete" / "example.py"
    f.write_text("def foo():\n    return 42\n")
    agent = ReviewAgent(context=review_context)
    result = agent.review_file(relative_path="mi_paquete/example.py")
    assert result.success


# -- OMP-004: severidad + confianza + veredicto -------------------------------

def test_review_file_anota_severidad_y_veredicto(review_context):
    f = review_context.root / "mi_paquete" / "buggy.py"
    f.write_text(
        "import torch\n"
        "def cargar(ruta):\n"
        "    return torch.load(ruta, weights_only=False)\n"
    )
    agent = ReviewAgent(context=review_context)
    result = agent.review_file(relative_path="mi_paquete/buggy.py")
    assert result.success
    assert result.data["verdict"] == "incorrect", "weights_only=False debe bloquear (P0)"
    p0 = [x for x in result.data["findings"] if x["severity"] == "P0"]
    assert p0 and p0[0]["kind"] == "weights_only_false"
    assert p0[0]["confidence"] == "high"


def test_review_verdict_review_con_hallazgo_p1(review_context):
    f = review_context.root / "mi_paquete" / "mod.py"
    f.write_text(
        "def registrar(items=[]):\n"  # mutable_default → P1
        "    return items\n"
    )
    agent = ReviewAgent(context=review_context)
    result = agent.review_file(relative_path="mi_paquete/mod.py")
    assert result.success
    assert result.data["verdict"] == "review"
    assert any(x["severity"] == "P1" and x["kind"] == "mutable_default" for x in result.data["findings"])


def test_review_verdict_correct_sin_p0_ni_p1(review_context):
    f = review_context.root / "mi_paquete" / "clean.py"
    f.write_text(
        "def suma(a: int, b: int) -> int:\n"
        "    return a + b\n"
    )
    agent = ReviewAgent(context=review_context)
    result = agent.review_file(relative_path="mi_paquete/clean.py")
    assert result.success
    assert result.data["verdict"] == "correct"


def test_review_findings_ordenados_por_severidad(review_context):
    f = review_context.root / "mi_paquete" / "mix.py"
    f.write_text(
        "import torch\n"
        "def ok():\n"
        "    pass\n"
        "# TODO: limpiar\n"
        "def cargar():\n"
        "    return torch.load('x', weights_only=False)\n"
    )
    agent = ReviewAgent(context=review_context)
    result = agent.review_package()
    findings = result.data["findings"]
    severities = [x["severity"] for x in findings if x["severity"] in ("P0", "P1", "P2", "P3")]
    assert severities == sorted(severities, key={"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get)
