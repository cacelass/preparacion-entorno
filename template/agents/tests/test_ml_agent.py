from __future__ import annotations

import pytest

from agents.agents.ml_agent import MLAgent
from agents.config import ProjectConfig
from agents.context import SharedContext


@pytest.fixture
def ml_context(tmp_path):
    (tmp_path / "models").mkdir()
    return SharedContext(root=tmp_path, config=ProjectConfig(project_slug="mi_paquete"))


def test_list_models_empty_when_no_models(ml_context):
    agent = MLAgent(context=ml_context)
    result = agent.list_models()
    assert result.success
    assert result.data == []


def test_list_models_finds_joblib(ml_context):
    (ml_context.root / "models" / "modelo.joblib").write_text("dummy")
    agent = MLAgent(context=ml_context)
    result = agent.list_models()
    assert result.success
    assert len(result.data) == 1


def test_list_models_ignores_non_joblib_in_supervised(ml_context):
    (ml_context.root / "models" / "modelo.pt").write_text("dummy")
    agent = MLAgent(context=ml_context)
    result = agent.list_models()
    assert result.success
    assert result.data == []


def test_inspect_model_not_found(ml_context):
    agent = MLAgent(context=ml_context)
    result = agent.inspect_model(model_name="no_existe")
    assert not result.success


def test_inspect_model_not_found_with_suffix(ml_context):
    agent = MLAgent(context=ml_context)
    result = agent.inspect_model(model_name="no_existe.joblib")
    assert not result.success


def test_feature_importance_not_found(ml_context):
    agent = MLAgent(context=ml_context)
    result = agent.feature_importance(model_name="no_existe")
    assert not result.success


def test_check_overfitting_ok():
    config = ProjectConfig(project_slug="x")
    ctx = SharedContext(root="/tmp", config=config)
    agent = MLAgent(context=ctx)
    result = agent.check_overfitting(train_score=0.95, test_score=0.92)
    assert result.success
    assert result.data["verdict"] == "ok"
    assert result.warnings == []


def test_check_overfitting_detected():
    config = ProjectConfig(project_slug="x")
    ctx = SharedContext(root="/tmp", config=config)
    agent = MLAgent(context=ctx)
    result = agent.check_overfitting(train_score=0.99, test_score=0.80)
    assert result.success
    assert result.data["verdict"] != "ok"
    assert result.warnings


def test_check_overfitting_custom_threshold():
    config = ProjectConfig(project_slug="x")
    ctx = SharedContext(root="/tmp", config=config)
    agent = MLAgent(context=ctx)
    result = agent.check_overfitting(train_score=0.98, test_score=0.90, gap_threshold=0.05)
    assert result.data["gap"] > 0.05
    assert result.data["verdict"] != "ok"
