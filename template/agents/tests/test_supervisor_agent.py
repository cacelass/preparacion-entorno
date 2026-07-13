from __future__ import annotations

from agents.agents.supervisor_agent import SupervisorAgent
from agents.config import ProjectConfig
from agents.context import SharedContext
from agents.core.base_agent import AgentResult


def _make_agent():
    return SupervisorAgent(context=SharedContext(root="/tmp", config=ProjectConfig(project_slug="x")))


def test_score_research_empty_returns_zero():
    agent = _make_agent()
    score = agent._score_research([], keywords=["ml", "data"], max_results=10)
    assert score == 0.0


def test_score_research_returns_positive():
    agent = _make_agent()
    papers = [{"title": "Machine learning advances", "abstract": "New methods in ML and data science."}]
    score = agent._score_research(papers, keywords=["ml", "data"], max_results=10)
    assert score > 0.0


def test_score_research_keywords_covered():
    agent = _make_agent()
    papers = [{"title": "Deep learning", "abstract": ""}]
    score = agent._score_research(papers, keywords=["deep", "learning"], max_results=10)
    assert score > 0.0


def test_score_research_empty_keywords():
    agent = _make_agent()
    papers = [{"title": "Any paper", "abstract": "Some content."}]
    score = agent._score_research(papers, keywords=[], max_results=10)
    assert score > 0.0


def test_default_score_failed_result():
    agent = _make_agent()
    res = AgentResult(False, "test", "test", "falló")
    score = agent._default_score(res)
    assert score == 0.0


def test_default_score_success():
    agent = _make_agent()
    res = AgentResult(True, "test", "test", "ok", data={"key": "value"})
    score = agent._default_score(res)
    assert score > 0.0


def test_default_score_penalizes_warnings():
    agent = _make_agent()
    res_ok = AgentResult(True, "test", "test", "ok", data={"k": "v"})
    res_warn = AgentResult(True, "test", "test", "ok", data={"k": "v"}, warnings=["cuidado"])
    assert agent._default_score(res_ok) > agent._default_score(res_warn)


def test_compete_empty_candidates():
    agent = _make_agent()
    result = agent.compete(candidates=[])
    assert not result.success


def test_compete_no_candidates_returns_error():
    agent = _make_agent()
    result = agent.compete(candidates=[])
    assert "No se pasaron candidatos" in result.message


def test_research_no_valid_backend():
    agent = _make_agent()
    result = agent.research(backends=["invalid_backend"])
    assert not result.success
    assert "Ningún backend válido" in result.message
