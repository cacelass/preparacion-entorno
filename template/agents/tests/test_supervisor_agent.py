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


# -- fan-in: sintetizar perspectivas, no competir ------------------------------

class TestSynthesize:
    """
    `compete` elige un ganador entre alternativas; `synthesize` integra ángulos
    distintos del mismo problema. Son cosas diferentes y no deben confundirse.
    """

    def _perspectivas(self):
        return [
            {"agent": "test", "action": "run_tests", "label": "calidad"},
            {"agent": "doctor", "action": "checkup", "label": "entorno"},
        ]

    def test_sin_perspectivas_falla(self, context):
        from agents.agents.supervisor_agent import SupervisorAgent
        r = SupervisorAgent(context=context).synthesize(perspectives=[])
        assert not r.success

    def test_integra_todas_las_respuestas(self, context, monkeypatch):
        from agents.agents.supervisor_agent import SupervisorAgent
        from agents.core.base_agent import AgentResult

        def fake_run(self, agent, action, **kw):
            return AgentResult(True, agent, action, f"{agent} respondio", data={"n": 1})

        monkeypatch.setattr("agents.orchestrator.Orchestrator.run", fake_run)
        r = SupervisorAgent(context=context).synthesize(
            perspectives=self._perspectivas(), parallel=False, question="¿como esta el proyecto?"
        )
        assert r.success
        assert r.data["consensus"] == "unánime"
        assert set(r.data["findings"]) == {"calidad", "entorno"}
        assert r.data["question"] == "¿como esta el proyecto?"

    def test_una_perspectiva_rota_no_tumba_al_resto(self, context, monkeypatch):
        from agents.agents.supervisor_agent import SupervisorAgent
        from agents.core.base_agent import AgentResult

        def fake_run(self, agent, action, **kw):
            if agent == "doctor":
                raise RuntimeError("boom")
            return AgentResult(True, agent, action, "ok")

        monkeypatch.setattr("agents.orchestrator.Orchestrator.run", fake_run)
        r = SupervisorAgent(context=context).synthesize(
            perspectives=self._perspectivas(), parallel=False
        )
        assert r.success, "con una perspectiva viva sigue habiendo sintesis"
        assert r.data["failed"] == ["entorno"]
        assert "parcial" in r.data["consensus"]

    def test_si_ninguna_responde_no_hay_sintesis(self, context, monkeypatch):
        from agents.agents.supervisor_agent import SupervisorAgent
        from agents.core.base_agent import AgentResult

        def fake_run(self, agent, action, **kw):
            return AgentResult(False, agent, action, "no pude", needs=["dame el dataset"])

        monkeypatch.setattr("agents.orchestrator.Orchestrator.run", fake_run)
        r = SupervisorAgent(context=context).synthesize(
            perspectives=self._perspectivas(), parallel=False
        )
        assert not r.success
        assert r.needs, "las preguntas de cada perspectiva deben subir, no perderse"

    def test_acumula_avisos_de_cada_perspectiva(self, context, monkeypatch):
        from agents.agents.supervisor_agent import SupervisorAgent
        from agents.core.base_agent import AgentResult

        def fake_run(self, agent, action, **kw):
            return AgentResult(True, agent, action, "ok", warnings=[f"ojo con {agent}"])

        monkeypatch.setattr("agents.orchestrator.Orchestrator.run", fake_run)
        r = SupervisorAgent(context=context).synthesize(
            perspectives=self._perspectivas(), parallel=False
        )
        assert len(r.warnings) == 2
        assert any("calidad:" in w for w in r.warnings)
