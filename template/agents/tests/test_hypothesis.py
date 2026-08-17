from __future__ import annotations

from typing import Any

from hypothesis import given, assume
from hypothesis import strategies as st

from agents.contracts import CONTRACTS, Contract, validate_contracts
from agents.core.base_agent import _CONJUGACIONES, AgentResult, BaseAgent
from agents.context import SharedContext
from agents.config import ProjectConfig


# ─── AgentResult invariants ────────────────────────────────────────────────


@given(success=st.booleans(), message=st.text())
def test_agent_result_bool_equals_success(success: bool, message: str):
    r = AgentResult(success=success, agent="x", action="y", message=message)
    assert bool(r) == success


@given(
    success=st.booleans(),
    agent=st.text(min_size=1),
    action=st.text(min_size=1),
    message=st.text(),
    data=st.none() | st.just(42) | st.lists(st.integers()),
    n_warnings=st.integers(min_value=0, max_value=5),
)
def test_agent_result_repr_roundtrip(
    success: bool, agent: str, action: str, message: str,
    data: Any, n_warnings: int,
):
    warnings = [f"w{i}" for i in range(n_warnings)]
    r = AgentResult(success=success, agent=agent, action=action,
                    message=message, data=data, warnings=warnings)
    status = "OK" if success else "FAIL"
    assert status in repr(r)
    assert agent in repr(r)
    assert action in repr(r)


# ─── can_handle scoring invariants ─────────────────────────────────────────


class _TestAgent(BaseAgent):
    name = "test_agent"
    description = "testing"
    capabilities: list[str] = []

    def actions(self):
        return {}


class _EmptyAgent(BaseAgent):
    name = "empty"
    description = "no capabilities"
    capabilities: list[str] = []

    def actions(self):
        return {}


@given(
    capabilities=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10, unique=True),
    query=st.text(max_size=100),
)
def test_can_handle_range(capabilities: list[str], query: str):
    agent = _TestAgent.__new__(_TestAgent)
    agent.capabilities = capabilities
    agent.ctx = SharedContext(root="/tmp", config=ProjectConfig())

    score = agent.can_handle(query)

    assert 0.0 <= score <= 1.0


@given(query=st.text(max_size=200))
def test_can_handle_empty_capabilities(query: str):
    agent = _EmptyAgent.__new__(_EmptyAgent)
    agent.capabilities = []
    agent.ctx = SharedContext(root="/tmp", config=ProjectConfig())

    assert agent.can_handle(query) == 0.0


@given(
    keyword=st.from_regex(r"[a-zA-Z]+", fullmatch=True).filter(lambda s: len(s) > 0),
    padding=st.text(max_size=50),
)
def test_can_handle_exact_match(keyword: str, padding: str):
    assume(keyword.strip())
    agent = _TestAgent.__new__(_TestAgent)
    agent.capabilities = [keyword]
    agent.ctx = SharedContext(root="/tmp", config=ProjectConfig())

    query = f"{padding} {keyword} {padding}"
    score = agent.can_handle(query)

    assert score >= 0.4


@given(
    keyword=st.from_regex(r"[a-zA-Z]+", fullmatch=True),
    infix=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=5),
)
def test_can_handle_no_substring_match(keyword: str, infix: str):
    """'ci' no debe matchear como subcadena de 'dependencias'."""
    assume(keyword)
    assume(infix)
    no_match = keyword + infix
    # Si `keyword` es una forma conjugada real (p.ej. "revisa") y `no_match`
    # su canónica ("revisar"), el fallback de conjugación legitima el match:
    # es palabra completa de la forma canónica, no una subcadena. Ese caso
    # está cubierto por los tests de ruteo conjugado; aquí se excluye para
    # probar el invariante anti-subcadena puro.
    assume(_CONJUGACIONES.get(keyword) != no_match)
    agent = _TestAgent.__new__(_TestAgent)
    agent.capabilities = [no_match]
    agent.ctx = SharedContext(root="/tmp", config=ProjectConfig())

    score = agent.can_handle(keyword)
    assert score == 0.0


# ─── best_action invariants ────────────────────────────────────────────────


class _ActionAgent(BaseAgent):
    name = "action_test"
    description = "testing actions"

    def __init__(self, actions: dict):
        self._acts = actions
        self.ctx = SharedContext(root="/tmp", config=ProjectConfig())

    def actions(self):
        return self._acts


@given(
    action_names=st.lists(
        st.from_regex(r"[a-z]+(_[a-z]+)*", fullmatch=True).filter(lambda s: len(s) > 0),
        min_size=1, max_size=10, unique=True,
    ),
    query=st.text(max_size=100),
)
def test_best_action_returns_valid_name(action_names: list[str], query: str):
    actions = {name: lambda: None for name in action_names}
    agent = _ActionAgent(actions)

    best = agent.best_action(query)
    if best is not None:
        assert best in action_names


@given(
    action_name=st.from_regex(r"[a-z]+(_[a-z]+)*", fullmatch=True).filter(lambda s: len(s) > 0),
)
def test_best_action_finds_exact_word_match(action_name: str):
    agent = _ActionAgent({action_name: lambda: None})
    # Extraer una palabra del nombre de la acción
    words = action_name.split("_")
    assume(len(words) > 0)
    query = words[0]
    best = agent.best_action(query)
    assert best == action_name


# ─── Contract invariants ───────────────────────────────────────────────────


@given(
    role=st.text(min_size=1, max_size=200),
    can=st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=10),
    cannot=st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=10),
    needs=st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=10),
    owns=st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=10),
    collaborates=st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=10),
)
def test_contract_roundtrip_via_as_dict(
    role: str, can: list[str], cannot: list[str],
    needs: list[str], owns: list[str], collaborates: list[str],
):
    c = Contract(role=role, can=tuple(can), cannot=tuple(cannot),
                 needs=tuple(needs), owns=tuple(owns),
                 collaborates=tuple(collaborates))
    d = c.as_dict()
    assert d["role"] == role
    assert d["can"] == list(can)
    assert d["cannot"] == list(cannot)
    assert d["needs"] == list(needs)
    assert d["owns"] == list(owns)
    assert d["collaborates"] == list(collaborates)


def test_all_contracts_have_role_and_limits():
    """Cada contrato debe tener rol no vacío y al menos un cannot."""
    for name, contract in CONTRACTS.items():
        assert contract.role.strip(), f"{name}: role vacío"
        assert contract.cannot, f"{name}: sin cannot (límites)"


def test_contract_owns_no_duplicates():
    """Regla 1 del equipo: ningún recurso puede tener dos dueños."""
    problems = validate_contracts()
    assert not problems, "Contratos incoherentes:\n" + "\n".join(problems)


@given(
    name=st.text(min_size=1, max_size=50),
    agent=st.text(min_size=1, max_size=50),
    action=st.text(min_size=1, max_size=50),
)
def test_agent_result_message_preserved(name: str, agent: str, action: str):
    r = AgentResult(success=True, agent=agent, action=action, message=name)
    assert r.message == name


@given(n=st.integers(min_value=0, max_value=20))
def test_agent_result_warnings_list(n: int):
    warnings = [f"w{i}" for i in range(n)]
    r = AgentResult(success=True, agent="a", action="b",
                    message="m", warnings=warnings)
    assert len(r.warnings) == n
