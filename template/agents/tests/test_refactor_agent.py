from __future__ import annotations

from agents.agents.refactor_agent import RefactorAgent
from agents.config import ProjectConfig
from agents.context import SharedContext


def _make_agent(tmp_path, slug="mi_paquete"):
    (tmp_path / slug).mkdir()
    cfg = ProjectConfig(project_slug=slug)
    return RefactorAgent(context=SharedContext(root=tmp_path, config=cfg))


def test_fix_mutable_defaults_detects_list(tmp_path):
    agent = _make_agent(tmp_path)
    src = agent.ctx.root / agent.ctx.config.project_slug / "mod.py"
    src.write_text("def f(x=[]):\n    return x\n")
    result = agent.fix_mutable_defaults(dry_run=True)
    assert result.success
    assert len(result.data["changes"]) == 1


def test_fix_mutable_defaults_detects_dict(tmp_path):
    agent = _make_agent(tmp_path)
    src = agent.ctx.root / agent.ctx.config.project_slug / "mod.py"
    src.write_text("def f(x={}):\n    return x\n")
    result = agent.fix_mutable_defaults(dry_run=True)
    assert result.success
    assert len(result.data["changes"]) == 1


def test_fix_mutable_defaults_ignores_none(tmp_path):
    agent = _make_agent(tmp_path)
    src = agent.ctx.root / agent.ctx.config.project_slug / "mod.py"
    src.write_text("def f(x=None):\n    return x\n")
    result = agent.fix_mutable_defaults(dry_run=True)
    assert result.success
    assert len(result.data["changes"]) == 0


def test_fix_mutable_defaults_dry_run_shows_warning(tmp_path):
    agent = _make_agent(tmp_path)
    src = agent.ctx.root / agent.ctx.config.project_slug / "mod.py"
    src.write_text("def f(x=[]):\n    return x\n")
    result = agent.fix_mutable_defaults(dry_run=True)
    assert any("simulación" in w for w in result.warnings)


def test_fix_bare_excepts_detects(tmp_path):
    agent = _make_agent(tmp_path)
    src = agent.ctx.root / agent.ctx.config.project_slug / "mod.py"
    src.write_text("try:\n    1/0\nexcept:\n    pass\n")
    result = agent.fix_bare_excepts(dry_run=False)
    assert result.success
    assert len(result.data["changes"]) == 1
    assert "except Exception:" in src.read_text()


def test_fix_bare_excepts_ignores_specific(tmp_path):
    agent = _make_agent(tmp_path)
    src = agent.ctx.root / agent.ctx.config.project_slug / "mod.py"
    src.write_text("try:\n    1/0\nexcept Exception:\n    pass\n")
    result = agent.fix_bare_excepts(dry_run=True)
    assert result.success
    assert len(result.data["changes"]) == 0


def test_add_type_hints_detects_missing(tmp_path):
    agent = _make_agent(tmp_path)
    src = agent.ctx.root / agent.ctx.config.project_slug / "mod.py"
    src.write_text("def f():\n    return 1\n")
    result = agent.add_type_hints(dry_run=False)
    assert result.success
    assert len(result.data["changes"]) == 1
    assert "-> None" in src.read_text()


def test_add_type_hints_ignores_typed(tmp_path):
    agent = _make_agent(tmp_path)
    src = agent.ctx.root / agent.ctx.config.project_slug / "mod.py"
    src.write_text("def f() -> int:\n    return 1\n")
    result = agent.add_type_hints(dry_run=True)
    assert result.success
    assert len(result.data["changes"]) == 0


def test_fix_weights_only_scans(tmp_path):
    agent = _make_agent(tmp_path)
    src = agent.ctx.root / agent.ctx.config.project_slug / "mod.py"
    src.write_text('torch.load("model.pt", weights_only=False)\n')
    result = agent.fix_weights_only(dry_run=True)
    assert result.success
    assert len(result.data["files"]) == 1


def test_fix_weights_only_clean(tmp_path):
    agent = _make_agent(tmp_path)
    src = agent.ctx.root / agent.ctx.config.project_slug / "mod.py"
    src.write_text('torch.load("model.pt")\n')
    result = agent.fix_weights_only(dry_run=True)
    assert result.success
    assert len(result.data["files"]) == 0


def test_refactor_outside_slug_ignored(tmp_path):
    agent = _make_agent(tmp_path)
    outside = agent.ctx.root / "other_pkg" / "mod.py"
    outside.parent.mkdir()
    outside.write_text("def f(x=[]):\n    return x\n")
    result = agent.fix_mutable_defaults(dry_run=True)
    assert result.success
    assert len(result.data["changes"]) == 0


def test_refactor_within_subdir(tmp_path):
    agent = _make_agent(tmp_path)
    sub = agent.ctx.root / agent.ctx.config.project_slug / "sub"
    sub.mkdir()
    src = sub / "mod.py"
    src.write_text("def f(x=[]):\n    return x\n")
    result = agent.fix_mutable_defaults(within=str(sub.relative_to(agent.ctx.root)), dry_run=True)
    assert result.success
    assert len(result.data["changes"]) == 1
