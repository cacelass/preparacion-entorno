from __future__ import annotations

import pytest

from agents.agents.doctor_agent import DoctorAgent
from agents.config import ProjectConfig
from agents.context import SharedContext


@pytest.fixture
def doctor_context(tmp_path):
    (tmp_path / "mi_paquete" / "utils").mkdir(parents=True)
    (tmp_path / "mi_paquete" / "utils" / "__init__.py").write_text("")
    (tmp_path / "tests").mkdir()
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "mi_paquete"\nrequires-python = ">=3.10"\n')
    return SharedContext(root=tmp_path, config=ProjectConfig(project_slug="mi_paquete"))


def test_checkup_returns_all_sections(doctor_context):
    agent = DoctorAgent(context=doctor_context)
    result = agent.checkup()
    assert isinstance(result.data, dict)
    assert "python" in result.data
    assert "structure" in result.data


def test_disk_usage_reports(doctor_context):
    agent = DoctorAgent(context=doctor_context)
    result = agent.disk_usage()
    assert result.success
