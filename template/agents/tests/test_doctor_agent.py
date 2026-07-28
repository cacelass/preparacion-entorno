from __future__ import annotations

from pathlib import Path

import pytest

from agents.agents.doctor_agent import DoctorAgent, _satisfies_requires_python
from agents.config import ProjectConfig
from agents.context import SharedContext


@pytest.fixture
def doctor_context(tmp_path):
    (tmp_path / "mi_paquete" / "utils").mkdir(parents=True)
    (tmp_path / "mi_paquete" / "utils" / "__init__.py").write_text("")
    (tmp_path / "tests").mkdir()
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "mi_paquete"\nrequires-python = ">=3.10"\n'
    )
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


def test_checkup_all_ok_on_healthy_project(doctor_context):
    (doctor_context.root / "mi_paquete" / "__init__.py").write_text("")
    agent = DoctorAgent(context=doctor_context)
    result = agent.checkup()
    assert isinstance(result.data, dict)
    assert "python" in result.data
    assert "structure" in result.data
    assert result.data["structure"]["ok"]


def test_checkup_missing_pyproject(tmp_path):
    (tmp_path / "pkg").mkdir()
    ctx = SharedContext(root=tmp_path, config=ProjectConfig(project_slug="pkg"))
    agent = DoctorAgent(context=ctx)
    result = agent.checkup()
    assert not result.data["python"]["ok"]


def test_checkup_no_project_slug_fails(tmp_path):
    ctx = SharedContext(root=tmp_path, config=ProjectConfig(project_slug=""))
    agent = DoctorAgent(context=ctx)
    result = agent.checkup()
    assert not result.data["project_config"]["ok"]


def test_checkup_missing_package_init(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "pkg"\nrequires-python = ">=3.10"\n')
    ctx = SharedContext(root=tmp_path, config=ProjectConfig(project_slug="pkg"))
    agent = DoctorAgent(context=ctx)
    result = agent.checkup()
    assert not result.data["structure"]["ok"]


def test_checkup_no_tests_dir_is_ok(tmp_path):
    ctx = SharedContext(root=tmp_path, config=ProjectConfig(project_slug="pkg"))
    agent = DoctorAgent(context=ctx)
    result = agent.checkup()
    assert result.data["tests"]["ok"]


def test_checkup_tests_dir_empty_is_fail(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "pkg"\nrequires-python = ">=3.10"\n')
    ctx = SharedContext(root=tmp_path, config=ProjectConfig(project_slug="pkg"))
    agent = DoctorAgent(context=ctx)
    result = agent.checkup()
    assert not result.data["tests"]["ok"]
    assert "no contiene tests" in result.data["tests"]["message"]


def test_disk_usage_nonexistent_dirs(tmp_path):
    ctx = SharedContext(root=tmp_path, config=ProjectConfig(project_slug="pkg"))
    agent = DoctorAgent(context=ctx)
    result = agent.disk_usage()
    assert result.success
    assert "no existe" in str(result.data)


def test_disk_usage_with_files(tmp_path):
    ctx = SharedContext(root=tmp_path, config=ProjectConfig(project_slug="pkg"))
    (ctx.data_dir / "raw").mkdir(parents=True)
    (ctx.data_dir / "raw" / "dataset.csv").write_text("a,b,c\n1,2,3\n")
    agent = DoctorAgent(context=ctx)
    result = agent.disk_usage()
    assert "no existe" not in result.data.get("data", "")


def test_summary_with_minimal_project(doctor_context):
    agent = DoctorAgent(context=doctor_context)
    result = agent.summary()
    assert result.success
    assert "mi_paquete" in result.data["project"]
    assert result.data["ml_type"] == "supervisado"


def test_human_size_bytes():
    assert DoctorAgent._human_size(0) == "0.0 B"


def test_human_size_kb():
    assert DoctorAgent._human_size(1500) == "1.5 KB"


def test_human_size_mb():
    assert DoctorAgent._human_size(2_500_000) == "2.4 MB"


def test_human_size_gb():
    assert DoctorAgent._human_size(2_500_000_000) == "2.3 GB"


# -- requires-python: comparar versiones, no subcadenas -------------------------


@pytest.mark.parametrize(
    "current,requires,esperado",
    [
        # El bug original: 3.13 con ">=3.12" daba FALLO por comparar subcadenas.
        ((3, 13), ">=3.12", True),
        ((3, 12), ">=3.12", True),
        ((3, 11), ">=3.12", False),
        # ...y el falso positivo simétrico: "3.1" es subcadena de ">=3.12".
        ((3, 1), ">=3.12", False),
        ((3, 13), ">=3.10,<4.0", True),
        ((4, 0), ">=3.10,<4.0", False),
        ((3, 12), "==3.12", True),
        ((3, 13), "==3.12", False),
        ((3, 12), "~=3.11", True),
        ((4, 0), "~=3.11", False),
        ((3, 13), "", True),
        ((3, 13), "vete a saber", True),  # no se entiende → no se avisa en falso
    ],
)
def test_requires_python_compara_versiones(current, requires, esperado):
    assert _satisfies_requires_python(current, requires) is esperado
