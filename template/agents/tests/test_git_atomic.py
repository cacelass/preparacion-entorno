"""
Tests del commit atómico (`git commit_atomic`).

Cierran el contrato de OMP-001: agrupar cambios no relacionados en commits
atómicos por área (código antes que tests, tests antes que docs), excluir los
lock files, validar mensajes Conventional, rechazar ciclos de dependencias y
pasar por la puerta de permisos.
"""

from __future__ import annotations

import subprocess

from agents.agents.git_agent import GitAgent
from agents.tools.git_tool import GitTool


# -- planificación (GitTool.plan_atomic) --------------------------------------

def test_plan_atomic_agrupa_por_area_y_ordena(tmp_path):
    plan = GitTool.plan_atomic(
        ["mi_paquete/nuevo.py", "tests/test_nuevo.py", "docs/nota.md", "uv.lock"],
        tmp_path,
    )
    areas = [g["area"] for g in plan["groups"]]
    assert areas == ["code", "test", "docs"], "código antes que tests, tests antes que docs"
    assert [g["type"] for g in plan["groups"]] == ["feat", "test", "docs"]
    assert plan["excluded"] == ["uv.lock"], "el lock file no entra en ningún grupo"
    assert plan["cycle"] is None


def test_plan_atomic_excluye_todos_los_lock_files(tmp_path):
    plan = GitTool.plan_atomic(["uv.lock", "package-lock.json"], tmp_path)
    assert plan["groups"] == []
    assert len(plan["excluded"]) == 2


def test_plan_atomic_solo_lock_files_no_genera_grupos_con_pyproject(tmp_path):
    plan = GitTool.plan_atomic(["pyproject.toml"], tmp_path)
    assert [g["area"] for g in plan["groups"]] == ["build"]


def test_plan_atomic_detecta_ciclo(tmp_path):
    (tmp_path / "mi_paquete").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "mi_paquete" / "a.py").write_text("from tests.test_a import helper\n", encoding="utf-8")
    (tmp_path / "tests" / "test_a.py").write_text("import mi_paquete.a\n", encoding="utf-8")

    plan = GitTool.plan_atomic(["mi_paquete/a.py", "tests/test_a.py"], tmp_path)
    assert plan["cycle"] is not None, "código que importa de tests debe rechazarse"
    assert "depende" in plan["cycle"]


def test_plan_atomic_no_detecta_ciclo_cuando_test_depende_de_codigo(tmp_path):
    (tmp_path / "mi_paquete").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "mi_paquete" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "tests" / "test_a.py").write_text("import mi_paquete.a\n", encoding="utf-8")

    plan = GitTool.plan_atomic(["mi_paquete/a.py", "tests/test_a.py"], tmp_path)
    assert plan["cycle"] is None, "tests que importan código es el orden correcto"


# -- agente --------------------------------------------------------------------

def test_commit_atomic_dry_run_propone_sin_escribir(context):
    (context.root / "mi_paquete" / "nuevo.py").write_text("x = 1\n")
    (context.root / "tests" / "test_nuevo.py").write_text("def test_x():\n    assert 1\n")
    subprocess.run(["git", "add", "."], cwd=context.root, check=True)

    result = GitAgent(context=context).commit_atomic(dry_run=True)
    assert result.success
    areas = [g["area"] for g in result.data["groups"]]
    assert "code" in areas and "test" in areas
    # dry_run no escribe: no hay commits nuevos
    log = subprocess.run(["git", "log", "--oneline"], cwd=context.root, capture_output=True, text=True)
    assert log.stdout.strip().count("\n") == 0, "el dry-run no debe commitear"


def test_commit_atomic_ejecuta_grupos_en_orden(context):
    (context.root / "mi_paquete" / "nuevo.py").write_text("x = 1\n")
    (context.root / "tests" / "test_nuevo.py").write_text("def test_x():\n    assert 1\n")
    (context.root / "docs").mkdir(exist_ok=True)
    (context.root / "docs" / "nota.md").write_text("# Nota\n")
    subprocess.run(["git", "add", "."], cwd=context.root, check=True)

    result = GitAgent(context=context).commit_atomic(
        subjects="feat: añade nuevo.py; test: cubre nuevo.py; docs: documenta nuevo.py"
    )
    assert result.success
    assert len(result.data["created"]) == 3

    log = subprocess.run(["git", "log", "--pretty=%s"], cwd=context.root, capture_output=True, text=True)
    subjects = log.stdout.strip().splitlines()
    assert subjects[0] == "docs: documenta nuevo.py", "el último commit es el primero del log"
    # el orden temporal fue code -> test -> docs; el log lo muestra al revés:
    # docs (más nuevo) antes que test, test antes que feat (más viejo)
    assert subjects.index("docs: documenta nuevo.py") < subjects.index("test: cubre nuevo.py")
    assert subjects.index("test: cubre nuevo.py") < subjects.index("feat: añade nuevo.py")


def test_commit_atomic_rechaza_ciclo(context):
    (context.root / "mi_paquete" / "a.py").write_text("from tests.test_a import helper\n")
    (context.root / "tests" / "test_a.py").write_text("import mi_paquete.a\n")
    subprocess.run(["git", "add", "."], cwd=context.root, check=True)

    result = GitAgent(context=context).commit_atomic(dry_run=True)
    assert not result.success
    assert "rechazado" in result.message


def test_commit_atomic_valida_mensajes_conventional(context):
    (context.root / "mi_paquete" / "nuevo.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=context.root, check=True)

    result = GitAgent(context=context).commit_atomic(subjects="esto no es conventional")
    assert not result.success
    assert result.needs, "debe pedir un mensaje Conventional válido"


def test_commit_atomic_es_destructiva_y_pide_confirmacion(context):
    (context.root / "mi_paquete" / "nuevo.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=context.root, check=True)

    result = GitAgent(context=context).run("commit_atomic")
    assert not result.success
    assert result.needs, "escribir en el historial git exige confirmación explícita"
