"""
test_mutation_agent.py — Tests del agente `mutation`.

Se prueba la lógica pura (parseo del informe, fórmula CRAP) sin depender de
subprocesos; las acciones que ejecutan tools/mutate.py y radon se comprueban
en su rama de fallo (target ausente, script ausente) que es la que no necesita
entorno.
"""

from __future__ import annotations


import pytest

from agents.agents.mutation_agent import MutationAgent, crap_value

REPORT_OK = """Mutación de mathx.py:
  4 sitio(s) · killed 3 · survived 1 · timeout 0
  Score de mutación: 75.0%
    ✔ killed     Lt
    ✔ killed     Gt
    ✘ survived   Eq
"""

REPORT_PERFECT = """Mutación de mathx.py:
  4 sitio(s) · killed 4 · survived 0 · timeout 0
  Score de mutación: 100.0%
    ✔ killed     Lt
    ✔ killed     Gt
    ✔ killed     Eq
    ✔ killed     True
"""


@pytest.fixture
def mutation(context) -> MutationAgent:
    return MutationAgent(context=context)


# -- CRAP (fórmula pura) -----------------------------------------------------
def test_crap_con_100_por_ciento_cobertura_es_la_complejidad():
    assert crap_value(5, 100.0) == 5.0


def test_crap_con_cero_cobertura_crece_con_la_complejidad():
    assert crap_value(2, 0.0) == 2.0**2 * 1 + 2
    assert crap_value(3, 0.0) == 3.0**2 * 1 + 3


def test_crap_aumenta_al_bajar_cobertura():
    assert crap_value(5, 50.0) > crap_value(5, 100.0)


def test_crap_umbral_detecta_complejidad_sin_cobertura():
    # cc=6, coverage=0: 6^2 + 6 = 42 > 30
    assert crap_value(6, 0.0) > 30.0


def test_crap_alta_cobertura_absorbe_complejidad_media():
    # cc=5, coverage=100: crap=5, por debajo del umbral
    assert crap_value(5, 100.0) < 30.0


# -- parseo del informe del mutador -------------------------------------------
def test_parse_report_lee_killed_survived_y_score(mutation):
    report = mutation._parse_report(REPORT_OK)
    assert report["killed"] == 3
    assert report["survived"] == 1
    assert report["timeout"] == 0
    assert report["score"] == 75.0
    assert report["total"] == 4


def test_parse_report_detalla_los_sitios(mutation):
    report = mutation._parse_report(REPORT_OK)
    statuses = {d["site"]: d["status"] for d in report["detail"]}
    assert statuses["Lt"] == "killed"
    assert statuses["Eq"] == "survived"


def test_parse_report_perfecto(mutation):
    report = mutation._parse_report(REPORT_PERFECT)
    assert report["killed"] == 4
    assert report["survived"] == 0
    assert report["score"] == 100.0


def test_parse_report_con_salida_rara_devuelve_none(mutation):
    assert mutation._parse_report("esto no es un informe") is None


# -- rama de fallo: target ----------------------------------------------------
def test_run_mutation_sin_target_pide_target(mutation):
    result = mutation.run_mutation_testing(target="")
    assert not result.success
    assert result.needs


def test_run_mutation_target_inexistente_falla(mutation):
    result = mutation.run_mutation_testing(target="no_existe.py")
    assert not result.success
    assert "no_existe.py" in result.message


def test_run_mutation_target_fuera_del_proyecto_falla(mutation, tmp_path):
    fuera = tmp_path.parent / "fuera.py"
    fuera.write_text("x = 1\n")
    result = mutation.run_mutation_testing(target=str(fuera))
    assert not result.success
    assert "fuera del proyecto" in result.message


def test_crap_sin_target_pide_target(mutation):
    result = mutation.crap_report(target="")
    assert not result.success
    assert result.needs


def test_crap_target_inexistente_falla(mutation):
    result = mutation.crap_report(target="no_existe.py")
    assert not result.success
    assert "no_existe.py" in result.message


# -- rama de fallo: script del mutador ausente ---------------------------------
def test_run_mutation_sin_tools_mutate_avisa(mutation):
    (mutation.ctx.root / "modulo.py").write_text("def f(): return 1\n")
    result = mutation.run_mutation_testing(target="modulo.py")
    assert not result.success
    assert "tools/mutate.py" in result.message or "use_sdd" in result.message
