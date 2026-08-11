"""Tests del PlanAgent: encargo → preguntas → delegación, sin inventar nada."""

from __future__ import annotations

from agents.agents.plan_agent import PlanAgent
from agents.orchestrator import Orchestrator


def _write_pyproject(project_root):
    (project_root / "pyproject.toml").write_text(
        '[project]\nname = "mi_paquete"\nversion = "0.1.0"\nrequires-python = ">=3.10"\n'
    )


def test_intake_asks_for_missing_args_and_execute_refuses(context):
    """
    Un paso cuya acción necesita un argumento obligatorio (tag_release →
    version) debe convertirse en pregunta, y execute debe NEGARSE mientras
    haya preguntas — pedir información, nunca adivinarla.
    """
    plan = PlanAgent(context=context)

    result = plan.intake(brief="haz un tag release del proyecto git")
    assert result.success
    assert result.data["steps"], f"El intake no asignó ningún paso: {result.message}"
    step = result.data["steps"][0]
    assert step["agent"] == "git"
    assert "version" in step["missing"]
    assert result.needs, "Debería haber preguntas pendientes (falta 'version')"

    order_id = result.data["id"]
    refused = plan.execute(order=order_id)
    assert not refused.success
    assert refused.needs, "execute debe devolver las preguntas pendientes, no ejecutar con huecos"

    answered = plan.answer(order=order_id, step0_version="9.9.9")
    assert answered.success
    assert not answered.needs, f"No deberían quedar preguntas: {answered.needs}"


def test_intake_execute_happy_path_delegates_and_records(context, project_root):
    """Encargo sin huecos: se ejecuta delegando en el agente dueño y queda auditado."""
    _write_pyproject(project_root)
    plan = PlanAgent(context=context)

    result = plan.intake(brief="verifica la python version del entorno")
    assert result.success
    assert result.data["steps"][0]["agent"] == "env"
    assert not result.needs, f"No debería preguntar nada: {result.needs}"

    executed = plan.execute(order=result.data["id"])
    assert executed.success, executed.message
    assert executed.data["status"] == "completado"
    assert executed.data["results"][0]["agent"] == "env"

    # La delegación pasó por run() → quedó en el log de auditoría.
    from agents import audit
    entries = audit.read_entries(context)
    assert any(e["agent"] == "env" and e["action"] == "check_python_version" for e in entries)


def test_status_lists_orders(context):
    plan = PlanAgent(context=context)
    plan.intake(brief="verifica la python version del entorno")
    listing = plan.status()
    assert listing.success
    assert len(listing.data) == 1


def test_plan_routes_via_orchestrator(context):
    """'planificar' y compañía rutean al agente plan, no a otro."""
    orchestrator = Orchestrator(context=context)
    decision = orchestrator.select_agent("planifica este encargo")
    assert decision.agent_name == "plan"


def test_scope_routes_via_orchestrator(context):
    """'scope' / 'objetivo' / 'empezar el proyecto' rutean al agente plan."""
    orchestrator = Orchestrator(context=context)
    assert orchestrator.select_agent("scope del proyecto").agent_name == "plan"
    assert orchestrator.select_agent("empezar el proyecto").agent_name == "plan"


def test_scope_pregunta_todo_lo_necesario(context):
    plan = PlanAgent(context=context)
    result = plan.scope(reset=True)
    assert result.success
    # 4 obligatorias (pregunta, metrica, datos, parada) + 3 opcionales
    assert len(result.needs) == 7
    claves = {n.split(":")[0] for n in result.needs}
    assert {"pregunta", "metrica", "datos", "parada", "usuarios", "alcance", "riesgos"} == claves


def test_scope_rechaza_metrica_sin_umbral(context):
    plan = PlanAgent(context=context)
    plan.scope(reset=True)
    result = plan.scope_answer(metrica="que funcione bien")
    assert not result.success
    assert "umbral" in result.message or "número" in result.message


def test_scope_commit_rehusa_sin_obligatorias(context):
    plan = PlanAgent(context=context)
    plan.scope(reset=True)
    plan.scope_answer(pregunta="¿Q?", metrica="AUC >= 0.85")
    # faltan datos y parada (obligatorias) → el commit se niega, no siembra a medias
    result = plan.scope_commit()
    assert not result.success
    assert result.needs


def test_scope_commit_escribe_spec_y_siembra_backlog(context, project_root):
    (context.root / "harness").mkdir(exist_ok=True)
    (context.root / "harness" / "featureslist.json").write_text(
        '{"version": 1, "project": "T", "features": []}', encoding="utf-8"
    )
    plan = PlanAgent(context=context)
    plan.scope(reset=True)
    answered = plan.scope_answer(
        pregunta="¿Podemos predecir churn?",
        metrica="AUC >= 0.85 en validación",
        datos="clientes.csv con 100k filas",
        parada="Si AUC no supera 0.60 tras el baseline",
        features="API-001",
    )
    assert answered.success, answered.message

    result = plan.scope_commit()
    assert result.success, result.message
    sembradas = result.data["sembradas"]
    assert "SCOPE-001" in sembradas and "MODEL-001" in sembradas
    assert "API-001" in sembradas

    # El spec queda escrito con los apartados y la métrica numérica
    objetivo = (context.root / "references" / "00-objetivo.md").read_text()
    assert "AUC >= 0.85" in objetivo
    assert "Podemos predecir churn" in objetivo

    # El orden lógico: la dirección antes que las features propuestas
    import json
    doc = json.loads((context.root / "harness" / "featureslist.json").read_text())
    ids = [f["id"] for f in doc["features"]]
    assert ids.index("SCOPE-001") < ids.index("MODEL-001") < ids.index("API-001")


def test_scope_sembrado_idempotente_no_duplica(context):
    """Volver a sembrar no duplica: las features ya existentes se saltan."""
    (context.root / "harness").mkdir(exist_ok=True)
    (context.root / "harness" / "featureslist.json").write_text(
        '{"version": 1, "project": "T", "features": []}', encoding="utf-8"
    )
    plan = PlanAgent(context=context)
    plan.scope(reset=True)
    plan.scope_answer(pregunta="Q", metrica="F1 >= 0.8", datos="d", parada="p")

    r1 = plan.scope_commit()
    r2 = plan.scope_commit()
    assert r1.success and r2.success
    assert "SCOPE-001" not in r2.data["sembradas"], "las ya existentes no se re-añaden"


def test_scope_features_se_deduprican(context):
    plan = PlanAgent(context=context)
    plan.scope(reset=True)
    plan.scope_answer(pregunta="Q", metrica="F1 >= 0.8", datos="d", parada="p",
                      features="API-001; MON-001")
    plan.scope_answer(features="API-001; OTRA")
    scope = plan._load_scope()
    assert scope["features"] == ["API-001", "MON-001", "OTRA"]


def test_detectar_riesgos_reconoce_login(context):
    """La heurística identifica los riesgos de un login sin que el usuario los declare."""
    from agents.agents.plan_agent import _detectar_riesgos

    riesgos = _detectar_riesgos("hacer un login que distinga por usuario con contraseña")
    assert "sql injection" in riesgos
    assert "fuga de credenciales" in riesgos


def test_detectar_riesgos_no_alarma_texto_inocuo(context):
    from agents.agents.plan_agent import _detectar_riesgos

    assert _detectar_riesgos("predecir la temperatura media anual") == []


def test_scope_commit_rehusa_si_riesgos_detectados_sin_decidir(context):
    """El gate: si la heurística detecta riesgos y no se han decidido, NO siembra."""
    (context.root / "harness").mkdir(exist_ok=True)
    (context.root / "harness" / "featureslist.json").write_text(
        '{"version": 1, "project": "T", "features": []}', encoding="utf-8"
    )
    plan = PlanAgent(context=context)
    plan.scope(reset=True)
    plan.scope_answer(pregunta="¿Cómo autenticar usuarios en el login?",
                      metrica="tasa >= 0.99", datos="tabla de usuarios", parada="p")

    result = plan.scope_commit()
    assert not result.success, "no puede sembrar con riesgos sin decidir"
    assert result.needs, "debe pedir decisión por cada riesgo detectado"
    assert any("sql injection" in n for n in result.needs)

    # Nada se sembró
    import json
    doc = json.loads((context.root / "harness" / "featureslist.json").read_text())
    assert doc["features"] == []


def test_scope_commit_siembra_solo_riesgos_aceptados(context):
    """Tras decidir, se siembran los aceptados como RISK y los descartados no."""
    (context.root / "harness").mkdir(exist_ok=True)
    (context.root / "harness" / "featureslist.json").write_text(
        '{"version": 1, "project": "T", "features": []}', encoding="utf-8"
    )
    plan = PlanAgent(context=context)
    plan.scope(reset=True)
    plan.scope_answer(pregunta="¿Cómo autenticar usuarios en el login?",
                      metrica="tasa >= 0.99", datos="tabla de usuarios", parada="p")

    plan.scope_answer(aceptar_riesgos="sql injection")
    plan.scope_answer(descartar_riesgos="enumeración de usuarios")
    # fuga de credenciales también quedó detectada → hay que decidirla también
    result = plan.scope_commit()
    assert not result.success, "fuga de credenciales sigue sin decidir"
    plan.scope_answer(aceptar_riesgos="fuga de credenciales")

    result = plan.scope_commit()
    assert result.success, result.message
    import json
    doc = json.loads((context.root / "harness" / "featureslist.json").read_text())
    titulos = [f["title"] for f in doc["features"] if f["id"].startswith("RISK")]
    assert any("sql injection" in t for t in titulos)
    assert any("fuga de credenciales" in t for t in titulos)
    assert not any("enumeración" in t for t in titulos), "el descartado NO se siembra"


def test_scope_riesgos_declarados_no_se_preguntan_de_nuevo(context):
    """Si el usuario ya declaró un riesgo, la heurística no lo vuelve a pedir."""
    plan = PlanAgent(context=context)
    plan.scope(reset=True)
    plan.scope_answer(pregunta="login con contraseña",
                      metrica="tasa >= 0.99", datos="d", parada="p",
                      riesgos="sql injection")
    pendientes = plan._pendientes_riesgo(plan._load_scope())
    assert "sql injection" not in pendientes, "ya declarado, no se vuelve a preguntar"
