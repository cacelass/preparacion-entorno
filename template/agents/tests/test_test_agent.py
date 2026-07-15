from __future__ import annotations

from pathlib import Path

import pytest

from agents.agents.test_agent import TestAgent
from agents.tools.pytest_tool import PytestTool, TestRunSummary


def test_list_untested_modules_detects_missing_and_present(context):
    (context.root / "mi_paquete" / "con_test.py").write_text("def f(): return 1\n")
    (context.root / "mi_paquete" / "sin_test.py").write_text("def g(): return 2\n")
    (context.root / "tests" / "test_con_test.py").write_text("from mi_paquete.con_test import f\n")

    agent = TestAgent(context=context)
    result = agent.list_untested_modules()

    assert result.success
    assert result.data == ["sin_test"]


def test_list_untested_modules_excludes_init(context):
    (context.root / "mi_paquete" / "__init__.py").write_text("x = 1\n")
    agent = TestAgent(context=context)
    result = agent.list_untested_modules()
    assert result.success
    assert "__init__" not in result.data


def test_list_untested_modules_missing_package_dir_fails(context):
    import shutil
    shutil.rmtree(context.root / "mi_paquete")
    agent = TestAgent(context=context)
    result = agent.list_untested_modules()
    assert not result.success


def test_coverage_report_without_project_slug_fails(context):
    from agents.config import ProjectConfig
    from agents.context import SharedContext
    ctx_sin_slug = SharedContext(root=context.root, config=ProjectConfig(project_slug=""))
    agent = TestAgent(context=ctx_sin_slug)
    result = agent.coverage_report()
    assert not result.success


# ─── PytestTool.parse_junit_xml edge cases ─────────────────────────────────


def _junit_xml(content: str, tmp_path: Path) -> Path:
    path = tmp_path / "junit.xml"
    path.write_text(content)
    return path


def test_parse_junit_all_passing(tmp_path: Path):
    xml = _junit_xml(
        '<testsuites><testsuite name="pytest" tests="3" failures="0" errors="0" skipped="0" time="0.5">'
        '<testcase classname="test_a" name="test_passes"/></testsuite></testsuites>', tmp_path,
    )
    result = PytestTool.parse_junit_xml(xml)
    assert result.total == 3
    assert result.passed == 3
    assert result.failures == 0


def test_parse_junit_with_failures(tmp_path: Path):
    xml = _junit_xml(
        '<testsuites><testsuite name="pytest" tests="2" failures="1" errors="0" skipped="0" time="0.3">'
        '<testcase classname="test_b" name="test_fails">'
        '<failure message="AssertionError">assert 1 == 2</failure>'
        '</testcase></testsuite></testsuites>', tmp_path,
    )
    result = PytestTool.parse_junit_xml(xml)
    assert result.total == 2
    assert result.failures == 1
    assert result.passed == 1
    assert len(result.failed_tests) == 1
    assert result.failed_tests[0].name == "test_fails"


def test_parse_junit_with_errors(tmp_path: Path):
    xml = _junit_xml(
        '<testsuites><testsuite name="pytest" tests="1" failures="0" errors="1" skipped="0" time="0.1">'
        '<testcase classname="test_c" name="test_errors">'
        '<error message="ImportError">No module named x</error>'
        '</testcase></testsuite></testsuites>', tmp_path,
    )
    result = PytestTool.parse_junit_xml(xml)
    assert result.errors == 1
    assert result.passed == 0


def test_parse_junit_with_skipped(tmp_path: Path):
    xml = _junit_xml(
        '<testsuites><testsuite name="pytest" tests="1" failures="0" errors="0" skipped="1" time="0.0">'
        '<testcase classname="test_d" name="test_skipped">'
        '<skipped message="reason"/>'
        '</testcase></testsuite></testsuites>', tmp_path,
    )
    result = PytestTool.parse_junit_xml(xml)
    assert result.skipped == 1


def test_parse_junit_empty_suite(tmp_path: Path):
    xml = _junit_xml(
        '<testsuites><testsuite name="pytest" tests="0" failures="0" errors="0" skipped="0" time="0.0"/>'
        '</testsuites>', tmp_path,
    )
    result = PytestTool.parse_junit_xml(xml)
    assert result.total == 0
    assert result.passed == 0


def test_parse_junit_malformed_raises(tmp_path: Path):
    xml = _junit_xml("not xml at all", tmp_path)
    with pytest.raises(Exception):
        PytestTool.parse_junit_xml(xml)


def test_parse_junit_no_testsuite_raises(tmp_path: Path):
    xml = _junit_xml("<notests><nothing/></notests>", tmp_path)
    with pytest.raises(ValueError, match="testsuite"):
        PytestTool.parse_junit_xml(xml)


# ─── PytestTool.parse_coverage_json edge cases ─────────────────────────────


def _coverage_json(data: dict, tmp_path: Path) -> Path:
    import json
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(data))
    return path


def test_parse_coverage_all_files(tmp_path: Path):
    cov = _coverage_json({
        "meta": {},
        "files": {
            "src/mod.py": {"summary": {"percent_covered": 85.0, "covered_lines": 17, "num_statements": 20}},
        },
        "totals": {"percent_covered": 85.0},
    }, tmp_path)
    result = PytestTool.parse_coverage_json(cov)
    assert result["total_percent_covered"] == 85.0
    assert "src/mod.py" in result["per_file"]


def test_parse_coverage_zero(tmp_path: Path):
    cov = _coverage_json({
        "meta": {},
        "files": {},
        "totals": {"percent_covered": 0.0},
    }, tmp_path)
    result = PytestTool.parse_coverage_json(cov)
    assert result["total_percent_covered"] == 0.0
    assert result["per_file"] == {}


def test_parse_coverage_one_hundred(tmp_path: Path):
    cov = _coverage_json({
        "meta": {},
        "files": {"src/mod.py": {"summary": {"percent_covered": 100.0, "covered_lines": 10, "num_statements": 10}}},
        "totals": {"percent_covered": 100.0},
    }, tmp_path)
    result = PytestTool.parse_coverage_json(cov)
    assert result["total_percent_covered"] == 100.0


def test_parse_coverage_missing_totals(tmp_path: Path):
    cov = _coverage_json({"meta": {}, "files": {}, "totals": {}}, tmp_path)
    result = PytestTool.parse_coverage_json(cov)
    assert result["total_percent_covered"] is None


# ─── generate_test_skeletons: edge cases ───────────────────────────────────


def test_generate_skeleton_existing_test_skips(context):
    (context.root / "mi_paquete" / "modulo.py").write_text("def f(): return 1\n")
    (context.root / "tests" / "test_modulo.py").write_text("from mi_paquete.modulo import f\n")
    agent = TestAgent(context=context)
    result = agent.generate_test_skeletons()
    assert result.success
    assert "modulo" not in result.data


def test_generate_skeleton_no_package_dir_empty_result(context):
    import shutil
    shutil.rmtree(context.root / "mi_paquete")
    agent = TestAgent(context=context)
    result = agent.generate_test_skeletons()
    assert result.success
    assert result.data == [] or result.data.get("created") == []


def test_generate_skeleton_empty_package_empty_result(context):
    (context.root / "mi_paquete" / "__init__.py").write_text("")
    agent = TestAgent(context=context)
    result = agent.generate_test_skeletons()
    assert result.success
    assert result.data == [] or result.data.get("created") == []


def test_generate_skeleton_creates_files_for_untested(context):
    (context.root / "mi_paquete" / "servicio.py").write_text("def serve(): return 42\n")
    agent = TestAgent(context=context)
    result = agent.generate_test_skeletons()
    assert result.success
    assert len(result.data.get("created", [])) >= 1
    created = " ".join(result.data["created"])
    assert "test_servicio" in created
