from __future__ import annotations

import pytest

from agents.agents.data_agent import DataAgent
from agents.config import ProjectConfig
from agents.context import SharedContext


@pytest.fixture
def data_context(tmp_path):
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "interim").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    return SharedContext(root=tmp_path, config=ProjectConfig(project_slug="mi_paquete"))


def test_list_datasets_empty_when_no_files(data_context):
    agent = DataAgent(context=data_context)
    result = agent.list_datasets()
    assert result.success
    assert result.data == []


def test_list_datasets_finds_csv_in_raw(data_context):
    (data_context.root / "data" / "raw" / "dataset.csv").write_text("a,b\n1,2\n")
    agent = DataAgent(context=data_context)
    result = agent.list_datasets()
    assert result.success
    assert len(result.data) == 1
    assert "dataset.csv" in result.data[0]


def test_eda_report_missing_file(data_context):
    agent = DataAgent(context=data_context)
    result = agent.eda_report(filename="no_existe.csv")
    assert not result.success


def test_eda_report_with_csv(data_context):
    csv = data_context.root / "data" / "raw" / "data.csv"
    csv.write_text("x,y,z\n1,2,3\n4,5,6\n7,8,9\n")
    agent = DataAgent(context=data_context)
    result = agent.eda_report(filename="data.csv")
    assert result.success
    assert result.data["summary"]["shape"] == (3, 3)


def test_quality_check_reports_constant_column(data_context):
    csv = data_context.root / "data" / "raw" / "const.csv"
    csv.write_text("a,b\n1,10\n1,20\n1,30\n")
    agent = DataAgent(context=data_context)
    result = agent.quality_check(filename="const.csv")
    assert result.success
    assert any("constante" in w for w in result.warnings)


def test_suggest_imputation_with_nulls(data_context):
    csv = data_context.root / "data" / "raw" / "nulls.csv"
    csv.write_text("num,cat\n1,a\n, b\n3,\n")
    agent = DataAgent(context=data_context)
    result = agent.suggest_imputation(filename="nulls.csv")
    assert result.success
    assert len(result.data["suggestions"]) > 0


def test_detect_skewness(data_context):
    csv = data_context.root / "data" / "raw" / "skew.csv"
    import numpy as np
    rng = np.random.default_rng(42)
    low_skew = rng.normal(0, 1, 100)
    high_skew = rng.exponential(1, 100)
    df_lines = "\n".join(f"{a},{b}" for a, b in zip(low_skew, high_skew))
    csv.write_text(f"normal,skewed\n{df_lines}\n")
    agent = DataAgent(context=data_context)
    result = agent.detect_skewness(filename="skew.csv", threshold=1.0)
    assert result.success
    assert "skewed" in result.data["high_skew"]


def test_statistical_summary(data_context):
    csv = data_context.root / "data" / "raw" / "stats.csv"
    csv.write_text("a,b\n1,10\n2,20\n3,30\n4,40\n5,50\n")
    agent = DataAgent(context=data_context)
    result = agent.statistical_summary(filename="stats.csv")
    assert result.success
    assert result.data["n_numeric"] == 2


def test_eda_report_with_target_detects_no_leakage(data_context):
    csv = data_context.root / "data" / "raw" / "clean.csv"
    csv.write_text("x,y,label\n1,2,0\n3,4,1\n5,6,0\n")
    agent = DataAgent(context=data_context)
    result = agent.eda_report(filename="clean.csv", target_col="label")
    assert result.success
    assert "leakage_suspects" in result.data


def test_detect_leakage_missing_file(data_context):
    agent = DataAgent(context=data_context)
    result = agent.detect_leakage(filename="fake.csv", target_col="y")
    assert not result.success
