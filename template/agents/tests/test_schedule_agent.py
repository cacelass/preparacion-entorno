from __future__ import annotations

import pytest

from agents.agents.schedule_agent import ScheduleAgent
from agents.config import ProjectConfig
from agents.context import SharedContext


@pytest.fixture
def schedule_context(tmp_path):
    return SharedContext(root=tmp_path, config=ProjectConfig(project_slug="mi_paquete"))


def test_validate_cron_valid(schedule_context):
    agent = ScheduleAgent(context=schedule_context)
    result = agent.validate_cron(expression="0 9 * * 1-5")
    assert result.success


def test_validate_cron_invalid(schedule_context):
    agent = ScheduleAgent(context=schedule_context)
    result = agent.validate_cron(expression="not-a-cron")
    assert not result.success


def test_to_human(schedule_context):
    agent = ScheduleAgent(context=schedule_context)
    result = agent.to_human(expression="0 6 * * *")
    assert result.success


def test_next_runs(schedule_context):
    agent = ScheduleAgent(context=schedule_context)
    result = agent.next_runs(expression="0 0 * * *", count=3)
    assert result.success
    assert len(result.data) == 3


def test_next_runs_invalid(schedule_context):
    agent = ScheduleAgent(context=schedule_context)
    result = agent.next_runs(expression="bad", count=5)
    assert not result.success
