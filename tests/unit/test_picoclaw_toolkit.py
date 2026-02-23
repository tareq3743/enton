"""Testes para PicoClawTools — integração com PicoClaw CLI."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from enton.skills.picoclaw_toolkit import (
    PicoClawTools,
)


@pytest.fixture
def tools():
    return PicoClawTools(timeout=10)


@pytest.fixture
def mock_cron(tmp_path, monkeypatch):
    """Cria cron jobs temporário pra testes."""
    cron_dir = tmp_path / "cron"
    cron_dir.mkdir()
    jobs_file = cron_dir / "jobs.json"
    jobs_file.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "abc123",
                        "name": "gpu-check",
                        "enabled": True,
                        "schedule": {"kind": "every", "everyMs": 1800000},
                        "payload": {
                            "kind": "agent_turn",
                            "message": "Check GPU temp",
                        },
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        "enton.skills.picoclaw_toolkit.CRON_JOBS_FILE", jobs_file
    )
    return jobs_file


class TestPicoClawTools:
    def test_init(self, tools):
        assert tools.name == "picoclaw_tools"
        assert len(tools.get_async_functions()) == 6

    async def test_pico_run_success(self, tools):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"Hello from PicoClaw!", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await tools.pico_run("ola mundo")

        assert "Hello from PicoClaw!" in result

    async def test_pico_run_error(self, tools):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"model not found")
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await tools.pico_run("test")

        assert "[ERRO" in result
        assert "model not found" in result

    async def test_pico_run_binary_not_found(self, tools):
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("No such file"),
        ):
            result = await tools.pico_run("test")

        assert "[ERRO]" in result
        assert "não encontrado" in result

    async def test_pico_cron_list(self, tools, mock_cron):
        result = await tools.pico_cron_list()
        assert "gpu-check" in result
        assert "30min" in result

    async def test_pico_cron_list_empty(self, tools, tmp_path, monkeypatch):
        empty_file = tmp_path / "empty.json"
        empty_file.write_text('{"jobs": []}')
        monkeypatch.setattr(
            "enton.skills.picoclaw_toolkit.CRON_JOBS_FILE", empty_file
        )
        result = await tools.pico_cron_list()
        assert "Nenhum" in result

    async def test_pico_cron_add(self, tools, mock_cron):
        result = await tools.pico_cron_add(
            name="disk-check",
            message="Check disk space",
            interval_minutes=60,
        )
        assert "disk-check" in result
        assert "60min" in result

        # Verifica que foi salvo
        data = json.loads(mock_cron.read_text())
        names = [j["name"] for j in data["jobs"]]
        assert "disk-check" in names
        assert "gpu-check" in names  # original ainda existe

    async def test_pico_cron_add_duplicate(self, tools, mock_cron):
        result = await tools.pico_cron_add(
            name="gpu-check", message="Duplicate"
        )
        assert "já existe" in result

    async def test_pico_cron_remove(self, tools, mock_cron):
        result = await tools.pico_cron_remove("gpu-check")
        assert "removido" in result

        data = json.loads(mock_cron.read_text())
        assert len(data["jobs"]) == 0

    async def test_pico_cron_remove_not_found(self, tools, mock_cron):
        result = await tools.pico_cron_remove("nao-existe")
        assert "não encontrado" in result

    async def test_pico_memory(self, tools, tmp_path, monkeypatch):
        ws = tmp_path / "workspace"
        ws.mkdir()
        mem_file = ws / "MEMORY.md"
        mem_file.write_text("# Memória\nAprendizado importante")
        monkeypatch.setattr(
            "enton.skills.picoclaw_toolkit.PICOCLAW_WORKSPACE", ws
        )
        result = await tools.pico_memory()
        assert "Aprendizado importante" in result

    async def test_pico_memory_not_found(self, tools, tmp_path, monkeypatch):
        ws = tmp_path / "empty_ws"
        ws.mkdir()
        monkeypatch.setattr(
            "enton.skills.picoclaw_toolkit.PICOCLAW_WORKSPACE", ws
        )
        result = await tools.pico_memory()
        assert "não tem memória" in result

    async def test_pico_status(self, tools, tmp_path, monkeypatch):
        # Mock paths
        monkeypatch.setattr(
            "enton.skills.picoclaw_toolkit.PICOCLAW_WORKSPACE", tmp_path
        )
        monkeypatch.setattr(
            "enton.skills.picoclaw_toolkit.CRON_JOBS_FILE",
            tmp_path / "cron" / "jobs.json",
        )
        result = await tools.pico_status()
        assert "PicoClaw Status" in result
