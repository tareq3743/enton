"""PicoClaw Toolkit — Integração com o agente PicoClaw (Go) via CLI.

PicoClaw é um agente AI ultra-leve em Go (<10MB RAM, boot <1s) que roda
no gato-pc com modelos locais (Ollama) e cloud (Gemini). Tem cron jobs,
web search, file ops, shell exec — tudo num binário Go.

Esta toolkit permite que o Enton delegue tarefas pro PicoClaw como um
sub-agente leve e rápido, especialmente pra:
- Tarefas assíncronas de longa duração (cron jobs, monitoramento)
- Operações em background enquanto Enton faz outra coisa
- Execução rápida sem overhead do Python
- Agendamento de tarefas recorrentes
"""

import asyncio
import json
import logging
from pathlib import Path

from agno.tools import Toolkit

logger = logging.getLogger(__name__)

PICOCLAW_BIN = "/home/gabriel-maia/.local/bin/picoclaw"
PICOCLAW_WORKSPACE = Path.home() / ".picoclaw" / "workspace"
PICOCLAW_CONFIG = Path.home() / ".picoclaw" / "config.json"
CRON_JOBS_FILE = PICOCLAW_WORKSPACE / "cron" / "jobs.json"


class PicoClawTools(Toolkit):
    """Integração com PicoClaw — agente AI leve em Go para tarefas rápidas e cron."""

    def __init__(self, timeout: int = 120):
        super().__init__(name="picoclaw_tools")
        self._timeout = timeout
        self.register(self.pico_run)
        self.register(self.pico_cron_list)
        self.register(self.pico_cron_add)
        self.register(self.pico_cron_remove)
        self.register(self.pico_memory)
        self.register(self.pico_status)

    async def _exec(self, *args: str, timeout: int | None = None) -> str:
        """Executa o binário picoclaw com os args dados."""
        cmd = [PICOCLAW_BIN, *args]
        t = timeout or self._timeout
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(PICOCLAW_WORKSPACE),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=t
            )
            output = stdout.decode("utf-8", errors="replace").strip()
            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace").strip()
                return f"[ERRO exit={proc.returncode}] {err}\n{output}"
            return output or "(sem output)"
        except TimeoutError:
            proc.kill()
            return f"[TIMEOUT] PicoClaw não respondeu em {t}s"
        except FileNotFoundError:
            return f"[ERRO] Binário não encontrado: {PICOCLAW_BIN}"
        except Exception as e:
            return f"[ERRO] {type(e).__name__}: {e}"

    async def pico_run(self, prompt: str, model: str = "") -> str:
        """Executa uma tarefa no PicoClaw (agente Go ultra-leve, <10MB RAM).

        Ideal pra tarefas que não precisam de visão/voz — file ops, shell,
        web search, coding rápido. Roda local com qwen3-coder ou cloud.

        Args:
            prompt: A tarefa para o PicoClaw executar.
            model: Modelo opcional (qwen3-coder-local, devstral-local,
                   gemini-2.5-pro, gemini-2.5-flash). Vazio = default.
        """
        args = ["agent", "-m", prompt]
        if model:
            args.extend(["--model", model])
        logger.info("PicoClaw run: %s (model=%s)", prompt[:80], model or "default")
        return await self._exec(*args, timeout=self._timeout)

    async def pico_cron_list(self) -> str:
        """Lista todos os cron jobs agendados no PicoClaw.

        Mostra jobs de monitoramento (GPU, disco, RAM) e tarefas recorrentes.
        """
        if not CRON_JOBS_FILE.exists():
            return "Nenhum cron job configurado."

        try:
            data = json.loads(CRON_JOBS_FILE.read_text("utf-8"))
            jobs = data.get("jobs", [])
            if not jobs:
                return "Nenhum cron job ativo."

            lines = [f"Cron jobs PicoClaw ({len(jobs)}):"]
            for job in jobs:
                name = job.get("name", job.get("id", "?"))
                enabled = "ON" if job.get("enabled", True) else "OFF"
                sched = job.get("schedule", {})
                kind = sched.get("kind", "?")
                if kind == "every":
                    ms = sched.get("everyMs", 0)
                    interval = f"cada {ms // 60000}min"
                elif kind == "cron":
                    interval = sched.get("expr", "?")
                else:
                    interval = str(sched)
                msg = job.get("payload", {}).get("message", "")[:60]
                lines.append(f"  [{enabled}] {name}: {interval} — {msg}")
            return "\n".join(lines)
        except Exception as e:
            return f"[ERRO] Lendo cron jobs: {e}"

    async def pico_cron_add(
        self, name: str, message: str, interval_minutes: int = 30
    ) -> str:
        """Agenda um novo cron job no PicoClaw.

        O job roda periodicamente — o PicoClaw executa a mensagem como
        um prompt de agente com acesso a shell, files e web search.

        Args:
            name: Nome do job (ex: 'backup-check', 'news-digest').
            message: Prompt que o PicoClaw vai executar periodicamente.
            interval_minutes: Intervalo em minutos (default: 30).
        """
        if not CRON_JOBS_FILE.parent.exists():
            CRON_JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Lê jobs existentes
        if CRON_JOBS_FILE.exists():
            data = json.loads(CRON_JOBS_FILE.read_text("utf-8"))
        else:
            data = {"jobs": []}

        # Checa duplicata
        for job in data["jobs"]:
            if job.get("name") == name:
                return f"Job '{name}' já existe. Remove primeiro com pico_cron_remove."

        # Gera ID único
        import hashlib
        job_id = hashlib.md5(f"{name}{message}".encode()).hexdigest()[:16]

        new_job = {
            "id": job_id,
            "name": name,
            "enabled": True,
            "schedule": {
                "kind": "every",
                "everyMs": interval_minutes * 60 * 1000,
            },
            "payload": {
                "kind": "agent_turn",
                "message": message,
            },
        }

        data["jobs"].append(new_job)
        CRON_JOBS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")

        logger.info("PicoClaw cron added: %s (every %dmin)", name, interval_minutes)
        return f"Job '{name}' criado — roda a cada {interval_minutes}min.\nPrompt: {message}"

    async def pico_cron_remove(self, name: str) -> str:
        """Remove um cron job do PicoClaw pelo nome.

        Args:
            name: Nome do job para remover.
        """
        if not CRON_JOBS_FILE.exists():
            return "Nenhum cron job configurado."

        data = json.loads(CRON_JOBS_FILE.read_text("utf-8"))
        original = len(data["jobs"])
        data["jobs"] = [j for j in data["jobs"] if j.get("name") != name]

        if len(data["jobs"]) == original:
            names = [j.get("name", "?") for j in data["jobs"]]
            return f"Job '{name}' não encontrado. Existentes: {', '.join(names)}"

        CRON_JOBS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")
        logger.info("PicoClaw cron removed: %s", name)
        return f"Job '{name}' removido."

    async def pico_memory(self) -> str:
        """Lê a memória persistente do PicoClaw (MEMORY.md).

        Útil pra ver o que o PicoClaw aprendeu nas sessões dele.
        """
        mem_file = PICOCLAW_WORKSPACE / "MEMORY.md"
        if not mem_file.exists():
            return "PicoClaw não tem memória salva ainda."

        content = mem_file.read_text("utf-8")
        if len(content) > 4000:
            content = content[:4000] + "\n\n... (truncado)"
        return f"=== Memória PicoClaw ===\n{content}"

    async def pico_status(self) -> str:
        """Verifica o status do PicoClaw — binário, config, workspace, cron.

        Retorna um diagnóstico completo do estado da instalação.
        """
        lines = ["=== PicoClaw Status ==="]

        # Binário
        bin_path = Path(PICOCLAW_BIN)
        if bin_path.exists():
            size_mb = bin_path.stat().st_size / (1024 * 1024)
            lines.append(f"Binário: {PICOCLAW_BIN} ({size_mb:.1f}MB)")
        else:
            lines.append(f"Binário: NÃO ENCONTRADO em {PICOCLAW_BIN}")

        # Config
        if PICOCLAW_CONFIG.exists():
            try:
                cfg = json.loads(PICOCLAW_CONFIG.read_text("utf-8"))
                models = [m.get("model_name", "?") for m in cfg.get("model_list", [])]
                default = cfg.get("agents", {}).get("defaults", {}).get("model", "?")
                lines.append(f"Config: {PICOCLAW_CONFIG}")
                lines.append(f"Modelos: {', '.join(models)} (default: {default})")
            except Exception as e:
                lines.append(f"Config: ERRO ao ler — {e}")
        else:
            lines.append(f"Config: NÃO ENCONTRADA em {PICOCLAW_CONFIG}")

        # Workspace
        if PICOCLAW_WORKSPACE.exists():
            files = list(PICOCLAW_WORKSPACE.glob("*.md"))
            lines.append(f"Workspace: {PICOCLAW_WORKSPACE} ({len(files)} .md files)")
        else:
            lines.append(f"Workspace: NÃO ENCONTRADO em {PICOCLAW_WORKSPACE}")

        # Cron
        cron_info = await self.pico_cron_list()
        lines.append(cron_info)

        return "\n".join(lines)
