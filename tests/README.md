# Testes do Enton

Estrutura de testes organizada para facilitar a manutenção.

## Organização

- **`manual/`**: Testes que requerem interação humana ou hardware específico não mockado.
- **`unit/`**: Testes unitários automatizados, divididos por módulo:
  - `core/`: Componentes fundamentais (config, events, etc).
  - `cognition/`: Cérebro, desejos, memória.
  - `perception/`: Visão, audição, ações.
  - `skills/`: Habilidades e toolkits.
  - `providers/`: Integrações com LLMs e serviços externos.

## Executando Testes

Rodar todos os testes automatizados:
```bash
make test
# ou
uv run pytest
```

Rodar apenas uma seção específica:
```bash
uv run pytest tests/unit/cognition
```
