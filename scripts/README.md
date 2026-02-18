# Scripts do Enton

Scripts utilitários organizados por função.

## Estrutura

- **`setup/`**: Configuração inicial e instalação.
  - `phone_setup.sh`: Configura ADB e dependências Android.
  - `load_commonsense.py`: Baixa e indexa conhecimento comum (ASCENT++).

- **`tests/`**: Verificação e testes manuais.
  - `smoke_test_f1.py`: Teste de fumaça (inicializa App e verifica componentes).
  - `verify_phase2.py`: Valida loop de pensamento do cérebro.
  - `verify_phase3.py`: Valida registro de skills dinâmicas.

- **`data/`**: Gestão de dados e modelos.
  - `optimize_models.py`: Converte modelos YOLO (.pt) para TensorRT (.engine).

- **`dev/`**: Ferramentas de desenvolvimento.
  - `live_yolo.py`: Visualizador em tempo real da visão computacional.

## Uso

Execute a partir da raiz do projeto:

```bash
# Exemplo: Rodar smoke test
uv run python scripts/tests/smoke_test_f1.py

# Exemplo: Otimizar modelos
uv run python scripts/data/optimize_models.py
```
