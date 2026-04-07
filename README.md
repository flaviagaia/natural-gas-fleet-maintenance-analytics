# Natural Gas Fleet Maintenance Analytics

## PT-BR

### Visão rápida
Este projeto mostra como estruturar um pipeline de **analytics e manutenção preditiva para uma frota movida a gás natural**, combinando telemetria operacional, score de saúde e priorização de ativos críticos. A proposta é transformar sinais de pressão, temperatura, eficiência e vibração em uma leitura acionável para manutenção e operação.

### Problema de negócio
Em frotas energizadas por gás natural, a confiabilidade dos ativos afeta diretamente:
- continuidade operacional;
- consumo e eficiência energética;
- disponibilidade da frota;
- custo de manutenção corretiva;
- planejamento de ativos reserva.

O problema prático é decidir:
- quais unidades estão se degradando mais rápido;
- quais sinais estão puxando esse risco;
- onde agir primeiro sem esperar a falha completa.

### Base pública de referência
O projeto usa como referência pública o **MetroPT-3 Dataset**, da UCI, um dataset clássico de manutenção preditiva em compressor industrial. A execução local usa uma amostra `MetroPT-3-style` adaptada para um contexto de frota movida a gás natural, o que mantém o runtime leve e reproduzível.

Referência oficial:
- [UCI - MetroPT-3 Dataset](https://archive.ics.uci.edu/dataset/791/metropt+3+)

### O que o projeto faz
1. Gera uma base local de telemetria por ativo.
2. Treina um classificador para prever `maintenance_required`.
3. Aplica o modelo sobre a última janela de cada ativo.
4. Converte a probabilidade em `health_score` e `fleet_band`.
5. Exporta um resumo pronto para control tower de manutenção.

### Estrutura do projeto
- `main.py`: entry point local.
- `src/sample_data.py`: gera a base sintética inspirada em compressor industrial e ativos movidos a gás natural.
- `src/modeling.py`: treina o pipeline, calcula métricas e produz o resumo final.
- `tests/test_project.py`: valida o contrato mínimo do pipeline.
- `data/raw/public_dataset_reference.json`: referência da base pública usada no framing.
- `data/processed/natural_gas_scored_cycles.csv`: holdout pontuado.
- `data/processed/natural_gas_fleet_summary.csv`: snapshot da frota.
- `data/processed/natural_gas_fleet_maintenance_report.json`: relatório consolidado.

### Variáveis usadas
- `asset_id`: identificador do ativo.
- `cycle`: posição temporal da leitura.
- `gas_pressure`: pressão do sistema de gás.
- `oil_temperature`: temperatura do óleo.
- `motor_current`: corrente do motor.
- `compression_efficiency`: eficiência de compressão.
- `exhaust_temperature`: temperatura de exaustão.
- `vibration`: vibração do ativo.
- `fuel_flow`: fluxo de combustível.
- `start_delay_index`: proxy de atraso e instabilidade de partida.
- `maintenance_required`: alvo supervisionado.

### Modelagem
O pipeline usa:
- imputação de variáveis numéricas e categóricas;
- `OneHotEncoder` para `asset_id`;
- `RandomForestClassifier` com balanceamento por subsample para lidar melhor com eventos de risco menos frequentes.

Essa modelagem faz sentido porque:
- capta não linearidade entre eficiência, pressão, temperatura e vibração;
- é robusta para um MVP local;
- permite explicar o raciocínio operacional em entrevista.

### Saídas operacionais
O projeto gera:

**1. scored cycles**
- previsão no conjunto de teste;
- útil para validação offline.

**2. fleet summary**
- última leitura por ativo;
- `predicted_probability`;
- `health_score`;
- `fleet_band` em `stable`, `watchlist` ou `critical`.

Essa segunda saída representa a visão que uma control tower usaria para priorização diária.

### Resultados atuais
- `dataset_source = natural_gas_fleet_sample_metropt3_style`
- `row_count = 627`
- `asset_count = 8`
- `positive_rate = 0.1834`
- `roc_auc = 0.8979`
- `average_precision = 0.8212`
- `f1 = 0.7119`
- `critical_assets = 6`

### Como executar
```bash
python3 main.py
python3 -m unittest discover -s tests -v
```

### Do básico ao avançado
No nível básico, este é um classificador supervisionado de necessidade de manutenção.

No nível intermediário, ele vira um sistema de **fleet maintenance analytics**.

No nível avançado, ele permite discutir:
- batch scoring versus atualização near real-time;
- governança de sinais de telemetria;
- monitoramento de drift e estabilidade operacional;
- escalabilidade por base, região e tipo de ativo energético.

### Batch vs stream
- `batch`:
  - recalcular o score completo da frota;
  - retreinar o modelo;
  - consolidar dashboards por turno ou período.

- `stream`:
  - atualizar risco quando entra telemetria nova;
  - sinalizar ativos críticos rapidamente;
  - alimentar monitoramento operacional em baixa latência.

Trade-off:
- batch é mais simples e auditável;
- stream melhora tempo de resposta, mas aumenta a complexidade operacional.

### Governança e monitoramento
Um pipeline real desse tipo exigiria:
- validação de ranges físicos dos sensores;
- controle de gaps e latência da telemetria;
- lineage entre leitura bruta, feature e score;
- monitoramento de drift;
- rastreabilidade do modelo usado em cada execução.

### Limitações
- a execução local usa uma amostra inspirada no `MetroPT-3`, não o dataset completo;
- a validação é offline;
- o rótulo de manutenção é um proxy supervisionado de risco.

## EN

### Quick overview
This project structures a **maintenance analytics workflow for a natural-gas-powered fleet**, turning telemetry into health scoring and fleet prioritization outputs.

### Public dataset framing
The project is framed around the **MetroPT-3 Dataset** from UCI, a classic compressor predictive-maintenance dataset. Runtime execution uses a compact local `MetroPT-3-style` sample adapted to a natural-gas fleet setting for deterministic execution.

### What the project does
1. Generates a local fleet telemetry sample.
2. Trains a classifier for `maintenance_required`.
3. Scores the latest cycle of each asset.
4. Converts probabilities into `health_score` and `fleet_band`.
5. Exports a fleet summary for maintenance prioritization.

### Current results
- `dataset_source = natural_gas_fleet_sample_metropt3_style`
- `row_count = 627`
- `asset_count = 8`
- `positive_rate = 0.1834`
- `roc_auc = 0.8979`
- `average_precision = 0.8212`
- `f1 = 0.7119`
- `critical_assets = 6`

### Run locally
```bash
python3 main.py
python3 -m unittest discover -s tests -v
```

### Advanced discussion points
This repository is useful to discuss:
- fleet-wide predictive maintenance;
- batch versus near-real-time scoring;
- telemetry governance;
- drift monitoring;
- scaling analytics across natural gas operations.
