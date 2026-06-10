# Risco e Precificação de Veículos — FIPE + SUSEP

> **Pergunta de negócio:** quais perfis de veículo merecem precificação diferenciada em uma carteira de proteção veicular?

Pipeline analítico ponta a ponta sobre **dados públicos do setor automotivo brasileiro**: ingestão via API, enriquecimento com o índice oficial do regulador de seguros, score de risco auditável, machine learning com validação rigorosa e simulação de precificação.

Construído com a ótica de quem trabalha no setor: uma associação de proteção veicular que cobra mensalidade única de riscos diferentes sofre **seleção adversa** — os bons riscos saem, os maus riscos ficam. Este projeto constrói o instrumento quantitativo para evitar isso.

---

## Destaques

| O que | Como | Resultado |
|---|---|---|
| **Score de risco composto** | 4 fatores normalizados, pesos explícitos e auditáveis | Faixas Baixo → Crítico por veículo |
| **Validação estatística dos perfis** | Kruskal-Wallis + IC bootstrap 95% | p ≈ 1,7e-16 — perfis distintos, não artefato |
| **Robustez do score** | Monte Carlo: 5.000 perturbações Dirichlet nos pesos | Ranking estável (Spearman ~0,91); achado honesto: cortes fixos são o ponto fraco → usar faixas do K-Means |
| **Previsão de valor** | Random Forest + RandomizedSearchCV, K-Fold 5, baseline | MAE ~R$ 8k vs ~R$ 64k do baseline (R² 0,96) |
| **Triagem antifraude** | Isolation Forest sobre valor × depreciação × furto | 5% mais atípicos = fila de revisão de cadastro |
| **Do score ao preço** | Simulação de prêmio puro por perfil | Mensalidade técnica por faixa de risco |
| **Fator condutor (idade)** | Multiplicadores relativos por faixa etária (IVR/SUSEP) | Matriz final: perfil do veículo × idade do condutor |

## Fontes de dados

| Fonte | Conteúdo | Acesso |
|---|---|---|
| [API FIPE](https://parallelum.com.br/fipe/api/v1) (parallelum) | Valor de mercado por marca/modelo/ano, mensal | REST gratuita — 500 req/dia (1.000 com token) |
| [SUSEP — IVR](https://www2.susep.gov.br/menuestatistica/rankroubo/menu1.asp) | Índice de Veículos Roubados, por veículo/região/ano | Export manual do formulário oficial (não há API) |
| [BCB/SGS](https://dadosabertos.bcb.gov.br) *(roadmap)* | Séries macro (Selic, IPCA, inadimplência) | REST pública |

⚠️ `data/ivr_susep_amostra.csv` é uma **amostra ilustrativa com a estrutura real do export** SUSEP, incluída para o projeto ser reprodutível offline. Para análise real, exporte o IVR oficial (ex.: região *SP — Grande Campinas*) e salve como `data/ivr_susep.csv`; o mesmo vale para o recorte por faixa etária do condutor (`data/ivr_faixa_etaria.csv`) — o pipeline detecta e prioriza o export real automaticamente.

## Estrutura

```
risco-veicular-fipe/
├── notebooks/
│   ├── risco_veicular_fipe.ipynb            # análise completa (limpo, p/ executar)
│   └── risco_veicular_fipe_executado.ipynb  # com outputs renderizados (p/ leitura)
├── src/
│   ├── fipe_ingestao.py     # ingestão API FIPE → medallion (raw/silver)
│   ├── susep_ivr.py         # normalização do IVR + fuzzy match com a FIPE
│   └── medidor_risco.py     # motor de score (camada gold)
├── data/
│   ├── ivr_susep_amostra.csv
│   └── ivr_faixa_etaria_amostra.csv
└── requirements.txt
```

## Como executar

```bash
pip install -r requirements.txt
jupyter notebook notebooks/risco_veicular_fipe.ipynb
```

Com internet, a ingestão usa a **API FIPE real**; offline, um gerador sintético calibrado (mesma metodologia, dados rotulados como sintéticos). O notebook declara explicitamente qual modo está ativo.

Pipeline via scripts (alternativa ao notebook):

```bash
python src/fipe_ingestao.py --tipo carros --marcas 21,59,23 --limite-modelos 5
python src/medidor_risco.py
```

## Decisões metodológicas (e por quê)

- **Gate de qualidade antes de qualquer análise** — asserções explícitas; dado quebrado para o pipeline, não produz conclusão errada.
- **k do K-Means limitado a 6** — tabela de preços com mais faixas é incomunicável; interpretabilidade > décimos de silhueta.
- **Importância por permutação no teste**, não Gini no treino — Gini é viesada para alta cardinalidade.
- **Baseline sempre presente** — MAE sem comparação não significa nada.
- **O Monte Carlo criticou o próprio score** — o ranking é robusto, mas os cortes fixos em 25/50/75 jogam veículos de fronteira entre faixas; a recomendação final usa as fronteiras naturais dos clusters.

## Limitações conhecidas

- O IVR mede **frequência** de roubo, não probabilidade ajustada por frota — modelos populares aparecem mais em parte porque há mais deles rodando. V2: normalizar pela frota (Senatran/Base dos Dados).
- A simulação de mensalidade usa frequência *proxy*; calibração real exige sinistralidade da carteira.
- Fuzzy match de nomes FIPE×SUSEP resolve a maioria dos casos, mas merece tabela de-para curada em produção.

## Roadmap

- [ ] Dashboard Power BI sobre a camada gold (modelo estrela)
- [ ] Calculadora de precificação what-if em Excel
- [ ] Fator macro via BCB/SGS (sensibilidade ao ciclo de crédito)
- [ ] Normalização do IVR pela frota circulante

---

**Stack:** Python · pandas · scikit-learn · scipy · matplotlib/seaborn · APIs REST · Parquet

*Autor: Johnny Walker — Analista de Dados | [LinkedIn](#) · [GitHub](#)*
