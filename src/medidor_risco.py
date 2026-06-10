"""
medidor_risco.py
Motor de scoring de risco veicular (camada gold do projeto FIPE).

Combina fatores normalizados (0-100) em um score composto auditável.
Pensado para o setor de proteção veicular: cada fator e peso é explícito,
para que a precificação seja defensável perante diretoria/regulação.

Fatores:
  1. exposicao_financeira  -> valor FIPE (maior valor = maior perda potencial)
  2. depreciacao_anual     -> derivado da curva de anos-modelo do próprio FIPE
  3. risco_furto           -> encaixe p/ índice externo OU sinistralidade interna
                              (CooperLink). Neutro (50) se não fornecido.
  4. fator_categoria       -> moto / caminhão / esportivo pesam diferente

Saída: score 0-100 + faixa (Baixo / Médio / Alto / Crítico).

Dependências: pip install pandas pyarrow
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# Pesos default (somam 1.0). Ajuste conforme apetite de risco da carteira.
PESOS_DEFAULT = {
    "exposicao_financeira": 0.30,
    "depreciacao_anual": 0.20,
    "risco_furto": 0.40,
    "fator_categoria": 0.10,
}

# Score base de categoria (0-100). Calibrável com dados reais de sinistro.
SCORE_CATEGORIA = {
    "motos": 85,        # alta exposição a roubo/acidente
    "caminhoes": 60,    # alto valor, mas menor frequência
    "carros": 50,
}

FAIXAS = [(0, 25, "Baixo"), (25, 50, "Médio"), (50, 75, "Alto"), (75, 101, "Crítico")]


def _normaliza(serie: pd.Series) -> pd.Series:
    """Min-max para 0-100. Constante vira 50 (neutro), evitando divisão por zero."""
    lo, hi = serie.min(), serie.max()
    if hi == lo:
        return pd.Series(50.0, index=serie.index)
    return (serie - lo) / (hi - lo) * 100


def calcular_depreciacao_anual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada (marca, modelo), estima a depreciação anual média comparando
    o valor entre anos-modelo. Usa apenas a própria FIPE.
    Espera colunas: marca, modelo, anomodelo, valor_num.
    """
    df = df.copy()
    df["anomodelo"] = pd.to_numeric(df["anomodelo"], errors="coerce")
    df = df.dropna(subset=["anomodelo", "valor_num"])

    def _taxa(grupo: pd.DataFrame) -> float:
        g = grupo.sort_values("anomodelo")
        if len(g) < 2:
            return np.nan
        anos = g["anomodelo"].max() - g["anomodelo"].min()
        if anos <= 0:
            return np.nan
        v_novo, v_antigo = g["valor_num"].iloc[-1], g["valor_num"].iloc[0]
        if v_novo <= 0:
            return np.nan
        # depreciação média anual composta entre o mais antigo e o mais novo
        return (1 - (v_antigo / v_novo) ** (1 / anos)) * 100

    taxas = (
        df.groupby(["marca", "modelo"])
        .apply(_taxa, include_groups=False)
        .rename("depreciacao_anual_pct")
        .reset_index()
    )
    return df.merge(taxas, on=["marca", "modelo"], how="left")


def calcular_score(
    df: pd.DataFrame,
    pesos: dict | None = None,
    coluna_risco_furto: str | None = None,
) -> pd.DataFrame:
    """
    Recebe o silver FIPE e devolve o gold com score de risco.

    coluna_risco_furto: nome de uma coluna 0-100 (índice externo OU
    sinistralidade interna CooperLink). Se None, usa 50 (neutro).
    """
    pesos = pesos or PESOS_DEFAULT
    assert abs(sum(pesos.values()) - 1.0) < 1e-6, "Os pesos devem somar 1.0"

    df = calcular_depreciacao_anual(df)

    # Fator 1: exposição financeira
    df["f_exposicao"] = _normaliza(df["valor_num"])

    # Fator 2: depreciação (preenche faltantes com a mediana; se tudo nulo, neutro)
    mediana_dep = df["depreciacao_anual_pct"].median()
    if pd.isna(mediana_dep):
        mediana_dep = 0.0
    dep = df["depreciacao_anual_pct"].fillna(mediana_dep)
    df["f_depreciacao"] = _normaliza(dep)

    # Fator 3: risco de furto (slot)
    if coluna_risco_furto and coluna_risco_furto in df.columns:
        df["f_furto"] = df[coluna_risco_furto].clip(0, 100)
    else:
        df["f_furto"] = 50.0

    # Fator 4: categoria
    df["f_categoria"] = df["tipo_veiculo"].map(SCORE_CATEGORIA).fillna(50.0)

    df["score_risco"] = (
        df["f_exposicao"] * pesos["exposicao_financeira"]
        + df["f_depreciacao"] * pesos["depreciacao_anual"]
        + df["f_furto"] * pesos["risco_furto"]
        + df["f_categoria"] * pesos["fator_categoria"]
    ).round(1)

    def _faixa(s: float) -> str:
        for lo, hi, nome in FAIXAS:
            if lo <= s < hi:
                return nome
        return "Crítico"

    df["faixa_risco"] = df["score_risco"].apply(_faixa)
    return df


def salvar_gold(df: pd.DataFrame, caminho: str = "data/gold/fipe_risco.parquet") -> str:
    from pathlib import Path

    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(caminho, index=False)
    return caminho


if __name__ == "__main__":
    # --- Demo com dados sintéticos (valida a lógica sem depender da API) ---
    demo = pd.DataFrame({
        "marca":       ["Fiat", "Fiat", "Fiat", "VW", "VW", "Honda", "Honda", "Scania"],
        "modelo":      ["Mobi", "Mobi", "Mobi", "Polo", "Polo", "CG 160", "CG 160", "R450"],
        "anomodelo":   [2020,   2022,   2024,   2021,  2024,  2022,    2024,    2023],
        "valor_num":   [38000,  46000,  62000,  72000, 98000, 12000,   16000,   620000],
        "tipo_veiculo":["carros","carros","carros","carros","carros","motos","motos","caminhoes"],
    })

    gold = calcular_score(demo)
    cols = ["marca", "modelo", "anomodelo", "valor_num",
            "depreciacao_anual_pct", "score_risco", "faixa_risco"]
    print(gold[cols].to_string(index=False))
