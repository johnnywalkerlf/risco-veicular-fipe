"""
susep_ivr.py
Integração do IVR (Índice de Veículos Roubados / SUSEP) ao medidor de risco.

Fonte: SUSEP - https://www2.susep.gov.br/menuestatistica/rankroubo/menu1.asp
O IVR é um formulário ASP legado. Fluxo recomendado:
  1. No site, selecione categoria(s), região (ex.: 'SP - Grande Campinas') e ordene.
  2. Exporte/cole o resultado em um CSV com colunas:
        categoria_susep ; descricao ; ano_modelo ; indice
     (o 'indice' é a frequência relativa de roubo; quanto maior, pior)
  3. Rode este módulo para normalizar (0-100) e casar com a FIPE.

Saída: a base FIPE silver enriquecida com a coluna 'risco_furto' (0-100),
pronta para entrar em medidor_risco.calcular_score(..., coluna_risco_furto="risco_furto").

Dependências: pip install pandas  (difflib é stdlib)
"""

from __future__ import annotations

import difflib
import pandas as pd

# Mapa categoria SUSEP -> tipo de veículo FIPE (join limpo, sem fuzzy).
CATEGORIA_SUSEP_PARA_FIPE = {
    "Passeio nacional": "carros",
    "Passeio importado": "carros",
    "Pick-up (nacional e importado)": "carros",
    "Utilitários (nacional e importado)": "carros",
    "Motocicleta (nacional e importado)": "motos",
    "Veículo de Carga (nacional e importado)": "caminhoes",
    "Ônibus (nacional e importado)": "caminhoes",
}


def normalizar_ivr(df_susep: pd.DataFrame) -> pd.DataFrame:
    """Normaliza o índice IVR para 0-100 dentro de cada tipo de veículo FIPE.

    Normalizar por tipo evita que motos (índice naturalmente alto) achatem
    a escala dos carros — cada segmento é comparado contra seus pares.
    """
    df = df_susep.copy()
    df["tipo_veiculo"] = df["categoria_susep"].map(CATEGORIA_SUSEP_PARA_FIPE)
    df = df.dropna(subset=["tipo_veiculo", "indice"])
    df["indice"] = pd.to_numeric(df["indice"], errors="coerce")

    def _norm(g: pd.Series) -> pd.Series:
        lo, hi = g.min(), g.max()
        if hi == lo:
            return pd.Series(50.0, index=g.index)
        return (g - lo) / (hi - lo) * 100

    df["risco_furto"] = df.groupby("tipo_veiculo")["indice"].transform(_norm).round(1)
    return df


def _match_modelo(nome_fipe: str, candidatos: list[str], corte: float = 0.6) -> str | None:
    """Casamento aproximado de nome de modelo (FIPE x descrição SUSEP)."""
    achados = difflib.get_close_matches(nome_fipe.upper(), candidatos, n=1, cutoff=corte)
    return achados[0] if achados else None


def enriquecer_fipe(
    df_fipe: pd.DataFrame,
    df_ivr_norm: pd.DataFrame,
    usar_fuzzy: bool = True,
) -> pd.DataFrame:
    """Acopla 'risco_furto' à base FIPE.

    Estratégia em camadas:
      1. Join por (tipo_veiculo, ano_modelo) + match aproximado de modelo (preciso).
      2. Fallback: média do risco_furto do tipo+ano.
      3. Fallback final: 50 (neutro), nunca deixa nulo.
    """
    df = df_fipe.copy()
    df["anomodelo"] = pd.to_numeric(df["anomodelo"], errors="coerce")
    df_ivr_norm = df_ivr_norm.copy()
    df_ivr_norm["ano_modelo"] = pd.to_numeric(df_ivr_norm["ano_modelo"], errors="coerce")

    riscos = []
    for _, linha in df.iterrows():
        tipo, ano, modelo = linha["tipo_veiculo"], linha["anomodelo"], str(linha["modelo"])
        sub = df_ivr_norm[(df_ivr_norm["tipo_veiculo"] == tipo) &
                          (df_ivr_norm["ano_modelo"] == ano)]

        valor = None
        if not sub.empty and usar_fuzzy:
            achado = _match_modelo(modelo, sub["descricao"].str.upper().tolist())
            if achado is not None:
                valor = sub.loc[sub["descricao"].str.upper() == achado, "risco_furto"].iloc[0]
        if valor is None and not sub.empty:
            valor = sub["risco_furto"].mean()            # fallback tipo+ano
        if valor is None:
            tipo_geral = df_ivr_norm[df_ivr_norm["tipo_veiculo"] == tipo]["risco_furto"]
            valor = tipo_geral.mean() if not tipo_geral.empty else 50.0  # fallback final
        riscos.append(round(float(valor), 1))

    df["risco_furto"] = riscos
    return df


if __name__ == "__main__":
    # --- Demo: IVR sintético + FIPE sintético ---
    ivr = pd.DataFrame({
        "categoria_susep": ["Passeio nacional", "Passeio nacional", "Passeio nacional",
                            "Motocicleta (nacional e importado)", "Motocicleta (nacional e importado)"],
        "descricao":       ["VW GOL", "FIAT MOBI", "VW POLO", "HONDA CG 160", "HONDA BIZ"],
        "ano_modelo":      [2022, 2022, 2024, 2022, 2024],
        "indice":          [3.10, 1.20, 2.40, 5.80, 4.10],
    })

    fipe = pd.DataFrame({
        "marca":        ["Fiat", "VW", "VW", "Honda"],
        "modelo":       ["Mobi", "Polo", "Gol", "CG 160 Titan"],
        "anomodelo":    [2022, 2024, 2022, 2022],
        "valor_num":    [46000, 98000, 52000, 12000],
        "tipo_veiculo": ["carros", "carros", "carros", "motos"],
    })

    ivr_norm = normalizar_ivr(ivr)
    fipe_enriquecida = enriquecer_fipe(fipe, ivr_norm)

    print(">> IVR normalizado (0-100 por tipo):")
    print(ivr_norm[["descricao", "tipo_veiculo", "indice", "risco_furto"]].to_string(index=False))
    print("\n>> FIPE enriquecida com risco_furto:")
    print(fipe_enriquecida[["marca", "modelo", "anomodelo", "tipo_veiculo", "risco_furto"]].to_string(index=False))
