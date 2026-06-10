"""
fipe_ingestao.py
Ingestão da Tabela FIPE em arquitetura medallion (raw -> silver).
Projeto-âncora de portfólio: precificação e depreciação automotiva.

Fonte: API pública FIPE (parallelum) - https://parallelum.com.br/fipe/api/v1
Limite: 500 req/dia sem token | 1.000 req/dia com token gratuito.

Uso:
    python fipe_ingestao.py --tipo carros --marcas 21,59,23 --limite-modelos 5

Dependências:
    pip install requests pandas pyarrow
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://parallelum.com.br/fipe/api/v1"
RAW_DIR = Path("data/raw/fipe")
SILVER_DIR = Path("data/silver/fipe")

# Respeito ao rate limit: pausa entre chamadas. Ajuste se usar token.
PAUSA_SEGUNDOS = 0.8
TIMEOUT = 15


def get(endpoint: str) -> list | dict:
    """GET resiliente com retry simples e backoff."""
    url = f"{BASE_URL}/{endpoint}"
    for tentativa in range(3):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            time.sleep(PAUSA_SEGUNDOS)
            return resp.json()
        except requests.HTTPError as e:
            if resp.status_code == 429:  # rate limit estourado
                espera = 5 * (tentativa + 1)
                print(f"  [429] aguardando {espera}s...")
                time.sleep(espera)
                continue
            raise e
        except requests.RequestException as e:
            print(f"  erro de rede ({e}); tentativa {tentativa + 1}/3")
            time.sleep(3)
    raise RuntimeError(f"Falha ao buscar {url}")


def salvar_raw(nome: str, payload) -> None:
    """Persiste o JSON cru antes de qualquer transformação (camada raw)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    caminho = RAW_DIR / f"{nome}.json"
    caminho.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def coletar(tipo: str, marcas_alvo: list[str] | None, limite_modelos: int | None) -> pd.DataFrame:
    """Percorre marcas -> modelos -> anos -> preço e devolve um DataFrame plano."""
    registros = []

    marcas = get(f"{tipo}/marcas")
    salvar_raw(f"{tipo}_marcas", marcas)

    if marcas_alvo:
        marcas = [m for m in marcas if m["codigo"] in marcas_alvo]

    for marca in marcas:
        cod_marca = marca["codigo"]
        print(f"Marca: {marca['nome']} ({cod_marca})")

        modelos_resp = get(f"{tipo}/marcas/{cod_marca}/modelos")
        modelos = modelos_resp.get("modelos", [])
        if limite_modelos:
            modelos = modelos[:limite_modelos]

        for modelo in modelos:
            cod_modelo = modelo["codigo"]
            anos = get(f"{tipo}/marcas/{cod_marca}/modelos/{cod_modelo}/anos")

            for ano in anos:
                cod_ano = ano["codigo"]
                preco = get(
                    f"{tipo}/marcas/{cod_marca}/modelos/{cod_modelo}/anos/{cod_ano}"
                )
                registros.append(preco)
                print(f"  {preco.get('Modelo')} {preco.get('AnoModelo')} -> {preco.get('Valor')}")

    return pd.DataFrame(registros)


def to_silver(df: pd.DataFrame, tipo: str) -> Path:
    """Normaliza tipos e salva Parquet particionado por data de ingestão (camada silver)."""
    if df.empty:
        raise ValueError("Nenhum registro coletado.")

    df = df.copy()
    df["valor_num"] = (
        df["Valor"].str.replace("R$", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .astype(float)
    )
    df["tipo_veiculo"] = tipo
    df["data_ingestao"] = datetime.now().date().isoformat()
    df.columns = [c.lower() for c in df.columns]

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    saida = SILVER_DIR / f"fipe_{tipo}_{datetime.now():%Y%m%d}.parquet"
    df.to_parquet(saida, index=False)
    return saida


def main():
    parser = argparse.ArgumentParser(description="Ingestão FIPE -> medallion")
    parser.add_argument("--tipo", default="carros", choices=["carros", "motos", "caminhoes"])
    parser.add_argument("--marcas", default=None, help="Códigos separados por vírgula. Ex: 21,59,23")
    parser.add_argument("--limite-modelos", type=int, default=3, help="Limita modelos por marca (cuida do rate limit)")
    args = parser.parse_args()

    marcas_alvo = args.marcas.split(",") if args.marcas else None

    print(f"== Coleta FIPE: {args.tipo} ==")
    df = coletar(args.tipo, marcas_alvo, args.limite_modelos)
    caminho = to_silver(df, args.tipo)
    print(f"\nOK: {len(df)} registros salvos em {caminho}")


if __name__ == "__main__":
    main()
