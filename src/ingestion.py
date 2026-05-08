"""
Ingestão de dados Supabase -> Parquet bruto.
O arquivo `ingestion.py` extrai dados do Supabase e salva uma camada raw
em Parquet para ser consumida no preprocessing.
"""
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import duckdb

BASE_DIR = Path(__file__).resolve().parent.parent

# Fazer a ingestão dos dados do Supabase para um DataFrame
def ingest_data():
    from supabase import create_client, Client
    import os

    # Carrega as credenciais do arquivo .env
    load_dotenv()

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Defina SUPABASE_URL e SUPABASE_KEY no arquivo .env")

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    page_size = 1000
    offset = 0
    data = []

    try:
        while True:
            response = (
                supabase
                .table("news_category")
                .select("headline, category")
                .range(offset, offset + page_size - 1)
                .execute()
            )
            page = response.data or []

            if not page:
                break

            data.extend(page)
            if len(page) < page_size:
                break

            offset += page_size
    except Exception as exc:
        print(f"Erro ao buscar dados do Supabase: {exc}")
        return pd.DataFrame(columns=["headline", "category"])

    if not data:
        print("Nenhum dado retornado da tabela news_category.")
        return pd.DataFrame(columns=["headline", "category"])

    df = pd.DataFrame(data)
    # Garante schema mínimo esperado para os próximos passos do pipeline.
    df = df[["headline", "category"]]
    print(f"Dados ingeridos com sucesso em DataFrame! Registros: {len(df)}")
    return df


def save_raw_parquet(df: pd.DataFrame, output_path: str = str(BASE_DIR / "data/raw/news_category_raw.parquet")):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect()
    conn.register("raw_df", df)
    conn.execute(f"COPY raw_df TO '{output.as_posix()}' (FORMAT PARQUET)")
    conn.close()

    print(f"Camada raw salva em: {output}")
    return output


if __name__ == "__main__":
    dataframe = ingest_data()
    if dataframe.empty:
        print("Ingestão finalizada sem dados para salvar.")
    else:
        save_raw_parquet(dataframe)
        print(dataframe.head())