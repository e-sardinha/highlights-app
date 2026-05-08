"""
Processamento dos dados para o modelo de machine learning.

Este módulo transforma dados brutos (Parquet/CSV), cria um split
treino/teste persistido em Parquet e disponibiliza funções para carregar
os splits em scripts de treino e avaliação.
"""
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

import json
from datetime import datetime, timezone
import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DEFAULT_PATH = BASE_DIR / "data/raw/news_category_raw.parquet"
PROCESSED_DEFAULT_DIR = BASE_DIR / "data/processed"


def _read_source(source_path: Path) -> pd.DataFrame:
    conn = duckdb.connect()
    source_sql = "read_parquet(?)" if source_path.suffix.lower() == ".parquet" else "read_csv_auto(?)"

    query = f"""
        SELECT
            headline,
            category
        FROM {source_sql}
        WHERE headline IS NOT NULL
          AND category IS NOT NULL
          AND trim(headline) <> ''
          AND category <> 'OTHER'
    """
    df = conn.execute(query, [str(source_path)]).fetchdf()
    conn.close()
    return df


def _write_series_parquet(series: pd.Series, output_path: Path, column_name: str):
    conn = duckdb.connect()
    conn.register("tmp_series", series.rename(column_name).to_frame())
    conn.execute(f"COPY tmp_series TO '{output_path.as_posix()}' (FORMAT PARQUET)")
    conn.close()


def _read_series_parquet(input_path: Path) -> pd.Series:
    conn = duckdb.connect()
    df = conn.execute("SELECT * FROM read_parquet(?)", [str(input_path)]).fetchdf()
    conn.close()
    return df.iloc[:, 0]


def build_and_save_splits(
    source_path: str = str(RAW_DEFAULT_PATH),
    output_dir: str = str(PROCESSED_DEFAULT_DIR),
    test_size: float = 0.2,
    seed: int = 42,
):
    """
    Lê a fonte, aplica limpeza básica e persiste os splits em Parquet.

    Retorna:
        X_train, X_test, y_train, y_test
    """
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Fonte não encontrada: {source}")

    print(f"Carregando dados de: {source}")
    df = _read_source(source)
    if df.empty:
        raise ValueError("A fonte carregada não possui dados válidos após limpeza.")

    print(f"Dataset após limpeza com {len(df)} amostras")
    print("Distribuição de categorias:")
    print(df["category"].value_counts())

    X = df["headline"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    print(f"Amostras de treino: {len(X_train)}")
    print(f"Amostras de teste : {len(X_test)}")
    print("Distribuição no treino:")
    print(y_train.value_counts())
    print("Distribuição no teste:")
    print(y_test.value_counts())

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    _write_series_parquet(X_train, output / "X_train.parquet", "headline")
    _write_series_parquet(X_test, output / "X_test.parquet", "headline")
    _write_series_parquet(y_train, output / "y_train.parquet", "category")
    _write_series_parquet(y_test, output / "y_test.parquet", "category")

    metadata = {
        "source_path": str(source),
        "rows_after_cleaning": int(len(df)),
        "test_size": test_size,
        "seed": seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "split_meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Splits persistidos em: {output}")
    return X_train, X_test, y_train, y_test


def load_saved_splits(input_dir: str = str(PROCESSED_DEFAULT_DIR)):
    """
    Carrega os splits já persistidos em Parquet.

    Retorna:
        X_train, X_test, y_train, y_test
    """
    base = Path(input_dir)
    required = [
        base / "X_train.parquet",
        base / "X_test.parquet",
        base / "y_train.parquet",
        base / "y_test.parquet",
    ]

    missing = [p for p in required if not p.exists()]
    if missing:
        missing_list = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Splits persistidos não encontrados. Execute build_and_save_splits() antes.\n"
            f"Arquivos ausentes:\n{missing_list}"
        )

    X_train = _read_series_parquet(base / "X_train.parquet")
    X_test = _read_series_parquet(base / "X_test.parquet")
    y_train = _read_series_parquet(base / "y_train.parquet")
    y_test = _read_series_parquet(base / "y_test.parquet")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    build_and_save_splits()