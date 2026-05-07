"""
Processamento dos dados para o modelo de machine learning.

O arquivo `preprocessing.py` é responsável por carregar, limpar e preparar os dados para o treinamento do modelo. Ele inclui:
- Carregamento do dataset de notícias a partir de um arquivo CSV.
- Limpeza e normalização dos textos das notícias.
- Tokenização e vetorização dos textos para uso em modelos de machine learning.
"""
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split(
    path: str = str(BASE_DIR / "data/News_Category_Dataset_balanced.csv"),
    test_size: float = 0.2,
    seed: int = 42,
):
    """
    Carrega o CSV de notícias já limpo e balanceado e divide em treino/teste.

    Parâmetros:
        path      : caminho para o arquivo CSV
        test_size : proporção do conjunto de teste (padrão: 20%)
        seed      : semente aleatória para reprodutibilidade

    Retorna:
        X_train, X_test, y_train, y_test
    """
    # Carrega o dataset limpo e balanceado (depois carregaremos do Supabase/DuckDB)
    print(f"Carregando dados de: {path}")
    df = pd.read_csv(path)

    print(f"Dataset carregado com {len(df)} amostras")
    print(df.head())

    # mantém apenas as colunas headline e category
    df = df[["headline", "category"]]

    # remove todas as linhas com category 'OTHER' (é uma categoria muito genérica e não tem um padrão claro de texto)
    df = df[df["category"] != "OTHER"]

    print("Distribuição de categorias:")
    print(df["category"].value_counts())

    X = df["headline"]
    y = df["category"]

    # stratify=y garante que a proporção de classes seja igual em treino e teste
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )