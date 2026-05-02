"""
prepare_data.py — Carrega, limpa e divide o dataset de notícias para classificação de highlights.

Uso standalone:
    python src/prepare_data.py

Uso como módulo (importado pelo train.py e evaluate.py):
    from src.prepare_data import load_and_split
"""
#from pathlib import Path
#BASE_DIR = Path(__file__).resolve().parent.parent

import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split(
    path: str = "data/News_Category_Dataset_v3.json",
    test_size: float = 0.2,
    seed: int = 42,
):
    """
    Carrega o JSON de notícias, faz limpeza básica e divide em treino/teste.

    Parâmetros:
        path      : caminho para o arquivo JSON
        test_size : proporção do conjunto de teste (padrão: 20%)
        seed      : semente aleatória para reprodutibilidade

    Retorna:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_json(path, orient='records', lines=True)

    # Remove linhas sem texto ou sem categoria
    df = df.dropna(subset=["short_description", "category"])

    # Remove linhas com texto vazio
    df = df[df["short_description"].str.strip() != ""]

    # Normalização básica do texto (minúsculas + remover espaços extras)
    df["short_description"] = df["short_description"].str.lower().str.strip()

    # Balanceamento: undersample para o tamanho da classe minoritária
    # Isso evita que o modelo fique enviesado para as classes majoritárias, mas reduz o número total de amostras.
    # O Exemplo é meramente didático, para um projeto real, técnicas mais avançadas de balanceamento (como oversampling ou geração de dados sintéticos) podem ser consideradas.
    min_count = df['category'].value_counts().min()
    df_balanced = df.groupby('category', group_keys=False).apply(lambda x: x.sample(min_count, random_state=seed)).reset_index(drop=True)

    print(f"Dataset balanceado com {len(df_balanced)} amostras (min_count: {min_count})")
    print(f"Distribuição de categorias após balanceamento:")
    print(df_balanced['category'].value_counts())

    X = df_balanced["short_description"]
    y = df_balanced["category"]

    # stratify=y garante que a proporção de classes seja igual em treino e teste
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()

    print(f"Total de amostras de treino : {len(X_train)}")
    print(f"Total de amostras de teste  : {len(X_test)}")
    print(f"\nDistribuição de categorias no treino:")
    print(y_train.value_counts())
