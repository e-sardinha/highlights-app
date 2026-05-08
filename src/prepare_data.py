"""
prepare_data.py — Transforma o dataset original em JSON para CSV balanceado.

Uso standalone:
    python src/prepare_data.py
"""
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

import pandas as pd


def prepare_json_to_csv(
    path: str = str(BASE_DIR / "data/News_Category_Dataset_v3.json"),
    output_path: str = str(BASE_DIR / "data/News_Category_Dataset_balanced.csv"),
    seed: int = 42,
):
    """
    Carrega o JSON de notícias, faz limpeza básica e salva um CSV balanceado.

    Parâmetros:
        path        : caminho para o arquivo JSON
        output_path : caminho do CSV de saída
        seed        : semente aleatória para reprodutibilidade
    """
    print(f"Carregando dados de: {path}")
    df = pd.read_json(path, orient='records', lines=True)
    print(f"Dataset original carregado com {len(df)} amostras")
    print(df.head())

    # Remove linhas sem texto ou sem categoria
    df = df.dropna(subset=["headline", "category"])

    # Remove linhas com texto vazio
    df = df[df["headline"].str.strip() != ""]

    # Normalização básica do texto (minúsculas + remover espaços extras)
    df["headline"] = df["headline"].str.lower().str.strip()

    # Reduz as categorias às 5 mais comuns e agrupa o restante como OTHER, Isso é para simplificação já que é um experimento meramente didático.
    # Em um projeto real, a escolha de manter ou agrupar categorias deve ser feita com base na análise do domínio e dos dados.
    top_categories = df["category"].value_counts().nlargest(5).index
    df["category"] = df["category"].where(df["category"].isin(top_categories), "OTHER")

    # Mantém apenas as colunas headline e category, removendo todas as demais
    df = df[["headline", "category"]]

    print("Distribuição de categorias após agregação para OTHER:")
    print(df["category"].value_counts())

    # Balanceamento: undersample para o tamanho da classe minoritária
    # Isso evita que o modelo fique enviesado para as classes majoritárias, mas reduz o número total de amostras.
    # O Exemplo é meramente didático, para um projeto real, técnicas mais avançadas de balanceamento (como oversampling ou geração de dados sintéticos) podem ser consideradas.
    min_count = df['category'].value_counts().min()
    df_balanced = df.groupby('category', group_keys=False).apply(lambda x: x.sample(min_count, random_state=seed)).reset_index(drop=True)

    print(f"Dataset balanceado com {len(df_balanced)} amostras (min_count: {min_count})")
    print(f"Distribuição de categorias após balanceamento:")
    print(df_balanced['category'].value_counts())

    balanced_path = Path(output_path)
    balanced_path.parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(balanced_path, index=False)
    print(f"Dataset balanceado salvo em: {balanced_path}")
    return balanced_path


if __name__ == "__main__":
    prepare_json_to_csv()
