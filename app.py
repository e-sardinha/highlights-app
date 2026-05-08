"""
app.py — Interface Streamlit para classificação de destaques de notícias.

Execute com:
    streamlit run app.py

Ou via Docker:
    docker build --build-arg DAGSHUB_USER=... -t highlights-app .
    docker run -p 8501:8501 highlights-app
"""

import streamlit as st
import pandas as pd
import joblib

# ── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Classificador de Highlights de Notícias",
    page_icon="💬",
    layout="centered",
)

# ── Carrega o modelo (cache: não recarrega a cada interação) ─────────────────
@st.cache_resource
def load_model():
    """
    @st.cache_resource garante que o modelo é carregado apenas uma vez.
    Sem isso, o arquivo seria lido a cada clique do usuário.
    """
    return joblib.load("model.pkl")

pipeline = load_model()

# Mapeamento visual por categoria
EMOJI = {
    "ENTERTAINMENT": "🎬", 
    "POLITICS": "🏛️", 
    "STYLE & BEAUTY": "💄", 
    "TRAVEL": "✈️", 
    "WELLNESS": "🧘"
}

# ── Interface principal ──────────────────────────────────────────────────────
st.title(" Classificação de Notícias")
st.markdown(
    "Classifica destaques de notícias nas categorias **ENTERTAINMENT**, **POLITICS**, "
    "**STYLE & BEAUTY**, **TRAVEL** ou **WELLNESS** usando um modelo "
    "treinado e rastreado no **DagsHub**."
)

st.divider()

# ── Seção 1: Análise de texto único ─────────────────────────────────────────
st.subheader("Analisar uma manchete")

texto = st.text_area(
    label="Cole a manchete da notícia aqui:",
    height=120,
    placeholder='Ex: "New study shows the benefits of morning meditation"',
)

if st.button("Classificar notícia", type="primary", use_container_width=True):
    if texto.strip():
        # Faz a predição
        pred   = pipeline.predict([texto])[0]
        probas = pipeline.predict_proba([texto])[0]
        classes = pipeline.classes_

        emoji = EMOJI.get(pred, "")
        st.markdown(f"### {emoji} Categoria detectada: **{pred}**")

        st.markdown("##### Probabilidade por classe")
        for cls, prob in zip(classes, probas):
            st.progress(float(prob), text=f"{cls}: {prob:.0%}")
    else:
        st.warning("Digite ou cole um texto antes de analisar.")

st.divider()

# ── Seção 2: Análise em lote via CSV ────────────────────────────────────────
st.subheader("Análise em lote (CSV)")
st.markdown(
    "Envie um arquivo CSV com uma coluna chamada **`headline`**. "
    "O modelo vai classificar cada linha e você pode baixar o resultado."
)

uploaded_file = st.file_uploader(
    "Selecione o arquivo CSV",
    type=["csv"],
    help="O arquivo deve ter uma coluna chamada 'headline'",
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "headline" not in df.columns:
        st.error("O CSV precisa ter uma coluna chamada 'headline'.")
    else:
        # Classifica todas as linhas
        df["categoria"] = pipeline.predict(df["headline"].astype(str))

        # Exibe o resultado
        st.success(f"{len(df)} manchetes classificadas!")
        st.dataframe(df, use_container_width=True)

        # Estatísticas rápidas
        contagem = df["categoria"].value_counts()
        col1, col2, col3 = st.columns(3)
        col1.metric("Entertainment",  contagem.get("ENTERTAINMENT", 0))
        col2.metric("Politics",       contagem.get("POLITICS", 0))
        col3.metric("Style & Beauty", contagem.get("STYLE & BEAUTY", 0))
        
        col4, col5, _ = st.columns(3)
        col4.metric("Travel",         contagem.get("TRAVEL", 0))
        col5.metric("Wellness",       contagem.get("WELLNESS", 0))

        # Download do resultado
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar resultado como CSV",
            data=csv_bytes,
            file_name="resultado_categorias.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.divider()

if st.button("Sobre"):
    st.info(
        "Aplicação meramente didatica, inspirada no dataset: "
        "https://www.kaggle.com/datasets/rmisra/news-category-dataset "
        "e desenvolvida por: Eduardo Sardinha, Gabriel Martin, Jhonata dos santos e "
        "Volnei Klehm para a disciplina: Infra-Estrutura em Nuvem Ministrada pelo professor: Fábio Santos"
    )

st.caption(
    "Modelo: TF-IDF + Logistic Regression · "
    "Rastreamento: MLflow + DagsHub · "
    "Deploy: Docker + Render"
)
