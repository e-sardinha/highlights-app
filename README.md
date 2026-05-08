# 📰 Classificador de Highlights de Notícias

Este é um projeto ponta-a-ponta de Processamento de Linguagem Natural (NLP) e MLOps. A aplicação classifica manchetes (highlights) de notícias em 5 categorias distintas usando um modelo de Machine Learning clássico (TF-IDF + Regressão Logística/Naive Bayes/etc).

As categorias suportadas são: **ENTERTAINMENT**, **POLITICS**, **STYLE & BEAUTY**, **TRAVEL** e **WELLNESS**.

A infraestrutura de Machine Learning inclui rastreamento de experimentos e registro de modelos via **MLflow** hospedado no **DagsHub**, com empacotamento da aplicação final via **Docker** e interface web construída com **Streamlit**.

---

## 🚀 Fluxo de Trabalho (MLOps)

1. **Ingestão e Preparação dos Dados (`src/ingestion.py` e `src/preprocessing.py`)**: Extrai os dados do **Supabase**, cria uma camada raw em **Parquet**, realiza a limpeza utilizando **DuckDB** e divide os dados persistindo os splits em disco.
2. **Treinamento (`src/train.py`)**: Processa o texto com TF-IDF, treina o modelo, avalia e envia (log) as métricas, parâmetros e o modelo para o servidor MLflow do DagsHub.
3. **Avaliação/Download (`src/evaluate.py`)**: Conecta ao DagsHub, busca a última versão do modelo treinado e salva localmente como `model.pkl`.
4. **Interface (`app.py`)**: Carrega o `model.pkl` e sobe um servidor web (Streamlit) onde o usuário pode testar textos únicos ou enviar arquivos CSV para análise em lote.

---

## 📋 Pré-requisitos

1. **Python 3.11+** ou **Docker** instalados.
2. Uma conta no DagsHub e um repositório criado.
3. Um banco de dados configurado no **Supabase** com a tabela `news_category` contendo os dados.
4. Um arquivo `.env` na raiz do projeto com suas credenciais do DagsHub e Supabase:

```env
DAGSHUB_USER="seu-usuario"
DAGSHUB_REPO="seu-repositorio"
DAGSHUB_TOKEN="seu-token-de-acesso"
SUPABASE_URL="sua-url-do-supabase"
SUPABASE_KEY="sua-chave-anon-do-supabase"
```

---

## 🛠️ Como usar (Ambiente Local)

### 1. Instale as dependências
Recomenda-se o uso de um ambiente virtual (venv).
```bash
pip install -r requirements.txt
```

### 2. Treine o modelo
Execute o script de treino. Ele cuidará do processamento e enviará tudo para o DagsHub.
*(Você pode alterar os hiperparâmetros dentro de `src/train.py` para realizar novos experimentos)*
```bash
python src/train.py
```

### 3. Baixe o modelo mais recente
Busca o modelo treinado diretamente do DagsHub e salva na raiz como `model.pkl`.
```bash
python src/evaluate.py
```

### 4. Inicie o App Streamlit
Acesse a aplicação no navegador em `http://localhost:8501`.
```bash
streamlit run app.py
```

---

## 🐳 Como usar (Docker)

Em vez de rodar localmente pelo Python, você pode isolar toda a aplicação de produção dentro de um contêiner Docker. O script de build se encarregará de usar suas credenciais do `.env` temporariamente para baixar o modelo e construir a imagem.

### 1. Construir a Imagem
Dê permissão de execução ao script (se ainda não tiver feito) e rode o construtor:
```bash
chmod +x build_docker.sh
./build_docker.sh
```

### 2. Executar o Contêiner
Inicie a aplicação utilizando o script run:
```bash
chmod +x run.sh
./run.sh
```

Acesse no seu navegador: `http://localhost:8501`

---

## 📂 Estrutura do Projeto

- `/app.py`: Interface principal do Streamlit.
- `/src`: Código-fonte (preparação de dados, treino, avaliação).
- `/data`: Local onde o dataset `.json` original deve ser colocado.
- `/Dockerfile`: Instruções para montar a imagem de produção.
- `/build_docker.sh` e `/run.sh`: Scripts utilitários para facilitar o uso do Docker.

---

## 👥 Autores

Desenvolvido por **Eduardo Sardinha**, **Gabriel Martin**, **Jhonata dos santos** e **Volnei Klehm** para a disciplina de Infra-Estrutura em Nuvem (Ministrada pelo professor Fábio Santos).