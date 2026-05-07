# 📰 Classificador de Highlights de Notícias

Este é um projeto ponta-a-ponta de Processamento de Linguagem Natural (NLP) e MLOps. A aplicação classifica manchetes (highlights) de notícias em 6 categorias distintas usando um modelo de Machine Learning clássico (TF-IDF + Regressão Logística/Naive Bayes/etc).

As categorias suportadas são: **ENTERTAINMENT**, **POLITICS**, **STYLE & BEAUTY**, **TRAVEL**, **WELLNESS** e **OTHER**.

A infraestrutura de Machine Learning inclui rastreamento de experimentos e registro de modelos via **MLflow** hospedado no **DagsHub**, com empacotamento da aplicação final via **Docker** e interface web construída com **Streamlit**.

---

## 🚀 Fluxo de Trabalho (MLOps)

1. **Preparação dos Dados (`src/prepare_data.py`)**: Carrega o arquivo JSON original, limpa o texto, agrupa categorias minoritárias em `OTHER`, aplica undersampling para balancear as classes e divide os dados em treino/teste.
2. **Treinamento (`src/train.py`)**: Processa o texto com TF-IDF, treina o modelo, avalia e envia (log) as métricas, parâmetros e o modelo para o servidor MLflow do DagsHub.
3. **Avaliação/Download (`src/evaluate.py`)**: Conecta ao DagsHub, busca a última versão do modelo treinado e salva localmente como `model.pkl`.
4. **Interface (`app.py`)**: Carrega o `model.pkl` e sobe um servidor web (Streamlit) onde o usuário pode testar textos únicos ou enviar arquivos CSV para análise em lote.

---

## 📋 Pré-requisitos

1. **Python 3.11+** ou **Docker** instalados.
2. Uma conta no DagsHub e um repositório criado.
3. O arquivo original de dados de notícias: `News_Category_Dataset_v3.json` dentro da pasta `data/`.
4. Um arquivo `.env` na raiz do projeto com suas credenciais do DagsHub:

```env
DAGSHUB_USER="seu-usuario"
DAGSHUB_REPO="seu-repositorio"
DAGSHUB_TOKEN="seu-token-de-acesso"
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