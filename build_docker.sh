#!/bin/bash

# Verifica se o arquivo .env existe
if [ ! -f .env ]; then
    echo "Erro: Arquivo .env não encontrado na raiz do projeto."
    exit 1
fi

# Carrega as variáveis do arquivo .env de forma segura
set -a
source .env
set +a

echo "Iniciando o build da imagem Docker 'highlights-app'..."
docker build \
    --build-arg DAGSHUB_USER="$DAGSHUB_USER" \
    --build-arg DAGSHUB_REPO="$DAGSHUB_REPO" \
    --build-arg DAGSHUB_TOKEN="$DAGSHUB_TOKEN" \
    -t highlights-app .

echo "Build finalizado!"