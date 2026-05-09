@echo off
setlocal EnableExtensions DisableDelayedExpansion

if not exist ".env" (
    echo Erro: Arquivo .env nao encontrado na raiz do projeto.
    exit /b 1
)

set "DAGSHUB_USER="
set "DAGSHUB_REPO="
set "DAGSHUB_TOKEN="

for /f "usebackq tokens=1,* delims==" %%A in (`findstr /R /V "^[ ]*# ^[ ]*$" ".env"`) do (
    if /I "%%A"=="DAGSHUB_USER" set "DAGSHUB_USER=%%B"
    if /I "%%A"=="DAGSHUB_REPO" set "DAGSHUB_REPO=%%B"
    if /I "%%A"=="DAGSHUB_TOKEN" set "DAGSHUB_TOKEN=%%B"
)

if "%DAGSHUB_USER%"=="" (
    echo Erro: DAGSHUB_USER nao definido no .env.
    exit /b 1
)

if "%DAGSHUB_REPO%"=="" (
    echo Erro: DAGSHUB_REPO nao definido no .env.
    exit /b 1
)

if "%DAGSHUB_TOKEN%"=="" (
    echo Erro: DAGSHUB_TOKEN nao definido no .env.
    exit /b 1
)

echo Iniciando o build da imagem Docker highlights-app...
docker build ^
  --build-arg "DAGSHUB_USER=%DAGSHUB_USER%" ^
  --build-arg "DAGSHUB_REPO=%DAGSHUB_REPO%" ^
  --build-arg "DAGSHUB_TOKEN=%DAGSHUB_TOKEN%" ^
  -t highlights-app .

if errorlevel 1 (
    echo Build falhou.
    exit /b 1
)

echo Build finalizado!
exit /b 0
