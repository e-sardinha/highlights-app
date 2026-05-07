"""
Ingestão de dados Supabase -> DuckDB.
O arquivo `ingestion.py` é responsável por extrair os dados do Supabase, transformá-los e carregá-los em um banco de dados DuckDB local. Ele inclui:
- Conexão com o Supabase usando as credenciais do arquivo .env
- Consulta SQL para extrair os dados necessários
- Transformação dos dados (limpeza, normalização, etc.)
- Carregamento dos dados transformados no DuckDB para uso posterior no treinamento e avaliação do modelo.
"""
import duckdb