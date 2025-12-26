# ============================================================================
# ModelForge - Dockerfile para desenvolvimento e CI/CD
# ============================================================================

FROM python:3.10-slim

# Metadados
LABEL maintainer="ModelForge Team"
LABEL version="0.1.0"

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Diretório de trabalho
WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements primeiro (melhor cache)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copia código fonte
COPY . .

# Instala o pacote em modo editável
RUN pip install -e .

# Expõe porta default da API
EXPOSE 8000

# Comando padrão
CMD ["modelforge", "--help"]

