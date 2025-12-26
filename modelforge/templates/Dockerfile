# ============================================================================
# ModelForge API Docker Image
# Gerado automaticamente pelo ModelForge
# ============================================================================

# Imagem base Python
FROM python:3.10-slim

# Metadados
LABEL maintainer="ModelForge"
LABEL version="1.0.0"
LABEL description="API de inferência para modelo ML fine-tuned com ModelForge"

# ============================================================================
# Variáveis de ambiente
# ============================================================================
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_PATH=/app/model
ENV API_PORT=8000
ENV API_HOST=0.0.0.0

# ============================================================================
# Configuração do sistema
# ============================================================================
WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Dependências Python
# ============================================================================
# Copia requirements primeiro para aproveitar cache do Docker
COPY requirements.txt .

# Instala dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Código da aplicação
# ============================================================================
# Copia modelo treinado
COPY model/ /app/model/

# Copia código da API
COPY api/ /app/api/

# ============================================================================
# Configuração de rede
# ============================================================================
# Expõe porta da API
EXPOSE ${API_PORT}

# ============================================================================
# Health check
# ============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

# ============================================================================
# Comando de inicialização
# ============================================================================
# Usa gunicorn para produção com 2 workers
CMD ["python", "-m", "gunicorn", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "api.app:app"]

