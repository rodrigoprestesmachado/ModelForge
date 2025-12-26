"""
Classe DockerBuilder para build de imagens Docker.

Este módulo implementa o padrão Builder para construção
de imagens Docker para deploy de modelos.
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from modelforge.config.schema import ExportConfig
from modelforge.utils.exceptions import ExportError
from modelforge.utils.logging import StructuredLogger


class DockerBuilder:
    """
    Builder para construção de imagens Docker.
    
    Implementa o padrão Builder para:
    - Gerar Dockerfiles dinamicamente
    - Construir imagens Docker
    - Fazer push para registries
    
    Attributes:
        config: Configuração de exportação
        logger: Logger estruturado
    
    Example:
        >>> builder = DockerBuilder(config)
        >>> dockerfile = builder.generate_dockerfile(model_path)
        >>> image_id = builder.build_image("my-model", "v1.0")
    """
    
    # Template base do Dockerfile
    DOCKERFILE_TEMPLATE = '''# ModelForge API Docker Image
# Gerado automaticamente pelo ModelForge

FROM python:3.10-slim

# Metadados
LABEL maintainer="ModelForge"
LABEL version="{version}"
LABEL description="API de inferência para modelo ML"

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_PATH=/app/model
ENV API_PORT={port}
ENV API_HOST=0.0.0.0

# Diretório de trabalho
WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copia requirements e instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o modelo
COPY model/ /app/model/

# Copia código da API
COPY api/ /app/api/

# Expõe a porta
EXPOSE {port}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{port}/health || exit 1

# Comando de inicialização
CMD ["python", "-m", "gunicorn", "-w", "2", "-b", "0.0.0.0:{port}", "api.app:app"]
'''
    
    REQUIREMENTS_TEMPLATE = '''# Dependências para API de inferência
torch>=2.1.0
transformers>=4.36.0
flask>=3.0.0
gunicorn>=21.2.0
pydantic>=2.5.0
numpy>=1.26.0
'''
    
    def __init__(
        self,
        config: ExportConfig,
        logger: Optional[StructuredLogger] = None
    ) -> None:
        """
        Inicializa o DockerBuilder.
        
        Args:
            config: Configuração de exportação
            logger: Logger estruturado
        """
        self._config = config
        self._logger = logger or StructuredLogger("docker_builder")
        self._dockerfile_content: Optional[str] = None
    
    @property
    def config(self) -> ExportConfig:
        """Retorna configuração."""
        return self._config
    
    def generate_dockerfile(
        self,
        model_path: str,
        version: str = "1.0.0"
    ) -> str:
        """
        Gera o conteúdo do Dockerfile.
        
        Args:
            model_path: Caminho do modelo
            version: Versão do modelo
            
        Returns:
            Conteúdo do Dockerfile
        """
        self._logger.info("Gerando Dockerfile", model_path=model_path)
        
        port = self._config.api.port
        
        self._dockerfile_content = self.DOCKERFILE_TEMPLATE.format(
            version=version,
            port=port
        )
        
        return self._dockerfile_content
    
    def _create_dockerfile_content(self, model_path: str) -> str:
        """Cria conteúdo do Dockerfile (método interno)."""
        return self.generate_dockerfile(model_path)
    
    def prepare_build_context(
        self,
        model_path: str,
        output_dir: str
    ) -> str:
        """
        Prepara o contexto de build Docker.
        
        Copia modelo, gera Dockerfile e requirements.
        
        Args:
            model_path: Caminho do modelo
            output_dir: Diretório de saída
            
        Returns:
            Caminho do contexto de build
        """
        import shutil
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self._logger.info(
            "Preparando contexto de build",
            output_dir=output_dir
        )
        
        # Cria diretórios
        model_dest = output_path / "model"
        api_dest = output_path / "api"
        
        # Copia modelo
        if Path(model_path).exists():
            if model_dest.exists():
                shutil.rmtree(model_dest)
            shutil.copytree(model_path, model_dest)
        
        # Cria diretório da API
        api_dest.mkdir(exist_ok=True)
        
        # Gera código da API
        self._generate_api_code(api_dest)
        
        # Gera Dockerfile
        dockerfile_path = output_path / "Dockerfile"
        dockerfile_content = self.generate_dockerfile(model_path)
        dockerfile_path.write_text(dockerfile_content)
        
        # Gera requirements.txt
        requirements_path = output_path / "requirements.txt"
        requirements_path.write_text(self.REQUIREMENTS_TEMPLATE)
        
        self._logger.info("Contexto de build preparado", path=str(output_path))
        
        return str(output_path)
    
    def _generate_api_code(self, api_dir: Path) -> None:
        """
        Gera código Python da API.
        
        Args:
            api_dir: Diretório para salvar código
        """
        # app.py principal
        app_code = '''"""
API de Inferência ModelForge - Compatível com OpenAI
"""

import os
from flask import Flask, request, jsonify
from api.handlers import ChatCompletionsHandler, CompletionsHandler
from api.model_loader import ModelLoader

app = Flask(__name__)

# Carrega modelo
model_path = os.environ.get("MODEL_PATH", "/app/model")
model_loader = ModelLoader(model_path)
model, tokenizer = model_loader.load()

# Inicializa handlers
chat_handler = ChatCompletionsHandler(model, tokenizer)
completions_handler = CompletionsHandler(model, tokenizer)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    data = request.get_json()
    response = chat_handler.handle_request(data)
    return jsonify(response)


@app.route("/v1/completions", methods=["POST"])
def completions():
    """OpenAI-compatible completions endpoint."""
    data = request.get_json()
    response = completions_handler.handle_request(data)
    return jsonify(response)


@app.route("/v1/models", methods=["GET"])
def list_models():
    """Lista modelos disponíveis."""
    return jsonify({
        "object": "list",
        "data": [{
            "id": "modelforge-model",
            "object": "model",
            "owned_by": "modelforge"
        }]
    })


if __name__ == "__main__":
    port = int(os.environ.get("API_PORT", 8000))
    app.run(host="0.0.0.0", port=port)
'''
        
        # handlers.py
        handlers_code = '''"""
Handlers para endpoints da API.
"""

import time
import uuid
from typing import Any, Dict, List


class ChatCompletionsHandler:
    """Handler para /v1/chat/completions."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Processa requisição de chat."""
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 100)
        temperature = request.get("temperature", 1.0)
        
        # Formata mensagens
        prompt = self._format_messages(messages)
        
        # Gera resposta
        response_text = self._generate(prompt, max_tokens, temperature)
        
        return self._format_response(response_text)
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """Formata mensagens para o modelo."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\\n".join(formatted)
    
    def _generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Gera resposta do modelo."""
        import torch
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def _format_response(self, content: str) -> Dict[str, Any]:
        """Formata resposta no padrão OpenAI."""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "modelforge-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }


class CompletionsHandler:
    """Handler para /v1/completions."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Processa requisição de completion."""
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 100)
        temperature = request.get("temperature", 1.0)
        
        response_text = self._generate(prompt, max_tokens, temperature)
        
        return self._format_response(prompt, response_text)
    
    def _generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Gera completion."""
        import torch
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def _format_response(self, prompt: str, text: str) -> Dict[str, Any]:
        """Formata resposta no padrão OpenAI."""
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "modelforge-model",
            "choices": [{
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
'''
        
        # model_loader.py
        model_loader_code = '''"""
Carregador de modelo para API.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelLoader:
    """Carrega modelo e tokenizer."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
    
    def load(self):
        """Carrega modelo e tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        
        # Garante pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        return self.model, self.tokenizer
'''
        
        # __init__.py
        init_code = '"""API ModelForge."""\n'
        
        # Escreve arquivos
        (api_dir / "__init__.py").write_text(init_code)
        (api_dir / "app.py").write_text(app_code)
        (api_dir / "handlers.py").write_text(handlers_code)
        (api_dir / "model_loader.py").write_text(model_loader_code)
    
    def build_image(
        self,
        image_name: str,
        tag: str = "latest",
        build_context: Optional[str] = None,
        no_cache: bool = False,
    ) -> str:
        """
        Constrói a imagem Docker.
        
        Args:
            image_name: Nome da imagem
            tag: Tag da imagem
            build_context: Caminho do contexto de build
            no_cache: Se deve ignorar cache
            
        Returns:
            ID da imagem construída
        """
        full_name = f"{image_name}:{tag}"
        
        self._logger.info(
            "Construindo imagem Docker",
            image=full_name
        )
        
        if not self._check_docker():
            raise ExportError(
                "Docker não está disponível",
                export_format="docker",
                stage="build"
            )
        
        # Prepara comando
        cmd = ["docker", "build", "-t", full_name]
        
        if no_cache:
            cmd.append("--no-cache")
        
        context = build_context or "."
        cmd.append(context)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            self._logger.info(
                "Imagem construída com sucesso",
                image=full_name
            )
            
            return full_name
            
        except subprocess.CalledProcessError as e:
            raise ExportError(
                f"Falha ao construir imagem: {e.stderr}",
                export_format="docker",
                stage="build",
                original_exception=e
            )
    
    def push_image(
        self,
        image_name: str,
        registry: Optional[str] = None
    ) -> None:
        """
        Faz push da imagem para um registry.
        
        Args:
            image_name: Nome da imagem (com tag)
            registry: URL do registry
        """
        if registry:
            full_name = f"{registry}/{image_name}"
            # Tag para o registry
            subprocess.run(
                ["docker", "tag", image_name, full_name],
                check=True
            )
        else:
            full_name = image_name
        
        self._logger.info("Fazendo push da imagem", image=full_name)
        
        try:
            subprocess.run(
                ["docker", "push", full_name],
                check=True
            )
            self._logger.info("Push concluído", image=full_name)
            
        except subprocess.CalledProcessError as e:
            raise ExportError(
                f"Falha ao fazer push: {e}",
                export_format="docker",
                stage="push",
                original_exception=e
            )
    
    def _check_docker(self) -> bool:
        """Verifica se Docker está disponível."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def __repr__(self) -> str:
        """Representação string."""
        return (
            f"DockerBuilder("
            f"port={self._config.api.port}, "
            f"format='{self._config.format}')"
        )

