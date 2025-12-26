"""
Classes para API Flask compatível com OpenAI.

Este módulo implementa os handlers e servidor de API
para servir modelos com endpoints compatíveis com OpenAI.
"""

import time
import uuid
from typing import Any, Dict, List, Optional

from modelforge.config.schema import ExportConfig
from modelforge.utils.logging import StructuredLogger


class OpenAIChatCompletionsHandler:
    """
    Handler para endpoint /v1/chat/completions.
    
    Implementa o formato de requisição e resposta
    compatível com a API de Chat da OpenAI.
    
    Attributes:
        model: Modelo de linguagem
        tokenizer: Tokenizer do modelo
    
    Example:
        >>> handler = OpenAIChatCompletionsHandler(model, tokenizer)
        >>> response = handler.handle_request({
        ...     "messages": [{"role": "user", "content": "Hello"}]
        ... })
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        logger: Optional[StructuredLogger] = None
    ) -> None:
        """
        Inicializa o handler.
        
        Args:
            model: Modelo de linguagem
            tokenizer: Tokenizer
            logger: Logger estruturado
        """
        self._model = model
        self._tokenizer = tokenizer
        self._logger = logger or StructuredLogger("chat_handler")
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa uma requisição de chat completion.
        
        Args:
            request: Requisição no formato OpenAI
            
        Returns:
            Resposta no formato OpenAI
        """
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 100)
        temperature = request.get("temperature", 1.0)
        model_id = request.get("model", "modelforge-model")
        
        self._logger.debug(
            "Processando chat request",
            messages_count=len(messages),
            max_tokens=max_tokens
        )
        
        # Gera resposta
        response_text = self._generate_response(messages, max_tokens, temperature)
        
        # Formata resposta
        return self._format_response(response_text, model_id)
    
    def _generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> str:
        """
        Gera resposta do modelo.
        
        Args:
            messages: Lista de mensagens do chat
            max_tokens: Número máximo de tokens
            temperature: Temperatura de sampling
            
        Returns:
            Texto gerado
        """
        import torch
        
        # Formata prompt
        prompt = self._format_messages(messages)
        
        # Tokeniza
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        # Move para o device do modelo
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Gera
        with torch.no_grad():
            if hasattr(self._model, 'generate'):
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
                
                response = self._tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
            else:
                # Para modelos de classificação, retorna label
                outputs = self._model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1).item()
                response = f"Predicted class: {predicted_class}"
        
        return response
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Formata mensagens para o modelo.
        
        Args:
            messages: Lista de mensagens
            
        Returns:
            Prompt formatado
        """
        formatted_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        formatted_parts.append("Assistant:")
        
        return "\n".join(formatted_parts)
    
    def _format_response(
        self,
        content: str,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Formata resposta no padrão OpenAI.
        
        Args:
            content: Conteúdo da resposta
            model_id: ID do modelo
            
        Returns:
            Resposta formatada
        """
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content.strip()
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # Poderia calcular
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }


class OpenAICompletionsHandler:
    """
    Handler para endpoint /v1/completions.
    
    Implementa o formato de requisição e resposta
    compatível com a API de Completions da OpenAI.
    
    Attributes:
        model: Modelo de linguagem
        tokenizer: Tokenizer do modelo
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        logger: Optional[StructuredLogger] = None
    ) -> None:
        """
        Inicializa o handler.
        
        Args:
            model: Modelo de linguagem
            tokenizer: Tokenizer
            logger: Logger estruturado
        """
        self._model = model
        self._tokenizer = tokenizer
        self._logger = logger or StructuredLogger("completions_handler")
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa uma requisição de completion.
        
        Args:
            request: Requisição no formato OpenAI
            
        Returns:
            Resposta no formato OpenAI
        """
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 100)
        temperature = request.get("temperature", 1.0)
        model_id = request.get("model", "modelforge-model")
        
        self._logger.debug(
            "Processando completion request",
            prompt_length=len(prompt),
            max_tokens=max_tokens
        )
        
        # Gera completion
        completion_text = self._generate_completion(prompt, max_tokens, temperature)
        
        # Formata resposta
        return self._format_response(completion_text, model_id)
    
    def _generate_completion(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """
        Gera completion do modelo.
        
        Args:
            prompt: Prompt de entrada
            max_tokens: Número máximo de tokens
            temperature: Temperatura de sampling
            
        Returns:
            Texto gerado
        """
        import torch
        
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if hasattr(self._model, 'generate'):
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
                
                response = self._tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
            else:
                outputs = self._model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1).item()
                response = f"Predicted class: {predicted_class}"
        
        return response
    
    def _format_response(
        self,
        text: str,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Formata resposta no padrão OpenAI.
        
        Args:
            text: Texto gerado
            model_id: ID do modelo
            
        Returns:
            Resposta formatada
        """
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "text": text.strip(),
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }


class ModelAPIServer:
    """
    Servidor de API Flask para modelos.
    
    Cria e configura uma aplicação Flask com endpoints
    compatíveis com a API da OpenAI.
    
    Attributes:
        model_path: Caminho do modelo
        config: Configuração de exportação
        app: Aplicação Flask
    
    Example:
        >>> server = ModelAPIServer(model_path, config)
        >>> app = server.create_app()
        >>> server.run(host="0.0.0.0", port=8000)
    """
    
    def __init__(
        self,
        model_path: str,
        config: ExportConfig,
        logger: Optional[StructuredLogger] = None
    ) -> None:
        """
        Inicializa o servidor.
        
        Args:
            model_path: Caminho do modelo
            config: Configuração de exportação
            logger: Logger estruturado
        """
        self._model_path = model_path
        self._config = config
        self._logger = logger or StructuredLogger("api_server")
        self._app = None
        self._model = None
        self._tokenizer = None
        self._chat_handler = None
        self._completions_handler = None
    
    @property
    def app(self):
        """Retorna aplicação Flask."""
        return self._app
    
    def create_app(self):
        """
        Cria e configura a aplicação Flask.
        
        Returns:
            Aplicação Flask configurada
        """
        from flask import Flask, request, jsonify
        
        self._logger.info("Criando aplicação Flask")
        
        app = Flask(__name__)
        
        # Carrega modelo
        self._load_model()
        
        # Inicializa handlers
        self._chat_handler = OpenAIChatCompletionsHandler(
            self._model, self._tokenizer, self._logger
        )
        self._completions_handler = OpenAICompletionsHandler(
            self._model, self._tokenizer, self._logger
        )
        
        # Registra rotas
        self._register_routes(app)
        
        self._app = app
        return app
    
    def _load_model(self) -> None:
        """Carrega modelo e tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
        
        self._logger.info("Carregando modelo", path=self._model_path)
        
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        
        # Tenta carregar como modelo causal, senão como modelo genérico
        try:
            self._model = AutoModelForCausalLM.from_pretrained(self._model_path)
        except Exception:
            self._model = AutoModel.from_pretrained(self._model_path)
        
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        self._logger.info("Modelo carregado")
    
    def _register_routes(self, app) -> None:
        """
        Registra rotas na aplicação Flask.
        
        Args:
            app: Aplicação Flask
        """
        from flask import request, jsonify
        
        @app.route("/health", methods=["GET"])
        def health():
            """Health check."""
            return jsonify({"status": "healthy"})
        
        @app.route("/v1/chat/completions", methods=["POST"])
        def chat_completions():
            """Chat completions endpoint."""
            data = request.get_json()
            response = self._chat_handler.handle_request(data)
            return jsonify(response)
        
        @app.route("/v1/completions", methods=["POST"])
        def completions():
            """Completions endpoint."""
            data = request.get_json()
            response = self._completions_handler.handle_request(data)
            return jsonify(response)
        
        @app.route("/v1/models", methods=["GET"])
        def list_models():
            """Lista modelos disponíveis."""
            return jsonify({
                "object": "list",
                "data": [
                    {
                        "id": "modelforge-model",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "modelforge"
                    }
                ]
            })
        
        self._logger.info("Rotas registradas")
    
    def run(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        debug: bool = False
    ) -> None:
        """
        Inicia o servidor.
        
        Args:
            host: Host para bind
            port: Porta para bind
            debug: Modo debug
        """
        if self._app is None:
            self.create_app()
        
        host = host or self._config.api.host
        port = port or self._config.api.port
        
        self._logger.info(
            "Iniciando servidor",
            host=host,
            port=port
        )
        
        self._app.run(host=host, port=port, debug=debug)
    
    def __repr__(self) -> str:
        """Representação string."""
        return (
            f"ModelAPIServer("
            f"model_path='{self._model_path}', "
            f"port={self._config.api.port})"
        )

