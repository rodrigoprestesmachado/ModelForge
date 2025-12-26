"""
Classe CredentialManager para gerenciamento seguro de credenciais.

Este módulo implementa o padrão Singleton para garantir que
as credenciais sejam carregadas e gerenciadas de forma centralizada e segura.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from modelforge.config.schema import CredentialsConfig
from modelforge.utils.exceptions import ConfigValidationError


class CredentialManager:
    """
    Gerenciador de credenciais usando padrão Singleton.
    
    Esta classe é responsável por:
    - Carregar credenciais de variáveis de ambiente
    - Carregar credenciais de arquivos .env
    - Validar tokens do Hugging Face
    - Fornecer acesso seguro a credenciais
    
    O padrão Singleton garante que exista apenas uma instância
    do gerenciador em toda a aplicação.
    
    Attributes:
        _instance: Instância única do Singleton
        _credentials: Dicionário com credenciais carregadas
        _env_loaded: Flag indicando se .env foi carregado
    
    Example:
        >>> manager = CredentialManager()
        >>> token = manager.get_secret("HF_TOKEN")
        >>> print(token)
        'hf_xxxxx'
    """
    
    _instance: Optional["CredentialManager"] = None
    _credentials: Dict[str, str]
    _env_loaded: bool
    
    def __new__(cls, env_file: Optional[str] = None) -> "CredentialManager":
        """
        Implementa o padrão Singleton.
        
        Args:
            env_file: Caminho para arquivo .env (opcional)
            
        Returns:
            CredentialManager: Instância única do gerenciador
        """
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._credentials = {}
            instance._env_loaded = False
            instance._initialize(env_file)
            cls._instance = instance
        return cls._instance
    
    def _initialize(self, env_file: Optional[str] = None) -> None:
        """
        Inicializa o gerenciador de credenciais.
        
        Args:
            env_file: Caminho para arquivo .env
        """
        self._load_env_file(env_file)
        self._credentials = self._load_from_env()
    
    def _load_env_file(self, env_file: Optional[str] = None) -> None:
        """
        Carrega variáveis de um arquivo .env.
        
        Args:
            env_file: Caminho para o arquivo .env
        """
        if self._env_loaded:
            return
        
        # Tenta carregar .env do diretório atual ou do caminho especificado
        env_paths = []
        
        if env_file:
            env_paths.append(Path(env_file))
        
        env_paths.extend([
            Path.cwd() / ".env",
            Path.home() / ".modelforge" / ".env",
        ])
        
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                self._env_loaded = True
                break
    
    def _load_from_env(self) -> Dict[str, str]:
        """
        Carrega credenciais de variáveis de ambiente.
        
        Returns:
            Dict com credenciais encontradas
        """
        credentials = {}
        
        # Lista de variáveis de ambiente conhecidas
        known_vars = [
            "HF_TOKEN",
            "HUGGINGFACE_TOKEN",
            "HF_USERNAME",
            "WANDB_API_KEY",
            "DOCKER_REGISTRY",
            "DOCKER_USERNAME",
            "DOCKER_PASSWORD",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GCP_PROJECT_ID",
            "AZURE_SUBSCRIPTION_ID",
        ]
        
        for var in known_vars:
            value = os.environ.get(var)
            if value:
                credentials[var] = value
        
        return credentials
    
    def load_credentials(
        self,
        config: Optional[CredentialsConfig] = None
    ) -> Dict[str, str]:
        """
        Carrega credenciais a partir de um objeto de configuração.
        
        Combina credenciais do ambiente com as especificadas na configuração.
        
        Args:
            config: Objeto CredentialsConfig com credenciais adicionais
            
        Returns:
            Dict com todas as credenciais
        """
        credentials = self._credentials.copy()
        
        if config:
            if config.huggingface_token:
                # Remove o padrão ${...} se ainda existir
                token = config.huggingface_token
                if not token.startswith("${"):
                    credentials["HF_TOKEN"] = token
            
            if config.wandb_api_key:
                key = config.wandb_api_key
                if not key.startswith("${"):
                    credentials["WANDB_API_KEY"] = key
            
            # Adiciona credenciais adicionais
            if config.additional:
                for key, value in config.additional.items():
                    if not value.startswith("${"):
                        credentials[key] = value
        
        return credentials
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Obtém uma credencial específica.
        
        Args:
            key: Nome da credencial
            default: Valor padrão se não encontrada
            
        Returns:
            Valor da credencial ou default
        """
        # Primeiro tenta no cache
        if key in self._credentials:
            return self._credentials[key]
        
        # Depois tenta nas variáveis de ambiente
        value = os.environ.get(key)
        if value:
            self._credentials[key] = value
            return value
        
        return default
    
    def set_secret(self, key: str, value: str) -> None:
        """
        Define uma credencial no cache.
        
        Args:
            key: Nome da credencial
            value: Valor da credencial
        """
        self._credentials[key] = value
    
    def validate_hf_token(self, token: Optional[str] = None) -> bool:
        """
        Valida um token do Hugging Face Hub.
        
        Args:
            token: Token a validar (usa HF_TOKEN se não fornecido)
            
        Returns:
            bool: True se o token é válido
            
        Raises:
            ConfigValidationError: Se o token for inválido
        """
        if token is None:
            token = self.get_secret("HF_TOKEN")
        
        if not token:
            raise ConfigValidationError(
                "Token do Hugging Face não encontrado. "
                "Defina a variável de ambiente HF_TOKEN ou "
                "configure credentials.huggingface_token no YAML."
            )
        
        # Validação básica do formato
        if not token.startswith("hf_"):
            raise ConfigValidationError(
                "Token do Hugging Face inválido. "
                "O token deve começar com 'hf_'."
            )
        
        # Validação online (opcional, pode ser implementada com huggingface_hub)
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            api.whoami()
            return True
        except ImportError:
            # Se huggingface_hub não estiver disponível, aceita o token
            return True
        except Exception as e:
            raise ConfigValidationError(
                f"Token do Hugging Face inválido ou expirado: {e}"
            ) from e
    
    def get_hf_token(self) -> Optional[str]:
        """
        Obtém o token do Hugging Face.
        
        Returns:
            Token do HF ou None
        """
        return self.get_secret("HF_TOKEN") or self.get_secret("HUGGINGFACE_TOKEN")
    
    def has_credentials_for(self, service: str) -> bool:
        """
        Verifica se há credenciais para um serviço específico.
        
        Args:
            service: Nome do serviço (huggingface, wandb, aws, gcp, azure)
            
        Returns:
            bool: True se credenciais estão disponíveis
        """
        service_credentials = {
            "huggingface": ["HF_TOKEN", "HUGGINGFACE_TOKEN"],
            "wandb": ["WANDB_API_KEY"],
            "aws": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
            "gcp": ["GOOGLE_APPLICATION_CREDENTIALS", "GCP_PROJECT_ID"],
            "azure": ["AZURE_SUBSCRIPTION_ID"],
            "docker": ["DOCKER_USERNAME", "DOCKER_PASSWORD"],
        }
        
        required = service_credentials.get(service.lower(), [])
        return any(self.get_secret(key) for key in required)
    
    def clear(self) -> None:
        """Limpa o cache de credenciais."""
        self._credentials.clear()
    
    @classmethod
    def reset(cls) -> None:
        """
        Reseta a instância Singleton.
        
        Útil para testes ou para recarregar credenciais.
        """
        cls._instance = None
    
    def __repr__(self) -> str:
        """Representação string do gerenciador."""
        keys = list(self._credentials.keys())
        return f"CredentialManager(credentials={keys})"

