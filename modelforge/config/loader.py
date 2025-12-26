"""
Classe ConfigLoader para carregamento e validação de arquivos YAML.

Este módulo fornece funcionalidades para carregar configurações YAML,
resolver variáveis de ambiente e validar contra o schema Pydantic.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import ValidationError

from modelforge.config.schema import Config, LoRAConfig
from modelforge.utils.exceptions import ConfigValidationError


class ConfigLoader:
    """
    Classe responsável pelo carregamento e validação de configurações YAML.
    
    A classe ConfigLoader oferece funcionalidades para:
    - Carregar arquivos YAML
    - Resolver variáveis de ambiente (${VAR_NAME})
    - Validar configurações contra o schema Pydantic
    - Fazer merge de configurações base com override
    
    Attributes:
        _env_pattern: Padrão regex para detectar variáveis de ambiente
    
    Example:
        >>> loader = ConfigLoader()
        >>> config = loader.load("config.yaml")
        >>> print(config.model.name)
        'bert-base-uncased'
    """
    
    # Padrão para detectar variáveis de ambiente: ${VAR_NAME}
    _env_pattern = re.compile(r'\$\{([^}]+)\}')
    
    def __init__(self) -> None:
        """Inicializa o ConfigLoader."""
        pass
    
    def load(self, config_path: Union[str, Path]) -> Config:
        """
        Carrega e valida um arquivo de configuração YAML.
        
        Args:
            config_path: Caminho para o arquivo YAML
            
        Returns:
            Config: Objeto de configuração validado
            
        Raises:
            ConfigValidationError: Se o arquivo não existir ou for inválido
            FileNotFoundError: Se o arquivo não for encontrado
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Arquivo de configuração não encontrado: {config_path}"
            )
        
        if not config_path.is_file():
            raise ConfigValidationError(
                f"O caminho especificado não é um arquivo: {config_path}"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigValidationError(
                f"Erro ao parsear YAML: {e}"
            ) from e
        
        if raw_config is None:
            raise ConfigValidationError("Arquivo YAML está vazio")
        
        # Resolve variáveis de ambiente
        resolved_config = self._resolve_env_variables(raw_config)
        
        # Valida e retorna o objeto Config
        return self._validate(resolved_config)
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> Config:
        """
        Carrega configuração a partir de um dicionário.
        
        Args:
            config_dict: Dicionário com a configuração
            
        Returns:
            Config: Objeto de configuração validado
        """
        resolved_config = self._resolve_env_variables(config_dict)
        return self._validate(resolved_config)
    
    def validate(self, config: Dict[str, Any]) -> bool:
        """
        Valida um dicionário de configuração sem criar o objeto Config.
        
        Args:
            config: Dicionário com a configuração
            
        Returns:
            bool: True se a configuração é válida
            
        Raises:
            ConfigValidationError: Se a configuração for inválida
        """
        try:
            resolved_config = self._resolve_env_variables(config)
            Config(**resolved_config)
            return True
        except ValidationError as e:
            raise ConfigValidationError(
                f"Configuração inválida: {self._format_validation_errors(e)}"
            ) from e
    
    def validate_file(self, config_path: Union[str, Path]) -> bool:
        """
        Valida um arquivo de configuração YAML.
        
        Args:
            config_path: Caminho para o arquivo YAML
            
        Returns:
            bool: True se a configuração é válida
        """
        config = self.load(config_path)
        return config is not None
    
    def merge_configs(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Faz merge de duas configurações, onde override tem precedência.
        
        O merge é feito de forma recursiva para dicionários aninhados.
        
        Args:
            base: Configuração base
            override: Configuração de override
            
        Returns:
            Dict[str, Any]: Configuração merged
        """
        result = base.copy()
        
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Merge recursivo para dicionários
                result[key] = self.merge_configs(result[key], value)
            else:
                # Sobrescreve o valor
                result[key] = value
        
        return result
    
    def _resolve_env_variables(self, config: Any) -> Any:
        """
        Resolve variáveis de ambiente em valores de configuração.
        
        Substitui padrões ${VAR_NAME} pelos valores das variáveis de ambiente.
        Se a variável não existir, mantém o padrão original.
        
        Args:
            config: Configuração (pode ser dict, list, ou valor)
            
        Returns:
            Configuração com variáveis resolvidas
        """
        if isinstance(config, dict):
            return {
                key: self._resolve_env_variables(value)
                for key, value in config.items()
            }
        elif isinstance(config, list):
            return [self._resolve_env_variables(item) for item in config]
        elif isinstance(config, str):
            return self._resolve_env_string(config)
        else:
            return config
    
    def _resolve_env_string(self, value: str) -> str:
        """
        Resolve variáveis de ambiente em uma string.
        
        Args:
            value: String que pode conter ${VAR_NAME}
            
        Returns:
            String com variáveis resolvidas
        """
        def replace_env(match: re.Match) -> str:
            var_name = match.group(1)
            # Suporta valores default: ${VAR_NAME:-default}
            if ':-' in var_name:
                var_name, default = var_name.split(':-', 1)
                return os.environ.get(var_name.strip(), default.strip())
            elif ':' in var_name:
                var_name, default = var_name.split(':', 1)
                return os.environ.get(var_name.strip(), default.strip())
            return os.environ.get(var_name.strip(), match.group(0))
        
        return self._env_pattern.sub(replace_env, value)
    
    def _validate(self, config: Dict[str, Any]) -> Config:
        """
        Valida e cria um objeto Config a partir de um dicionário.
        
        Args:
            config: Dicionário de configuração
            
        Returns:
            Config: Objeto de configuração validado
            
        Raises:
            ConfigValidationError: Se a validação falhar
        """
        try:
            return Config(**config)
        except ValidationError as e:
            raise ConfigValidationError(
                f"Configuração inválida:\n{self._format_validation_errors(e)}"
            ) from e
    
    def _format_validation_errors(self, error: ValidationError) -> str:
        """
        Formata erros de validação Pydantic de forma legível.
        
        Args:
            error: Erro de validação Pydantic
            
        Returns:
            String formatada com os erros
        """
        messages = []
        for err in error.errors():
            location = " -> ".join(str(loc) for loc in err["loc"])
            message = err["msg"]
            messages.append(f"  - {location}: {message}")
        return "\n".join(messages)
    
    @staticmethod
    def get_template() -> Dict[str, Any]:
        """
        Retorna um template de configuração básico.
        
        Returns:
            Dict com configuração template
        """
        return {
            "model": {
                "name": "bert-base-uncased",
                "version": "latest",
                "repository": "huggingface",
                "type": "transformer",
                "framework": "pytorch",
            },
            "dataset": {
                "name": "imdb",
                "repository": "huggingface",
                "splits": {
                    "train": "train",
                    "validation": "test",
                },
                "columns": {
                    "text": "text",
                    "label": "label",
                },
                "preprocessing": {
                    "max_length": 512,
                },
            },
            "training": {
                "batch_size": 16,
                "learning_rate": 2e-5,
                "epochs": 3,
                "scheduler": "linear",
            },
            "evaluation": {
                "metrics": ["accuracy", "f1"],
                "save_strategy": "epoch",
            },
            "checkpoints": {
                "save_dir": "./checkpoints",
                "max_to_keep": 3,
            },
            "versioning": {
                "push_to_hub": False,
            },
            "infrastructure": {
                "type": "local",
                "resources": {
                    "gpu": True,
                    "gpu_count": 1,
                },
            },
            "credentials": {
                "huggingface_token": "${HF_TOKEN}",
            },
            "export": {
                "format": "docker",
                "api": {
                    "framework": "flask",
                    "endpoints": ["chat/completions", "completions"],
                    "port": 8000,
                },
            },
        }

