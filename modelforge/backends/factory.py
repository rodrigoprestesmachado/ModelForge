"""
Classe BackendFactory para criação de backends.

Este módulo implementa o padrão Factory para criar instâncias
de backends de forma centralizada e extensível.
"""

from typing import Dict, Optional, Type

from modelforge.backends.base import BackendBase
from modelforge.backends.huggingface import HuggingFaceBackend
from modelforge.utils.exceptions import BackendError
from modelforge.utils.logging import StructuredLogger


class BackendFactory:
    """
    Factory para criação de backends de treinamento.
    
    Implementa o padrão Factory Method para criar instâncias
    de backends baseado no tipo especificado.
    
    A factory mantém um registro de backends disponíveis,
    permitindo fácil extensão com novos backends.
    
    Attributes:
        _backends: Dicionário mapeando tipos para classes de backend
    
    Example:
        >>> backend = BackendFactory.create_backend(
        ...     "huggingface",
        ...     {"HF_TOKEN": "hf_..."}
        ... )
        >>> print(type(backend))
        <class 'HuggingFaceBackend'>
    """
    
    # Registro de backends disponíveis
    _backends: Dict[str, Type[BackendBase]] = {
        "huggingface": HuggingFaceBackend,
        "hf": HuggingFaceBackend,  # Alias
    }
    
    _logger = StructuredLogger("backend_factory")
    
    @classmethod
    def create_backend(
        cls,
        backend_type: str,
        credentials: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> BackendBase:
        """
        Cria uma instância de backend.
        
        Args:
            backend_type: Tipo do backend (huggingface, hf, etc.)
            credentials: Credenciais para o backend
            **kwargs: Argumentos adicionais para o backend
            
        Returns:
            BackendBase: Instância do backend
            
        Raises:
            BackendError: Se o tipo de backend não for suportado
        """
        backend_type = backend_type.lower()
        
        if backend_type not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise BackendError(
                f"Backend '{backend_type}' não suportado. "
                f"Backends disponíveis: {available}",
                backend_type=backend_type,
                operation="create_backend"
            )
        
        backend_class = cls._backends[backend_type]
        
        cls._logger.info(
            "Criando backend",
            backend_type=backend_type,
            backend_class=backend_class.__name__
        )
        
        return backend_class(credentials=credentials, **kwargs)
    
    @classmethod
    def register_backend(
        cls,
        name: str,
        backend_class: Type[BackendBase]
    ) -> None:
        """
        Registra um novo tipo de backend.
        
        Permite extensão do sistema com backends customizados.
        
        Args:
            name: Nome do backend para registro
            backend_class: Classe do backend (deve herdar de BackendBase)
            
        Raises:
            ValueError: Se a classe não herdar de BackendBase
        """
        if not issubclass(backend_class, BackendBase):
            raise ValueError(
                f"A classe {backend_class.__name__} deve herdar de BackendBase"
            )
        
        cls._backends[name.lower()] = backend_class
        cls._logger.info(
            "Backend registrado",
            name=name,
            backend_class=backend_class.__name__
        )
    
    @classmethod
    def unregister_backend(cls, name: str) -> None:
        """
        Remove um backend do registro.
        
        Args:
            name: Nome do backend a remover
        """
        name = name.lower()
        if name in cls._backends:
            del cls._backends[name]
            cls._logger.info("Backend removido", name=name)
    
    @classmethod
    def list_backends(cls) -> Dict[str, str]:
        """
        Lista todos os backends disponíveis.
        
        Returns:
            Dict mapeando nomes para nomes de classes
        """
        return {
            name: backend_class.__name__
            for name, backend_class in cls._backends.items()
        }
    
    @classmethod
    def get_backend_class(cls, backend_type: str) -> Type[BackendBase]:
        """
        Obtém a classe de um backend sem instanciar.
        
        Args:
            backend_type: Tipo do backend
            
        Returns:
            Classe do backend
            
        Raises:
            BackendError: Se o backend não existir
        """
        backend_type = backend_type.lower()
        
        if backend_type not in cls._backends:
            raise BackendError(
                f"Backend '{backend_type}' não encontrado",
                backend_type=backend_type
            )
        
        return cls._backends[backend_type]
    
    @classmethod
    def is_backend_available(cls, backend_type: str) -> bool:
        """
        Verifica se um backend está disponível.
        
        Args:
            backend_type: Tipo do backend
            
        Returns:
            True se o backend está registrado
        """
        return backend_type.lower() in cls._backends

