"""
Hierarquia de exceções customizadas do ModelForge.

Este módulo define todas as exceções específicas do sistema,
organizadas em uma hierarquia clara para facilitar o tratamento de erros.
"""

from typing import Any, Dict, List, Optional


class ModelForgeException(Exception):
    """
    Exceção base para todas as exceções do ModelForge.
    
    Todas as exceções específicas do sistema herdam desta classe,
    permitindo captura genérica de erros do ModelForge.
    
    Attributes:
        message: Mensagem de erro
        details: Detalhes adicionais do erro
        original_exception: Exceção original que causou este erro
    
    Example:
        >>> try:
        ...     raise ModelForgeException("Erro genérico")
        ... except ModelForgeException as e:
        ...     print(e)
        Erro genérico
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ) -> None:
        """
        Inicializa a exceção.
        
        Args:
            message: Mensagem de erro
            details: Detalhes adicionais do erro
            original_exception: Exceção original que causou este erro
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_exception = original_exception
    
    def __str__(self) -> str:
        """Retorna representação string da exceção."""
        if self.details:
            details_str = ", ".join(
                f"{k}={v}" for k, v in self.details.items()
            )
            return f"{self.message} ({details_str})"
        return self.message
    
    def __repr__(self) -> str:
        """Retorna representação detalhada da exceção."""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"details={self.details})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte a exceção para um dicionário.
        
        Returns:
            Dict com informações da exceção
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ConfigValidationError(ModelForgeException):
    """
    Exceção para erros de validação de configuração.
    
    Lançada quando:
    - Arquivo YAML é inválido
    - Campos obrigatórios estão faltando
    - Valores estão fora do range permitido
    - Tipos de dados estão incorretos
    
    Attributes:
        field: Campo que causou o erro (se aplicável)
        errors: Lista de erros de validação
    
    Example:
        >>> raise ConfigValidationError(
        ...     "Campo 'model.name' é obrigatório",
        ...     field="model.name"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        errors: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ) -> None:
        """
        Inicializa a exceção de validação.
        
        Args:
            message: Mensagem de erro
            field: Campo que causou o erro
            errors: Lista de erros de validação
            details: Detalhes adicionais
            original_exception: Exceção original
        """
        details = details or {}
        if field:
            details["field"] = field
        if errors:
            details["errors"] = errors
        
        super().__init__(message, details, original_exception)
        self.field = field
        self.errors = errors or []


class TrainingError(ModelForgeException):
    """
    Exceção para erros durante o treinamento.
    
    Lançada quando:
    - Treinamento falha por falta de memória
    - Gradientes explodem (NaN/Inf)
    - Modelo não converge
    - Erro de GPU/TPU
    
    Attributes:
        epoch: Época em que o erro ocorreu
        step: Passo em que o erro ocorreu
        loss: Valor do loss quando o erro ocorreu
    
    Example:
        >>> raise TrainingError(
        ...     "Gradientes explodiram",
        ...     epoch=5,
        ...     step=1234
        ... )
    """
    
    def __init__(
        self,
        message: str,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        loss: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ) -> None:
        """
        Inicializa a exceção de treinamento.
        
        Args:
            message: Mensagem de erro
            epoch: Época do erro
            step: Passo do erro
            loss: Valor do loss
            details: Detalhes adicionais
            original_exception: Exceção original
        """
        details = details or {}
        if epoch is not None:
            details["epoch"] = epoch
        if step is not None:
            details["step"] = step
        if loss is not None:
            details["loss"] = loss
        
        super().__init__(message, details, original_exception)
        self.epoch = epoch
        self.step = step
        self.loss = loss


class InfrastructureError(ModelForgeException):
    """
    Exceção para erros de infraestrutura.
    
    Lançada quando:
    - GPU/TPU não está disponível
    - Falta de memória
    - Container não inicia
    - Erro de conexão com cloud
    
    Attributes:
        infrastructure_type: Tipo de infraestrutura (colab, cloud, container)
        resource: Recurso que causou o erro
    
    Example:
        >>> raise InfrastructureError(
        ...     "GPU não disponível",
        ...     infrastructure_type="colab",
        ...     resource="GPU"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        infrastructure_type: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ) -> None:
        """
        Inicializa a exceção de infraestrutura.
        
        Args:
            message: Mensagem de erro
            infrastructure_type: Tipo de infraestrutura
            resource: Recurso problemático
            details: Detalhes adicionais
            original_exception: Exceção original
        """
        details = details or {}
        if infrastructure_type:
            details["infrastructure_type"] = infrastructure_type
        if resource:
            details["resource"] = resource
        
        super().__init__(message, details, original_exception)
        self.infrastructure_type = infrastructure_type
        self.resource = resource


class BackendError(ModelForgeException):
    """
    Exceção para erros no backend de treinamento.
    
    Lançada quando:
    - Modelo não pode ser carregado
    - Dataset não encontrado
    - Tokenizer incompatível
    - Erro de autenticação no Hub
    
    Attributes:
        backend_type: Tipo de backend (huggingface, local, custom)
        operation: Operação que falhou
    
    Example:
        >>> raise BackendError(
        ...     "Modelo não encontrado no Hub",
        ...     backend_type="huggingface",
        ...     operation="load_model"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        backend_type: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ) -> None:
        """
        Inicializa a exceção de backend.
        
        Args:
            message: Mensagem de erro
            backend_type: Tipo de backend
            operation: Operação que falhou
            details: Detalhes adicionais
            original_exception: Exceção original
        """
        details = details or {}
        if backend_type:
            details["backend_type"] = backend_type
        if operation:
            details["operation"] = operation
        
        super().__init__(message, details, original_exception)
        self.backend_type = backend_type
        self.operation = operation


class ExportError(ModelForgeException):
    """
    Exceção para erros durante exportação.
    
    Lançada quando:
    - Docker build falha
    - Push para registry falha
    - Modelo não pode ser serializado
    - API não inicia
    
    Attributes:
        export_format: Formato de exportação (docker, onnx, etc.)
        stage: Estágio que falhou
    
    Example:
        >>> raise ExportError(
        ...     "Docker build falhou",
        ...     export_format="docker",
        ...     stage="build"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        export_format: Optional[str] = None,
        stage: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ) -> None:
        """
        Inicializa a exceção de exportação.
        
        Args:
            message: Mensagem de erro
            export_format: Formato de exportação
            stage: Estágio que falhou
            details: Detalhes adicionais
            original_exception: Exceção original
        """
        details = details or {}
        if export_format:
            details["export_format"] = export_format
        if stage:
            details["stage"] = stage
        
        super().__init__(message, details, original_exception)
        self.export_format = export_format
        self.stage = stage


class DatasetError(ModelForgeException):
    """
    Exceção para erros relacionados a datasets.
    
    Lançada quando:
    - Dataset não encontrado
    - Colunas inválidas
    - Erro de pré-processamento
    - Dataset corrompido
    
    Attributes:
        dataset_name: Nome do dataset
        split: Split que causou o erro
    """
    
    def __init__(
        self,
        message: str,
        dataset_name: Optional[str] = None,
        split: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ) -> None:
        """
        Inicializa a exceção de dataset.
        
        Args:
            message: Mensagem de erro
            dataset_name: Nome do dataset
            split: Split problemático
            details: Detalhes adicionais
            original_exception: Exceção original
        """
        details = details or {}
        if dataset_name:
            details["dataset_name"] = dataset_name
        if split:
            details["split"] = split
        
        super().__init__(message, details, original_exception)
        self.dataset_name = dataset_name
        self.split = split


class AuthenticationError(ModelForgeException):
    """
    Exceção para erros de autenticação.
    
    Lançada quando:
    - Token inválido ou expirado
    - Credenciais não encontradas
    - Acesso negado
    
    Attributes:
        service: Serviço que requer autenticação
    """
    
    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ) -> None:
        """
        Inicializa a exceção de autenticação.
        
        Args:
            message: Mensagem de erro
            service: Serviço problemático
            details: Detalhes adicionais
            original_exception: Exceção original
        """
        details = details or {}
        if service:
            details["service"] = service
        
        super().__init__(message, details, original_exception)
        self.service = service


class CheckpointError(ModelForgeException):
    """
    Exceção para erros de checkpoint.
    
    Lançada quando:
    - Checkpoint não pode ser salvo
    - Checkpoint corrompido
    - Upload para Hub falha
    
    Attributes:
        checkpoint_path: Caminho do checkpoint
    """
    
    def __init__(
        self,
        message: str,
        checkpoint_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ) -> None:
        """
        Inicializa a exceção de checkpoint.
        
        Args:
            message: Mensagem de erro
            checkpoint_path: Caminho do checkpoint
            details: Detalhes adicionais
            original_exception: Exceção original
        """
        details = details or {}
        if checkpoint_path:
            details["checkpoint_path"] = checkpoint_path
        
        super().__init__(message, details, original_exception)
        self.checkpoint_path = checkpoint_path

