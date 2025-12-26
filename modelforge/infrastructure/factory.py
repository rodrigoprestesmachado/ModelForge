"""
Classe InfrastructureFactory para criação de infraestruturas.

Este módulo implementa o padrão Factory para criar instâncias
de infraestruturas de forma centralizada.
"""

from typing import Dict, Optional, Type

from modelforge.config.schema import InfrastructureConfig
from modelforge.infrastructure.base import InfrastructureBase
from modelforge.infrastructure.cloud import CloudInfrastructure
from modelforge.infrastructure.colab import ColabInfrastructure
from modelforge.infrastructure.container import ContainerInfrastructure
from modelforge.utils.exceptions import InfrastructureError
from modelforge.utils.logging import StructuredLogger


class LocalInfrastructure(InfrastructureBase):
    """
    Infraestrutura local padrão.
    
    Implementação simples para execução local sem configurações especiais.
    """
    
    def __init__(self, logger: Optional[StructuredLogger] = None) -> None:
        """Inicializa infraestrutura local."""
        super().__init__()
        self._logger = logger or StructuredLogger("local_infrastructure")
    
    def setup(self, config: InfrastructureConfig) -> None:
        """Configura infraestrutura local."""
        self._config = config
        self._logger.info("Configurando ambiente local")
        
        self._resources = self.detect_resources()
        self.configure_environment()
        
        self._is_setup = True
        self._logger.info(
            "Ambiente local configurado",
            resources=self._resources.to_dict()
        )
    
    def detect_resources(self):
        """Detecta recursos locais."""
        from modelforge.infrastructure.base import ResourceInfo
        import torch
        import os
        
        resources = ResourceInfo()
        
        try:
            import psutil
            resources.cpu_count = psutil.cpu_count(logical=True) or 1
            resources.memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            resources.cpu_count = os.cpu_count() or 1
        
        if torch.cuda.is_available():
            resources.gpu_available = True
            resources.gpu_count = torch.cuda.device_count()
            resources.device_type = "cuda"
            
            for i in range(resources.gpu_count):
                props = torch.cuda.get_device_properties(i)
                resources.gpu_names.append(props.name)
                resources.gpu_memory_gb.append(
                    props.total_memory / (1024 ** 3)
                )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resources.gpu_available = True
            resources.gpu_count = 1
            resources.device_type = "mps"
            resources.gpu_names = ["Apple Silicon"]
        else:
            resources.device_type = "cpu"
        
        self._resources = resources
        return resources
    
    def configure_environment(self) -> None:
        """Configura ambiente local."""
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    def get_device(self):
        """Retorna dispositivo local."""
        if self._device is None:
            import torch
            
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        
        return self._device
    
    def get_infrastructure_type(self) -> str:
        """Retorna tipo de infraestrutura."""
        return "local"


class InfrastructureFactory:
    """
    Factory para criação de infraestruturas.
    
    Implementa o padrão Factory Method para criar instâncias
    de infraestruturas baseado no tipo especificado.
    
    Attributes:
        _infrastructures: Dicionário mapeando tipos para classes
    
    Example:
        >>> infra = InfrastructureFactory.create_infrastructure(
        ...     "colab",
        ...     config
        ... )
        >>> print(type(infra))
        <class 'ColabInfrastructure'>
    """
    
    # Registro de infraestruturas disponíveis
    _infrastructures: Dict[str, Type[InfrastructureBase]] = {
        "local": LocalInfrastructure,
        "colab": ColabInfrastructure,
        "container": ContainerInfrastructure,
        "cloud": CloudInfrastructure,
    }
    
    _logger: Optional[StructuredLogger] = None
    
    @classmethod
    def _get_logger(cls) -> StructuredLogger:
        """Retorna o logger da factory, criando se necessário."""
        if cls._logger is None:
            cls._logger = StructuredLogger("infrastructure_factory")
        return cls._logger
    
    @classmethod
    def create_infrastructure(
        cls,
        infra_type: str,
        config: Optional[InfrastructureConfig] = None,
        **kwargs
    ) -> InfrastructureBase:
        """
        Cria uma instância de infraestrutura.
        
        Args:
            infra_type: Tipo da infraestrutura
            config: Configuração de infraestrutura
            **kwargs: Argumentos adicionais
            
        Returns:
            InfrastructureBase: Instância da infraestrutura
            
        Raises:
            InfrastructureError: Se o tipo não for suportado
        """
        infra_type = infra_type.lower()
        
        if infra_type not in cls._infrastructures:
            available = ", ".join(cls._infrastructures.keys())
            raise InfrastructureError(
                f"Infraestrutura '{infra_type}' não suportada. "
                f"Disponíveis: {available}",
                infrastructure_type=infra_type
            )
        
        infra_class = cls._infrastructures[infra_type]
        
        cls._get_logger().info(
            "Criando infraestrutura",
            type=infra_type,
            class_name=infra_class.__name__
        )
        
        # Cria instância
        infrastructure = infra_class(**kwargs)
        
        # Configura se config fornecido
        if config is not None:
            infrastructure.setup(config)
        
        return infrastructure
    
    @classmethod
    def register_infrastructure(
        cls,
        name: str,
        infra_class: Type[InfrastructureBase]
    ) -> None:
        """
        Registra uma nova infraestrutura.
        
        Args:
            name: Nome da infraestrutura
            infra_class: Classe da infraestrutura
        """
        if not issubclass(infra_class, InfrastructureBase):
            raise ValueError(
                f"A classe {infra_class.__name__} deve herdar de InfrastructureBase"
            )
        
        cls._infrastructures[name.lower()] = infra_class
        cls._get_logger().info(
            "Infraestrutura registrada",
            name=name,
            class_name=infra_class.__name__
        )
    
    @classmethod
    def unregister_infrastructure(cls, name: str) -> None:
        """Remove uma infraestrutura do registro."""
        name = name.lower()
        if name in cls._infrastructures:
            del cls._infrastructures[name]
            cls._get_logger().info("Infraestrutura removida", name=name)
    
    @classmethod
    def list_infrastructures(cls) -> Dict[str, str]:
        """
        Lista infraestruturas disponíveis.
        
        Returns:
            Dict mapeando nomes para classes
        """
        return {
            name: cls.__name__
            for name, cls in cls._infrastructures.items()
        }
    
    @classmethod
    def auto_detect(cls) -> str:
        """
        Detecta automaticamente o tipo de infraestrutura.
        
        Returns:
            Nome do tipo de infraestrutura detectado
        """
        # Tenta detectar Colab
        try:
            import google.colab  # noqa
            return "colab"
        except ImportError:
            pass
        
        # Tenta detectar container
        from pathlib import Path
        if Path("/.dockerenv").exists():
            return "container"
        
        # Default para local
        return "local"
    
    @classmethod
    def create_auto(
        cls,
        config: Optional[InfrastructureConfig] = None,
        **kwargs
    ) -> InfrastructureBase:
        """
        Cria infraestrutura detectando automaticamente o ambiente.
        
        Args:
            config: Configuração de infraestrutura
            **kwargs: Argumentos adicionais
            
        Returns:
            Infraestrutura apropriada para o ambiente
        """
        infra_type = cls.auto_detect()
        cls._get_logger().info(f"Infraestrutura detectada automaticamente: {infra_type}")
        return cls.create_infrastructure(infra_type, config, **kwargs)

