"""
Classe abstrata InfrastructureBase para ambientes de execução.

Este módulo define a interface base para diferentes infraestruturas,
permitindo execução em diversos ambientes de forma transparente.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from modelforge.config.schema import InfrastructureConfig


@dataclass
class ResourceInfo:
    """
    Informações sobre recursos computacionais disponíveis.
    
    Attributes:
        gpu_available: Se GPU está disponível
        gpu_count: Número de GPUs disponíveis
        gpu_names: Nomes das GPUs
        gpu_memory_gb: Memória de cada GPU em GB
        cpu_count: Número de CPUs
        memory_gb: Memória RAM total em GB
        device_type: Tipo de dispositivo (cuda, mps, cpu)
    """
    gpu_available: bool = False
    gpu_count: int = 0
    gpu_names: list = field(default_factory=list)
    gpu_memory_gb: list = field(default_factory=list)
    cpu_count: int = 1
    memory_gb: float = 0.0
    device_type: str = "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            "gpu_available": self.gpu_available,
            "gpu_count": self.gpu_count,
            "gpu_names": self.gpu_names,
            "gpu_memory_gb": self.gpu_memory_gb,
            "cpu_count": self.cpu_count,
            "memory_gb": self.memory_gb,
            "device_type": self.device_type,
        }
    
    def __repr__(self) -> str:
        """Representação string."""
        if self.gpu_available:
            return (
                f"ResourceInfo(device='{self.device_type}', "
                f"gpus={self.gpu_count}, "
                f"gpu_memory={self.gpu_memory_gb}GB)"
            )
        return f"ResourceInfo(device='cpu', cpus={self.cpu_count})"


class InfrastructureBase(ABC):
    """
    Classe abstrata base para infraestruturas de execução.
    
    Define a interface que todas as infraestruturas devem implementar,
    permitindo execução transparente em diferentes ambientes.
    
    Attributes:
        config: Configuração de infraestrutura
        resources: Informações sobre recursos disponíveis
    
    Example:
        >>> class MyInfra(InfrastructureBase):
        ...     def setup(self, config):
        ...         # Configuração específica
        ...         pass
    """
    
    def __init__(self) -> None:
        """Inicializa a infraestrutura."""
        self._config: Optional[InfrastructureConfig] = None
        self._resources: Optional[ResourceInfo] = None
        self._device: Optional[Any] = None
        self._is_setup = False
    
    @property
    def config(self) -> Optional[InfrastructureConfig]:
        """Retorna a configuração."""
        return self._config
    
    @property
    def resources(self) -> Optional[ResourceInfo]:
        """Retorna informações de recursos."""
        return self._resources
    
    @property
    def is_setup(self) -> bool:
        """Retorna se está configurado."""
        return self._is_setup
    
    @abstractmethod
    def setup(self, config: InfrastructureConfig) -> None:
        """
        Configura a infraestrutura.
        
        Args:
            config: Configuração de infraestrutura
            
        Raises:
            InfrastructureError: Se a configuração falhar
        """
        pass
    
    @abstractmethod
    def detect_resources(self) -> ResourceInfo:
        """
        Detecta recursos computacionais disponíveis.
        
        Returns:
            ResourceInfo com informações dos recursos
        """
        pass
    
    @abstractmethod
    def configure_environment(self) -> None:
        """
        Configura variáveis de ambiente necessárias.
        
        Raises:
            InfrastructureError: Se a configuração falhar
        """
        pass
    
    @abstractmethod
    def get_device(self) -> Any:
        """
        Retorna o dispositivo de computação.
        
        Returns:
            torch.device ou equivalente
        """
        pass
    
    @abstractmethod
    def get_infrastructure_type(self) -> str:
        """
        Retorna o tipo de infraestrutura.
        
        Returns:
            String identificando o tipo (local, colab, cloud, container)
        """
        pass
    
    def validate(self) -> bool:
        """
        Valida se a infraestrutura está pronta para uso.
        
        Returns:
            True se a infraestrutura está pronta
        """
        return self._is_setup and self._resources is not None
    
    def cleanup(self) -> None:
        """Limpa recursos alocados pela infraestrutura."""
        self._is_setup = False
    
    def __repr__(self) -> str:
        """Representação string."""
        return (
            f"{self.__class__.__name__}("
            f"type='{self.get_infrastructure_type()}', "
            f"is_setup={self._is_setup})"
        )

