"""
Classe ColabInfrastructure para execução no Google Colab.

Este módulo implementa suporte específico para o ambiente Google Colab,
detectando e configurando automaticamente recursos disponíveis.
"""

import os
from typing import Any, Optional

from modelforge.config.schema import InfrastructureConfig
from modelforge.infrastructure.base import InfrastructureBase, ResourceInfo
from modelforge.utils.exceptions import InfrastructureError
from modelforge.utils.logging import StructuredLogger


class ColabInfrastructure(InfrastructureBase):
    """
    Infraestrutura para Google Colab.
    
    Fornece:
    - Detecção automática do ambiente Colab
    - Configuração de GPU/TPU
    - Integração com Google Drive (opcional)
    - Otimizações específicas para Colab
    
    Attributes:
        is_colab: Se está executando no Colab
        has_gpu: Se GPU está disponível
        logger: Logger estruturado
    
    Example:
        >>> infra = ColabInfrastructure()
        >>> if infra.is_colab_environment():
        ...     infra.setup(config)
        ...     device = infra.get_device()
    """
    
    def __init__(self, logger: Optional[StructuredLogger] = None) -> None:
        """
        Inicializa a infraestrutura Colab.
        
        Args:
            logger: Logger estruturado
        """
        super().__init__()
        self._logger = logger or StructuredLogger("colab_infrastructure")
        self._is_colab = self._detect_colab_environment()
    
    @property
    def is_colab(self) -> bool:
        """Retorna se está no Colab."""
        return self._is_colab
    
    def _detect_colab_environment(self) -> bool:
        """
        Detecta se está executando no Google Colab.
        
        Returns:
            True se está no Colab
        """
        try:
            import google.colab  # noqa
            return True
        except ImportError:
            return False
    
    def is_colab_environment(self) -> bool:
        """
        Verifica se está no ambiente Colab.
        
        Returns:
            True se está no Colab
        """
        return self._is_colab
    
    def setup(self, config: InfrastructureConfig) -> None:
        """
        Configura a infraestrutura Colab.
        
        Args:
            config: Configuração de infraestrutura
            
        Raises:
            InfrastructureError: Se não estiver no Colab
        """
        if not self._is_colab:
            raise InfrastructureError(
                "Não está executando no Google Colab",
                infrastructure_type="colab"
            )
        
        self._config = config
        self._logger.info("Configurando ambiente Colab")
        
        # Detecta recursos
        self._resources = self.detect_resources()
        
        # Configura ambiente
        self.configure_environment()
        
        self._is_setup = True
        self._logger.info(
            "Ambiente Colab configurado",
            resources=self._resources.to_dict()
        )
    
    def detect_resources(self) -> ResourceInfo:
        """
        Detecta recursos disponíveis no Colab.
        
        Returns:
            ResourceInfo com recursos do Colab
        """
        import torch
        import psutil
        
        resources = ResourceInfo()
        
        # CPU
        resources.cpu_count = psutil.cpu_count(logical=True) or 1
        resources.memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        # GPU
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
        else:
            resources.device_type = "cpu"
        
        self._resources = resources
        return resources
    
    def configure_environment(self) -> None:
        """Configura variáveis de ambiente para Colab."""
        # Otimizações para Colab
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Desabilita warnings desnecessários
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        
        self._logger.info("Variáveis de ambiente configuradas")
    
    def get_device(self) -> Any:
        """
        Retorna o dispositivo de computação.
        
        Returns:
            torch.device
        """
        if self._device is None:
            import torch
            
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
        
        return self._device
    
    def get_infrastructure_type(self) -> str:
        """Retorna tipo de infraestrutura."""
        return "colab"
    
    def mount_drive(self, mount_point: str = "/content/drive") -> str:
        """
        Monta o Google Drive.
        
        Args:
            mount_point: Ponto de montagem
            
        Returns:
            Caminho do Google Drive montado
        """
        if not self._is_colab:
            raise InfrastructureError(
                "Montagem de Drive só disponível no Colab",
                infrastructure_type="colab"
            )
        
        try:
            from google.colab import drive
            drive.mount(mount_point)
            self._logger.info("Google Drive montado", path=mount_point)
            return f"{mount_point}/MyDrive"
        except Exception as e:
            raise InfrastructureError(
                f"Falha ao montar Google Drive: {e}",
                infrastructure_type="colab",
                original_exception=e
            )
    
    def install_packages(self, packages: list) -> None:
        """
        Instala pacotes pip no Colab.
        
        Args:
            packages: Lista de pacotes a instalar
        """
        import subprocess
        
        for package in packages:
            self._logger.info(f"Instalando pacote: {package}")
            subprocess.run(
                ["pip", "install", "-q", package],
                check=True
            )
    
    def clear_output(self) -> None:
        """Limpa output do Colab."""
        if self._is_colab:
            try:
                from google.colab import output
                output.clear()
            except ImportError:
                pass

