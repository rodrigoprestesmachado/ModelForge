"""
Classe CloudInfrastructure para execução em provedores cloud.

Este módulo implementa suporte genérico para provedores cloud,
com extensibilidade para AWS, GCP e Azure.
"""

import os
from typing import Any, Dict, Optional

from modelforge.config.schema import InfrastructureConfig
from modelforge.infrastructure.base import InfrastructureBase, ResourceInfo
from modelforge.utils.exceptions import InfrastructureError
from modelforge.utils.logging import StructuredLogger


class CloudInfrastructure(InfrastructureBase):
    """
    Infraestrutura genérica para provedores cloud.
    
    Fornece suporte base para:
    - AWS (Amazon Web Services)
    - GCP (Google Cloud Platform)
    - Azure (Microsoft Azure)
    
    Pode ser estendida para outros provedores.
    
    Attributes:
        provider: Nome do provedor cloud
        credentials: Credenciais do provedor
        logger: Logger estruturado
    
    Example:
        >>> infra = CloudInfrastructure(provider="aws")
        >>> infra.setup(config)
        >>> device = infra.get_device()
    """
    
    SUPPORTED_PROVIDERS = ["aws", "gcp", "azure", "generic"]
    
    def __init__(
        self,
        provider: str = "generic",
        credentials: Optional[Dict[str, str]] = None,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        """
        Inicializa a infraestrutura cloud.
        
        Args:
            provider: Provedor cloud (aws, gcp, azure, generic)
            credentials: Credenciais do provedor
            logger: Logger estruturado
        """
        super().__init__()
        
        if provider.lower() not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Provedor '{provider}' não suportado. "
                f"Use: {self.SUPPORTED_PROVIDERS}"
            )
        
        self._provider = provider.lower()
        self._credentials = credentials or {}
        self._logger = logger or StructuredLogger("cloud_infrastructure")
    
    @property
    def provider(self) -> str:
        """Retorna o provedor cloud."""
        return self._provider
    
    def setup(self, config: InfrastructureConfig) -> None:
        """
        Configura a infraestrutura cloud.
        
        Args:
            config: Configuração de infraestrutura
        """
        self._config = config
        self._logger.info(
            "Configurando ambiente cloud",
            provider=self._provider
        )
        
        # Configura credenciais específicas do provedor
        self._setup_provider_credentials()
        
        # Detecta recursos
        self._resources = self.detect_resources()
        
        # Configura ambiente
        self.configure_environment()
        
        self._is_setup = True
        self._logger.info(
            "Ambiente cloud configurado",
            provider=self._provider,
            resources=self._resources.to_dict()
        )
    
    def _setup_provider_credentials(self) -> None:
        """Configura credenciais específicas do provedor."""
        if self._provider == "aws":
            self._setup_aws_credentials()
        elif self._provider == "gcp":
            self._setup_gcp_credentials()
        elif self._provider == "azure":
            self._setup_azure_credentials()
    
    def _setup_aws_credentials(self) -> None:
        """Configura credenciais AWS."""
        if "AWS_ACCESS_KEY_ID" in self._credentials:
            os.environ["AWS_ACCESS_KEY_ID"] = self._credentials["AWS_ACCESS_KEY_ID"]
        if "AWS_SECRET_ACCESS_KEY" in self._credentials:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self._credentials["AWS_SECRET_ACCESS_KEY"]
        if "AWS_REGION" in self._credentials:
            os.environ["AWS_DEFAULT_REGION"] = self._credentials["AWS_REGION"]
    
    def _setup_gcp_credentials(self) -> None:
        """Configura credenciais GCP."""
        if "GOOGLE_APPLICATION_CREDENTIALS" in self._credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                self._credentials["GOOGLE_APPLICATION_CREDENTIALS"]
            )
        if "GCP_PROJECT_ID" in self._credentials:
            os.environ["GOOGLE_CLOUD_PROJECT"] = self._credentials["GCP_PROJECT_ID"]
    
    def _setup_azure_credentials(self) -> None:
        """Configura credenciais Azure."""
        if "AZURE_SUBSCRIPTION_ID" in self._credentials:
            os.environ["AZURE_SUBSCRIPTION_ID"] = (
                self._credentials["AZURE_SUBSCRIPTION_ID"]
            )
    
    def detect_resources(self) -> ResourceInfo:
        """
        Detecta recursos disponíveis na instância cloud.
        
        Returns:
            ResourceInfo com recursos detectados
        """
        import torch
        
        resources = ResourceInfo()
        
        try:
            import psutil
            resources.cpu_count = psutil.cpu_count(logical=True) or 1
            resources.memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            resources.cpu_count = os.cpu_count() or 1
        
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
        """Configura variáveis de ambiente para cloud."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Configurações específicas por provedor
        if self._provider == "aws":
            # SageMaker specific
            os.environ.setdefault("SM_MODEL_DIR", "/opt/ml/model")
        elif self._provider == "gcp":
            # Vertex AI specific
            os.environ.setdefault("AIP_MODEL_DIR", "/gcs/model")
        
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
        return f"cloud-{self._provider}"
    
    def get_instance_metadata(self) -> Dict[str, str]:
        """
        Obtém metadados da instância cloud.
        
        Returns:
            Dict com metadados da instância
        """
        metadata = {"provider": self._provider}
        
        if self._provider == "aws":
            metadata.update(self._get_aws_metadata())
        elif self._provider == "gcp":
            metadata.update(self._get_gcp_metadata())
        elif self._provider == "azure":
            metadata.update(self._get_azure_metadata())
        
        return metadata
    
    def _get_aws_metadata(self) -> Dict[str, str]:
        """Obtém metadados AWS."""
        import requests
        
        metadata = {}
        try:
            base_url = "http://169.254.169.254/latest/meta-data/"
            timeout = 1
            
            metadata["instance_id"] = requests.get(
                f"{base_url}instance-id", timeout=timeout
            ).text
            metadata["instance_type"] = requests.get(
                f"{base_url}instance-type", timeout=timeout
            ).text
            metadata["region"] = requests.get(
                f"{base_url}placement/region", timeout=timeout
            ).text
        except Exception:
            pass
        
        return metadata
    
    def _get_gcp_metadata(self) -> Dict[str, str]:
        """Obtém metadados GCP."""
        import requests
        
        metadata = {}
        try:
            base_url = "http://metadata.google.internal/computeMetadata/v1/"
            headers = {"Metadata-Flavor": "Google"}
            timeout = 1
            
            metadata["instance_id"] = requests.get(
                f"{base_url}instance/id",
                headers=headers,
                timeout=timeout
            ).text
            metadata["zone"] = requests.get(
                f"{base_url}instance/zone",
                headers=headers,
                timeout=timeout
            ).text.split("/")[-1]
            metadata["machine_type"] = requests.get(
                f"{base_url}instance/machine-type",
                headers=headers,
                timeout=timeout
            ).text.split("/")[-1]
        except Exception:
            pass
        
        return metadata
    
    def _get_azure_metadata(self) -> Dict[str, str]:
        """Obtém metadados Azure."""
        import requests
        
        metadata = {}
        try:
            url = "http://169.254.169.254/metadata/instance"
            headers = {"Metadata": "true"}
            params = {"api-version": "2021-02-01"}
            timeout = 1
            
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=timeout
            )
            data = response.json()
            
            metadata["vm_id"] = data.get("compute", {}).get("vmId", "")
            metadata["vm_size"] = data.get("compute", {}).get("vmSize", "")
            metadata["location"] = data.get("compute", {}).get("location", "")
        except Exception:
            pass
        
        return metadata

