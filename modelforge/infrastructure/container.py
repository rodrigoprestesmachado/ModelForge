"""
Classe ContainerInfrastructure para execução em containers Docker.

Este módulo implementa suporte para execução em ambientes containerizados,
com detecção automática de recursos e validação do ambiente.
"""

import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from modelforge.config.schema import InfrastructureConfig
from modelforge.infrastructure.base import InfrastructureBase, ResourceInfo
from modelforge.utils.exceptions import InfrastructureError
from modelforge.utils.logging import StructuredLogger


class ContainerInfrastructure(InfrastructureBase):
    """
    Infraestrutura para execução em containers Docker.
    
    Fornece:
    - Detecção de ambiente containerizado
    - Validação de Docker
    - Configuração de recursos em containers
    - Suporte a nvidia-docker para GPUs
    
    Attributes:
        is_container: Se está executando em container
        docker_available: Se Docker está disponível
        logger: Logger estruturado
    
    Example:
        >>> infra = ContainerInfrastructure()
        >>> if infra.is_container_environment():
        ...     infra.setup(config)
    """
    
    def __init__(self, logger: Optional[StructuredLogger] = None) -> None:
        """
        Inicializa a infraestrutura de container.
        
        Args:
            logger: Logger estruturado
        """
        super().__init__()
        self._logger = logger or StructuredLogger("container_infrastructure")
        self._is_container = self._detect_container_environment()
        self._docker_available = self._check_docker()
    
    @property
    def is_container(self) -> bool:
        """Retorna se está em container."""
        return self._is_container
    
    @property
    def docker_available(self) -> bool:
        """Retorna se Docker está disponível."""
        return self._docker_available
    
    def _detect_container_environment(self) -> bool:
        """
        Detecta se está executando em um container.
        
        Returns:
            True se está em container
        """
        # Verifica /.dockerenv
        if Path("/.dockerenv").exists():
            return True
        
        # Verifica cgroup
        try:
            with open("/proc/1/cgroup", "r") as f:
                content = f.read()
                if "docker" in content or "kubepods" in content:
                    return True
        except FileNotFoundError:
            pass
        
        # Verifica variável de ambiente
        if os.environ.get("DOCKER_CONTAINER"):
            return True
        
        return False
    
    def _check_docker(self) -> bool:
        """
        Verifica se Docker está disponível.
        
        Returns:
            True se Docker está instalado e acessível
        """
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
    
    def is_container_environment(self) -> bool:
        """
        Verifica se está em ambiente containerizado.
        
        Returns:
            True se está em container
        """
        return self._is_container
    
    def validate_docker(self) -> bool:
        """
        Valida instalação do Docker.
        
        Returns:
            True se Docker está funcionando corretamente
        """
        if not self._docker_available:
            return False
        
        try:
            # Tenta listar containers
            result = subprocess.run(
                ["docker", "ps"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def setup(self, config: InfrastructureConfig) -> None:
        """
        Configura a infraestrutura de container.
        
        Args:
            config: Configuração de infraestrutura
        """
        self._config = config
        self._logger.info("Configurando ambiente de container")
        
        # Detecta recursos
        self._resources = self.detect_resources()
        
        # Configura ambiente
        self.configure_environment()
        
        self._is_setup = True
        self._logger.info(
            "Ambiente de container configurado",
            is_container=self._is_container,
            resources=self._resources.to_dict()
        )
    
    def detect_resources(self) -> ResourceInfo:
        """
        Detecta recursos disponíveis no container.
        
        Returns:
            ResourceInfo com recursos detectados
        """
        import torch
        
        resources = ResourceInfo()
        
        # CPU - considera limites de cgroup se em container
        try:
            import psutil
            resources.cpu_count = psutil.cpu_count(logical=True) or 1
            resources.memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            resources.cpu_count = os.cpu_count() or 1
        
        # Verifica limites de container
        if self._is_container:
            resources.cpu_count = self._get_container_cpu_limit(resources.cpu_count)
            resources.memory_gb = self._get_container_memory_limit(resources.memory_gb)
        
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
    
    def _get_container_cpu_limit(self, default: int) -> int:
        """Obtém limite de CPU do container."""
        try:
            # cgroup v2
            cpu_max_path = Path("/sys/fs/cgroup/cpu.max")
            if cpu_max_path.exists():
                content = cpu_max_path.read_text().strip()
                if content != "max":
                    quota, period = content.split()
                    return max(1, int(int(quota) / int(period)))
            
            # cgroup v1
            quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
            period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
            
            if quota_path.exists() and period_path.exists():
                quota = int(quota_path.read_text().strip())
                period = int(period_path.read_text().strip())
                if quota > 0:
                    return max(1, int(quota / period))
        except Exception:
            pass
        
        return default
    
    def _get_container_memory_limit(self, default: float) -> float:
        """Obtém limite de memória do container."""
        try:
            # cgroup v2
            memory_max_path = Path("/sys/fs/cgroup/memory.max")
            if memory_max_path.exists():
                content = memory_max_path.read_text().strip()
                if content != "max":
                    return int(content) / (1024 ** 3)
            
            # cgroup v1
            memory_limit_path = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
            if memory_limit_path.exists():
                limit = int(memory_limit_path.read_text().strip())
                # Valor muito alto indica sem limite
                if limit < 10 ** 18:
                    return limit / (1024 ** 3)
        except Exception:
            pass
        
        return default
    
    def configure_environment(self) -> None:
        """Configura variáveis de ambiente para container."""
        # Configurações para otimização em container
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = str(self._resources.cpu_count if self._resources else 1)
        
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
        return "container"
    
    def run_container(
        self,
        image: str,
        command: str,
        volumes: Optional[dict] = None,
        gpu: bool = False,
        env_vars: Optional[dict] = None,
    ) -> str:
        """
        Executa um comando em um novo container.
        
        Args:
            image: Imagem Docker
            command: Comando a executar
            volumes: Mapeamento de volumes {host: container}
            gpu: Se deve usar GPU (nvidia-docker)
            env_vars: Variáveis de ambiente
            
        Returns:
            Output do container
        """
        if not self._docker_available:
            raise InfrastructureError(
                "Docker não está disponível",
                infrastructure_type="container"
            )
        
        docker_cmd = ["docker", "run", "--rm"]
        
        # GPU support
        if gpu:
            docker_cmd.extend(["--gpus", "all"])
        
        # Volumes
        if volumes:
            for host_path, container_path in volumes.items():
                docker_cmd.extend(["-v", f"{host_path}:{container_path}"])
        
        # Environment variables
        if env_vars:
            for key, value in env_vars.items():
                docker_cmd.extend(["-e", f"{key}={value}"])
        
        docker_cmd.extend([image, command])
        
        self._logger.info(
            "Executando container",
            image=image,
            command=command
        )
        
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise InfrastructureError(
                f"Falha ao executar container: {e.stderr}",
                infrastructure_type="container",
                original_exception=e
            )

