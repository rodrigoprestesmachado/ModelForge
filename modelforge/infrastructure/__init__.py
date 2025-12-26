"""
Módulo de infraestrutura do ModelForge.

Contém as classes para diferentes ambientes de execução (Colab, Cloud, Container).
"""

from modelforge.infrastructure.base import InfrastructureBase, ResourceInfo
from modelforge.infrastructure.colab import ColabInfrastructure
from modelforge.infrastructure.cloud import CloudInfrastructure
from modelforge.infrastructure.container import ContainerInfrastructure
from modelforge.infrastructure.factory import InfrastructureFactory

__all__ = [
    "InfrastructureBase",
    "ResourceInfo",
    "ColabInfrastructure",
    "CloudInfrastructure",
    "ContainerInfrastructure",
    "InfrastructureFactory",
]

