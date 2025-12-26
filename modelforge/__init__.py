"""
ModelForge - Sistema de Fine-tuning de Modelos ML

Um sistema completo para fine-tuning de modelos de machine learning,
orientado por configuração YAML, com CLI intuitivo e deploy via Docker.
"""

__version__ = "0.1.0"
__author__ = "ModelForge Team"

from modelforge.config.schema import Config
from modelforge.config.loader import ConfigLoader
from modelforge.core.trainer import ModelTrainer
from modelforge.backends.factory import BackendFactory
from modelforge.infrastructure.factory import InfrastructureFactory

__all__ = [
    "Config",
    "ConfigLoader",
    "ModelTrainer",
    "BackendFactory",
    "InfrastructureFactory",
    "__version__",
]

