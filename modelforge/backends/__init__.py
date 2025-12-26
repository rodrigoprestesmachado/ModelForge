"""
Módulo de backends do ModelForge.

Contém as classes para diferentes backends de treinamento (Hugging Face, etc.).
"""

from modelforge.backends.base import BackendBase
from modelforge.backends.huggingface import HuggingFaceBackend
from modelforge.backends.factory import BackendFactory

__all__ = [
    "BackendBase",
    "HuggingFaceBackend",
    "BackendFactory",
]

