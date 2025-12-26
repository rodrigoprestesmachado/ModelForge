"""
Módulo core do ModelForge.

Contém as classes principais para treinamento, avaliação e gerenciamento de checkpoints.
"""

from modelforge.core.trainer import ModelTrainer, TrainingResult
from modelforge.core.evaluator import ModelEvaluator, EvaluationResult
from modelforge.core.checkpoint import CheckpointManager

__all__ = [
    "ModelTrainer",
    "TrainingResult",
    "ModelEvaluator",
    "EvaluationResult",
    "CheckpointManager",
]

