"""
Módulo core do ModelForge.

Contém as classes principais para treinamento, avaliação, inferência e gerenciamento de checkpoints.
"""

from modelforge.core.trainer import ModelTrainer, TrainingResult
from modelforge.core.evaluator import ModelEvaluator, EvaluationResult
from modelforge.core.checkpoint import CheckpointManager
from modelforge.core.inference import ModelInference, InferenceError

__all__ = [
    "ModelTrainer",
    "TrainingResult",
    "ModelEvaluator",
    "EvaluationResult",
    "CheckpointManager",
    "ModelInference",
    "InferenceError",
]

