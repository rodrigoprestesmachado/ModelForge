"""
Classe ModelEvaluator para avaliação de modelos.

Este módulo fornece funcionalidades para avaliar modelos
com diferentes métricas e gerar relatórios de avaliação.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from modelforge.utils.exceptions import TrainingError
from modelforge.utils.logging import StructuredLogger


@dataclass
class EvaluationResult:
    """
    Resultado de uma avaliação.
    
    Attributes:
        metrics: Dicionário com métricas calculadas
        split: Nome do split avaliado
        num_samples: Número de amostras avaliadas
        predictions: Predições do modelo (opcional)
        labels: Labels reais (opcional)
    """
    metrics: Dict[str, float] = field(default_factory=dict)
    split: str = "eval"
    num_samples: int = 0
    predictions: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            "metrics": self.metrics,
            "split": self.split,
            "num_samples": self.num_samples,
        }
    
    def __repr__(self) -> str:
        """Representação string."""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"EvaluationResult(split='{self.split}', {metrics_str})"


class ModelEvaluator:
    """
    Avaliador de modelos com suporte a múltiplas métricas.
    
    Esta classe fornece:
    - Avaliação com métricas padrão (accuracy, f1, precision, recall)
    - Suporte a métricas customizadas
    - Geração de relatórios de avaliação
    - Comparação entre avaliações
    
    Attributes:
        model: Modelo a avaliar
        tokenizer: Tokenizer do modelo
        metrics: Lista de métricas a calcular
        logger: Logger estruturado
    
    Example:
        >>> evaluator = ModelEvaluator(model, tokenizer, ["accuracy", "f1"])
        >>> result = evaluator.evaluate(test_dataset)
        >>> print(result.metrics)
        {'accuracy': 0.95, 'f1': 0.94}
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        metrics: Optional[List[str]] = None,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        """
        Inicializa o avaliador.
        
        Args:
            model: Modelo a avaliar
            tokenizer: Tokenizer do modelo
            metrics: Lista de métricas (default: ["accuracy"])
            logger: Logger estruturado
        """
        self._model = model
        self._tokenizer = tokenizer
        self._metrics = metrics or ["accuracy"]
        self._logger = logger or StructuredLogger("evaluator")
        self._custom_metrics: Dict[str, Callable] = {}
        self._evaluation_history: List[EvaluationResult] = []
    
    @property
    def model(self) -> Any:
        """Retorna o modelo."""
        return self._model
    
    @property
    def metrics(self) -> List[str]:
        """Retorna lista de métricas."""
        return self._metrics
    
    @property
    def evaluation_history(self) -> List[EvaluationResult]:
        """Retorna histórico de avaliações."""
        return self._evaluation_history
    
    def add_metric(self, name: str, metric_fn: Callable) -> None:
        """
        Adiciona uma métrica customizada.
        
        Args:
            name: Nome da métrica
            metric_fn: Função que recebe (predictions, labels) e retorna float
        """
        self._custom_metrics[name] = metric_fn
        self._logger.info(f"Métrica customizada adicionada: {name}")
    
    def evaluate(
        self,
        dataset: Any,
        split: str = "eval",
        batch_size: int = 32,
    ) -> EvaluationResult:
        """
        Avalia o modelo em um dataset.
        
        Args:
            dataset: Dataset de avaliação
            split: Nome do split
            batch_size: Tamanho do batch para inferência
            
        Returns:
            EvaluationResult com métricas calculadas
        """
        self._logger.info(f"Avaliando modelo no split '{split}'")
        
        try:
            # Obtém predições
            predictions, labels = self._get_predictions(dataset, batch_size)
            
            # Calcula métricas
            metrics = self.compute_metrics(predictions, labels)
            
            # Cria resultado
            result = EvaluationResult(
                metrics=metrics,
                split=split,
                num_samples=len(labels),
                predictions=predictions,
                labels=labels,
            )
            
            # Adiciona ao histórico
            self._evaluation_history.append(result)
            
            self._logger.info(
                "Avaliação concluída",
                split=split,
                metrics=metrics
            )
            
            return result
            
        except Exception as e:
            raise TrainingError(
                f"Erro durante avaliação: {e}",
                original_exception=e
            )
    
    def _get_predictions(
        self,
        dataset: Any,
        batch_size: int
    ) -> tuple:
        """
        Obtém predições do modelo para o dataset.
        
        Args:
            dataset: Dataset de avaliação
            batch_size: Tamanho do batch
            
        Returns:
            Tuple (predictions, labels)
        """
        import torch
        from torch.utils.data import DataLoader
        
        self._model.eval()
        device = next(self._model.parameters()).device
        
        # Cria dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move para dispositivo
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch.get("labels")
                
                # Forward pass
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                if labels is not None:
                    all_labels.extend(labels.numpy())
        
        return np.array(all_predictions), np.array(all_labels)
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcula métricas para predições e labels.
        
        Args:
            predictions: Array de predições
            labels: Array de labels reais
            
        Returns:
            Dict com métricas calculadas
        """
        results = {}
        
        # Métricas padrão usando evaluate library
        for metric_name in self._metrics:
            if metric_name in self._custom_metrics:
                # Usa métrica customizada
                results[metric_name] = self._custom_metrics[metric_name](
                    predictions, labels
                )
            else:
                # Usa evaluate library
                try:
                    import evaluate
                    metric = evaluate.load(metric_name)
                    result = metric.compute(
                        predictions=predictions,
                        references=labels
                    )
                    if isinstance(result, dict):
                        # Algumas métricas retornam dict com múltiplos valores
                        for key, value in result.items():
                            if isinstance(value, (int, float)):
                                results[key] = float(value)
                    else:
                        results[metric_name] = float(result)
                except Exception as e:
                    self._logger.warning(
                        f"Falha ao calcular métrica {metric_name}: {e}"
                    )
        
        return results
    
    def generate_report(
        self,
        result: Optional[EvaluationResult] = None,
        include_confusion_matrix: bool = False
    ) -> str:
        """
        Gera relatório de avaliação em formato texto.
        
        Args:
            result: Resultado de avaliação (usa último se não fornecido)
            include_confusion_matrix: Se deve incluir matriz de confusão
            
        Returns:
            String com relatório formatado
        """
        if result is None:
            if not self._evaluation_history:
                return "Nenhuma avaliação realizada."
            result = self._evaluation_history[-1]
        
        lines = [
            "=" * 50,
            "RELATÓRIO DE AVALIAÇÃO",
            "=" * 50,
            f"Split: {result.split}",
            f"Amostras avaliadas: {result.num_samples}",
            "",
            "MÉTRICAS:",
            "-" * 30,
        ]
        
        for metric, value in result.metrics.items():
            lines.append(f"  {metric}: {value:.4f}")
        
        if include_confusion_matrix and result.predictions is not None:
            lines.extend([
                "",
                "MATRIZ DE CONFUSÃO:",
                "-" * 30,
            ])
            try:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(result.labels, result.predictions)
                lines.append(str(cm))
            except ImportError:
                lines.append("  (sklearn não disponível)")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def compare_evaluations(
        self,
        result1: EvaluationResult,
        result2: EvaluationResult
    ) -> Dict[str, float]:
        """
        Compara duas avaliações.
        
        Args:
            result1: Primeira avaliação
            result2: Segunda avaliação
            
        Returns:
            Dict com diferenças entre métricas
        """
        differences = {}
        
        all_metrics = set(result1.metrics.keys()) | set(result2.metrics.keys())
        
        for metric in all_metrics:
            val1 = result1.metrics.get(metric, 0.0)
            val2 = result2.metrics.get(metric, 0.0)
            differences[f"{metric}_diff"] = val2 - val1
            differences[f"{metric}_pct_change"] = (
                ((val2 - val1) / val1 * 100) if val1 != 0 else 0.0
            )
        
        return differences
    
    def get_best_evaluation(
        self,
        metric: str = "accuracy",
        higher_is_better: bool = True
    ) -> Optional[EvaluationResult]:
        """
        Retorna a melhor avaliação baseado em uma métrica.
        
        Args:
            metric: Nome da métrica
            higher_is_better: Se maior valor é melhor
            
        Returns:
            Melhor EvaluationResult ou None
        """
        if not self._evaluation_history:
            return None
        
        valid_results = [
            r for r in self._evaluation_history
            if metric in r.metrics
        ]
        
        if not valid_results:
            return None
        
        if higher_is_better:
            return max(valid_results, key=lambda r: r.metrics[metric])
        else:
            return min(valid_results, key=lambda r: r.metrics[metric])
    
    def __repr__(self) -> str:
        """Representação string."""
        return (
            f"ModelEvaluator("
            f"metrics={self._metrics}, "
            f"evaluations={len(self._evaluation_history)})"
        )

