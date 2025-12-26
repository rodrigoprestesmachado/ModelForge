"""
Classe StructuredLogger para logging estruturado.

Este módulo fornece uma classe de logging que suporta
logs estruturados em formato JSON para produção.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


class StructuredLogger:
    """
    Logger estruturado com suporte a formato JSON.
    
    Esta classe fornece logging com:
    - Formato JSON para facilitar parsing em produção
    - Formato texto para desenvolvimento
    - Contexto adicional em cada log
    - Suporte a diferentes níveis de log
    - Saída para arquivo e console
    
    Attributes:
        name: Nome do logger
        level: Nível de log
        json_format: Se deve usar formato JSON
        _logger: Logger interno do Python
    
    Example:
        >>> logger = StructuredLogger("training", level="INFO")
        >>> logger.info("Treinamento iniciado", epoch=1, batch_size=16)
        {"timestamp": "...", "level": "INFO", "message": "Treinamento iniciado", "epoch": 1, "batch_size": 16}
    """
    
    # Mapeamento de níveis de log
    LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        json_format: bool = True,
        log_file: Optional[Union[str, Path]] = None,
        include_timestamp: bool = True,
    ) -> None:
        """
        Inicializa o logger estruturado.
        
        Args:
            name: Nome do logger
            level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            json_format: Se deve usar formato JSON
            log_file: Arquivo para salvar logs (opcional)
            include_timestamp: Incluir timestamp nos logs
        """
        self.name = name
        self.level = level.upper()
        self.json_format = json_format
        self.include_timestamp = include_timestamp
        self._context: Dict[str, Any] = {}
        
        # Configura o logger interno
        self._logger = logging.getLogger(name)
        self._logger.setLevel(self.LEVELS.get(self.level, logging.INFO))
        self._logger.handlers.clear()  # Remove handlers existentes
        
        # Adiciona handler de console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.LEVELS.get(self.level, logging.INFO))
        self._logger.addHandler(console_handler)
        
        # Adiciona handler de arquivo se especificado
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(self.LEVELS.get(self.level, logging.INFO))
            self._logger.addHandler(file_handler)
    
    def set_context(self, **kwargs: Any) -> None:
        """
        Define contexto adicional para todos os logs.
        
        Args:
            **kwargs: Pares chave-valor para adicionar ao contexto
        """
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Limpa o contexto adicional."""
        self._context.clear()
    
    def _format_log(
        self,
        level: str,
        message: str,
        **kwargs: Any
    ) -> str:
        """
        Formata uma mensagem de log.
        
        Args:
            level: Nível do log
            message: Mensagem principal
            **kwargs: Dados adicionais
            
        Returns:
            String formatada (JSON ou texto)
        """
        log_data: Dict[str, Any] = {}
        
        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        log_data["level"] = level
        log_data["logger"] = self.name
        log_data["message"] = message
        
        # Adiciona contexto
        log_data.update(self._context)
        
        # Adiciona dados extras
        log_data.update(kwargs)
        
        if self.json_format:
            return json.dumps(log_data, default=str, ensure_ascii=False)
        else:
            # Formato texto simples
            extras = ""
            if kwargs:
                extras = " | " + " ".join(f"{k}={v}" for k, v in kwargs.items())
            timestamp = log_data.get("timestamp", "")
            return f"[{timestamp}] [{level}] {self.name}: {message}{extras}"
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """
        Log de nível DEBUG.
        
        Args:
            message: Mensagem de log
            **kwargs: Dados adicionais
        """
        formatted = self._format_log("DEBUG", message, **kwargs)
        self._logger.debug(formatted)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """
        Log de nível INFO.
        
        Args:
            message: Mensagem de log
            **kwargs: Dados adicionais
        """
        formatted = self._format_log("INFO", message, **kwargs)
        self._logger.info(formatted)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """
        Log de nível WARNING.
        
        Args:
            message: Mensagem de log
            **kwargs: Dados adicionais
        """
        formatted = self._format_log("WARNING", message, **kwargs)
        self._logger.warning(formatted)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """
        Log de nível ERROR.
        
        Args:
            message: Mensagem de log
            **kwargs: Dados adicionais
        """
        formatted = self._format_log("ERROR", message, **kwargs)
        self._logger.error(formatted)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """
        Log de nível CRITICAL.
        
        Args:
            message: Mensagem de log
            **kwargs: Dados adicionais
        """
        formatted = self._format_log("CRITICAL", message, **kwargs)
        self._logger.critical(formatted)
    
    def exception(self, message: str, exc: Exception, **kwargs: Any) -> None:
        """
        Log de exceção com stack trace.
        
        Args:
            message: Mensagem de log
            exc: Exceção a logar
            **kwargs: Dados adicionais
        """
        kwargs["exception_type"] = type(exc).__name__
        kwargs["exception_message"] = str(exc)
        formatted = self._format_log("ERROR", message, **kwargs)
        self._logger.error(formatted, exc_info=True)
    
    def log_training_step(
        self,
        epoch: int,
        step: int,
        loss: float,
        learning_rate: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """
        Log específico para passos de treinamento.
        
        Args:
            epoch: Época atual
            step: Passo atual
            loss: Valor do loss
            learning_rate: Learning rate atual
            **kwargs: Métricas adicionais
        """
        data = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
        }
        if learning_rate is not None:
            data["learning_rate"] = learning_rate
        data.update(kwargs)
        
        self.info("Training step", **data)
    
    def log_evaluation(
        self,
        epoch: int,
        metrics: Dict[str, float],
        **kwargs: Any
    ) -> None:
        """
        Log específico para avaliação.
        
        Args:
            epoch: Época da avaliação
            metrics: Métricas de avaliação
            **kwargs: Dados adicionais
        """
        data = {
            "epoch": epoch,
            "metrics": metrics,
        }
        data.update(kwargs)
        
        self.info("Evaluation completed", **data)
    
    def log_checkpoint(
        self,
        checkpoint_path: str,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any
    ) -> None:
        """
        Log específico para salvamento de checkpoint.
        
        Args:
            checkpoint_path: Caminho do checkpoint
            epoch: Época do checkpoint
            metrics: Métricas no momento do salvamento
            **kwargs: Dados adicionais
        """
        data = {
            "checkpoint_path": checkpoint_path,
            "epoch": epoch,
        }
        if metrics:
            data["metrics"] = metrics
        data.update(kwargs)
        
        self.info("Checkpoint saved", **data)
    
    def __repr__(self) -> str:
        """Representação string do logger."""
        return (
            f"StructuredLogger(name='{self.name}', "
            f"level='{self.level}', "
            f"json_format={self.json_format})"
        )


def get_logger(
    name: str = "modelforge",
    level: str = "INFO",
    json_format: bool = True,
) -> StructuredLogger:
    """
    Factory function para criar loggers.
    
    Args:
        name: Nome do logger
        level: Nível de log
        json_format: Se deve usar formato JSON
        
    Returns:
        StructuredLogger configurado
    """
    return StructuredLogger(name=name, level=level, json_format=json_format)

