"""
Classe CheckpointManager para gerenciamento de checkpoints.

Este módulo fornece funcionalidades para salvar, carregar e
gerenciar checkpoints de modelos durante o treinamento.
"""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from modelforge.config.schema import CheckpointConfig
from modelforge.utils.exceptions import CheckpointError
from modelforge.utils.logging import StructuredLogger


@dataclass
class CheckpointInfo:
    """
    Informações sobre um checkpoint.
    
    Attributes:
        path: Caminho do checkpoint
        epoch: Época do checkpoint
        step: Passo global
        metrics: Métricas no momento do salvamento
        timestamp: Data/hora do salvamento
    """
    path: str
    epoch: int
    step: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            "path": self.path,
            "epoch": self.epoch,
            "step": self.step,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointInfo":
        """Cria a partir de dicionário."""
        return cls(
            path=data["path"],
            epoch=data["epoch"],
            step=data.get("step", 0),
            metrics=data.get("metrics", {}),
            timestamp=data.get("timestamp", ""),
        )


class CheckpointManager:
    """
    Gerenciador de checkpoints para modelos.
    
    Esta classe fornece:
    - Salvamento de checkpoints com metadados
    - Carregamento de checkpoints
    - Limpeza automática de checkpoints antigos
    - Upload para Hugging Face Hub
    - Rastreamento de histórico de checkpoints
    
    Attributes:
        config: Configuração de checkpoints
        save_dir: Diretório para salvar checkpoints
        max_to_keep: Número máximo de checkpoints a manter
        logger: Logger estruturado
    
    Example:
        >>> manager = CheckpointManager(checkpoint_config)
        >>> path = manager.save(model, epoch=1, metrics={"loss": 0.5})
        >>> model = manager.load(path)
    """
    
    def __init__(
        self,
        config: CheckpointConfig,
        hub_client: Optional[Any] = None,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        """
        Inicializa o gerenciador de checkpoints.
        
        Args:
            config: Configuração de checkpoints
            hub_client: Cliente do Hugging Face Hub (opcional)
            logger: Logger estruturado
        """
        self._config = config
        self._hub_client = hub_client
        self._logger = logger or StructuredLogger("checkpoint_manager")
        
        # Configura diretório
        self._save_dir = Path(config.save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        
        # Histórico de checkpoints
        self._checkpoints: List[CheckpointInfo] = []
        self._load_checkpoint_history()
    
    @property
    def save_dir(self) -> Path:
        """Retorna diretório de salvamento."""
        return self._save_dir
    
    @property
    def checkpoints(self) -> List[CheckpointInfo]:
        """Retorna lista de checkpoints."""
        return self._checkpoints.copy()
    
    @property
    def latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """Retorna o checkpoint mais recente."""
        if not self._checkpoints:
            return None
        return self._checkpoints[-1]
    
    def save(
        self,
        model: Any,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        step: int = 0,
        tokenizer: Optional[Any] = None,
    ) -> str:
        """
        Salva um checkpoint do modelo.
        
        Args:
            model: Modelo a salvar
            epoch: Época atual
            metrics: Métricas no momento do salvamento
            step: Passo global
            tokenizer: Tokenizer a salvar (opcional)
            
        Returns:
            Caminho do checkpoint salvo
        """
        metrics = metrics or {}
        
        # Gera nome do checkpoint
        checkpoint_name = f"checkpoint-epoch-{epoch}"
        if step > 0:
            checkpoint_name += f"-step-{step}"
        
        checkpoint_path = self._save_dir / checkpoint_name
        
        self._logger.info(
            "Salvando checkpoint",
            path=str(checkpoint_path),
            epoch=epoch,
            step=step
        )
        
        try:
            # Cria diretório
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            # Salva modelo
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(checkpoint_path)
            else:
                # Fallback para PyTorch
                import torch
                torch.save(model.state_dict(), checkpoint_path / "pytorch_model.bin")
            
            # Salva tokenizer se fornecido
            if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(checkpoint_path)
            
            # Salva metadados
            info = CheckpointInfo(
                path=str(checkpoint_path),
                epoch=epoch,
                step=step,
                metrics=metrics,
            )
            self._save_metadata(checkpoint_path, info)
            
            # Adiciona ao histórico
            self._checkpoints.append(info)
            self._save_checkpoint_history()
            
            # Limpa checkpoints antigos
            self._cleanup_old_checkpoints()
            
            self._logger.log_checkpoint(
                checkpoint_path=str(checkpoint_path),
                epoch=epoch,
                metrics=metrics
            )
            
            return str(checkpoint_path)
            
        except Exception as e:
            raise CheckpointError(
                f"Falha ao salvar checkpoint: {e}",
                checkpoint_path=str(checkpoint_path),
                original_exception=e
            )
    
    def load(
        self,
        checkpoint_path: str,
        model_class: Optional[type] = None,
    ) -> Dict[str, Any]:
        """
        Carrega um checkpoint.
        
        Args:
            checkpoint_path: Caminho do checkpoint
            model_class: Classe do modelo para carregar
            
        Returns:
            Dict com modelo e metadados carregados
        """
        path = Path(checkpoint_path)
        
        if not path.exists():
            raise CheckpointError(
                f"Checkpoint não encontrado: {checkpoint_path}",
                checkpoint_path=checkpoint_path
            )
        
        self._logger.info("Carregando checkpoint", path=checkpoint_path)
        
        try:
            result: Dict[str, Any] = {"path": checkpoint_path}
            
            # Carrega metadados
            metadata_path = path / "checkpoint_info.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    result["info"] = CheckpointInfo.from_dict(json.load(f))
            
            # Carrega modelo se classe fornecida
            if model_class is not None:
                if hasattr(model_class, "from_pretrained"):
                    result["model"] = model_class.from_pretrained(path)
                else:
                    import torch
                    model = model_class()
                    model.load_state_dict(
                        torch.load(path / "pytorch_model.bin")
                    )
                    result["model"] = model
            
            return result
            
        except Exception as e:
            raise CheckpointError(
                f"Falha ao carregar checkpoint: {e}",
                checkpoint_path=checkpoint_path,
                original_exception=e
            )
    
    def _save_metadata(self, checkpoint_path: Path, info: CheckpointInfo) -> None:
        """Salva metadados do checkpoint."""
        metadata_path = checkpoint_path / "checkpoint_info.json"
        with open(metadata_path, "w") as f:
            json.dump(info.to_dict(), f, indent=2)
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove checkpoints antigos mantendo apenas max_to_keep."""
        max_to_keep = self._config.max_to_keep
        
        if len(self._checkpoints) <= max_to_keep:
            return
        
        # Ordena por época/step
        sorted_checkpoints = sorted(
            self._checkpoints,
            key=lambda c: (c.epoch, c.step)
        )
        
        # Remove os mais antigos
        to_remove = sorted_checkpoints[:-max_to_keep]
        
        for checkpoint in to_remove:
            checkpoint_path = Path(checkpoint.path)
            if checkpoint_path.exists():
                self._logger.info(
                    "Removendo checkpoint antigo",
                    path=str(checkpoint_path)
                )
                shutil.rmtree(checkpoint_path)
            self._checkpoints.remove(checkpoint)
        
        self._save_checkpoint_history()
    
    def _load_checkpoint_history(self) -> None:
        """Carrega histórico de checkpoints do disco."""
        history_path = self._save_dir / "checkpoint_history.json"
        
        if history_path.exists():
            try:
                with open(history_path) as f:
                    data = json.load(f)
                    self._checkpoints = [
                        CheckpointInfo.from_dict(c) for c in data
                    ]
            except Exception as e:
                self._logger.warning(
                    f"Falha ao carregar histórico de checkpoints: {e}"
                )
                self._checkpoints = []
    
    def _save_checkpoint_history(self) -> None:
        """Salva histórico de checkpoints no disco."""
        history_path = self._save_dir / "checkpoint_history.json"
        
        try:
            with open(history_path, "w") as f:
                json.dump(
                    [c.to_dict() for c in self._checkpoints],
                    f,
                    indent=2
                )
        except Exception as e:
            self._logger.warning(
                f"Falha ao salvar histórico de checkpoints: {e}"
            )
    
    def upload_to_hub(
        self,
        checkpoint_path: str,
        repo_name: str,
        private: bool = False,
    ) -> str:
        """
        Faz upload de checkpoint para o Hugging Face Hub.
        
        Args:
            checkpoint_path: Caminho do checkpoint
            repo_name: Nome do repositório (user/model)
            private: Se o repositório é privado
            
        Returns:
            URL do modelo no Hub
        """
        self._logger.info(
            "Fazendo upload para o Hub",
            checkpoint=checkpoint_path,
            repo=repo_name
        )
        
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            
            # Cria repositório se não existir
            api.create_repo(
                repo_id=repo_name,
                private=private,
                exist_ok=True
            )
            
            # Faz upload
            api.upload_folder(
                folder_path=checkpoint_path,
                repo_id=repo_name,
            )
            
            url = f"https://huggingface.co/{repo_name}"
            self._logger.info("Upload concluído", url=url)
            
            return url
            
        except Exception as e:
            raise CheckpointError(
                f"Falha ao fazer upload para o Hub: {e}",
                checkpoint_path=checkpoint_path,
                original_exception=e
            )
    
    def get_best_checkpoint(
        self,
        metric: str,
        higher_is_better: bool = True
    ) -> Optional[CheckpointInfo]:
        """
        Retorna o melhor checkpoint baseado em uma métrica.
        
        Args:
            metric: Nome da métrica
            higher_is_better: Se maior valor é melhor
            
        Returns:
            Melhor CheckpointInfo ou None
        """
        valid_checkpoints = [
            c for c in self._checkpoints
            if metric in c.metrics
        ]
        
        if not valid_checkpoints:
            return None
        
        if higher_is_better:
            return max(valid_checkpoints, key=lambda c: c.metrics[metric])
        else:
            return min(valid_checkpoints, key=lambda c: c.metrics[metric])
    
    def delete_checkpoint(self, checkpoint_path: str) -> None:
        """
        Remove um checkpoint específico.
        
        Args:
            checkpoint_path: Caminho do checkpoint a remover
        """
        path = Path(checkpoint_path)
        
        if path.exists():
            shutil.rmtree(path)
            self._logger.info("Checkpoint removido", path=checkpoint_path)
        
        # Remove do histórico
        self._checkpoints = [
            c for c in self._checkpoints
            if c.path != checkpoint_path
        ]
        self._save_checkpoint_history()
    
    def clear_all(self) -> None:
        """Remove todos os checkpoints."""
        self._logger.warning("Removendo todos os checkpoints")
        
        for checkpoint in self._checkpoints:
            path = Path(checkpoint.path)
            if path.exists():
                shutil.rmtree(path)
        
        self._checkpoints.clear()
        self._save_checkpoint_history()
    
    def __repr__(self) -> str:
        """Representação string."""
        return (
            f"CheckpointManager("
            f"save_dir='{self._save_dir}', "
            f"checkpoints={len(self._checkpoints)}, "
            f"max_to_keep={self._config.max_to_keep})"
        )

