"""
Classe ModelTrainer para orquestração do processo de fine-tuning.

Este módulo implementa a classe principal que coordena todo o processo
de treinamento, usando o padrão Facade para simplificar a interface.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from modelforge.backends.base import BackendBase
from modelforge.backends.factory import BackendFactory
from modelforge.config.schema import Config
from modelforge.config.security import CredentialManager
from modelforge.core.checkpoint import CheckpointManager
from modelforge.core.evaluator import ModelEvaluator
from modelforge.infrastructure.base import InfrastructureBase
from modelforge.infrastructure.factory import InfrastructureFactory
from modelforge.utils.exceptions import TrainingError
from modelforge.utils.logging import StructuredLogger


@dataclass
class TrainingResult:
    """
    Resultado do treinamento.
    
    Attributes:
        metrics: Métricas finais do treinamento
        best_checkpoint: Caminho do melhor checkpoint
        total_steps: Total de passos de treinamento
        epochs_completed: Número de épocas completadas
        final_loss: Loss final do treinamento
        evaluation_results: Resultados de avaliação por época
    """
    metrics: Dict[str, float] = field(default_factory=dict)
    best_checkpoint: Optional[str] = None
    total_steps: int = 0
    epochs_completed: int = 0
    final_loss: Optional[float] = None
    evaluation_results: List[Dict[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            "metrics": self.metrics,
            "best_checkpoint": self.best_checkpoint,
            "total_steps": self.total_steps,
            "epochs_completed": self.epochs_completed,
            "final_loss": self.final_loss,
            "evaluation_results": self.evaluation_results,
        }


class ModelTrainer:
    """
    Orquestrador principal do processo de fine-tuning.
    
    Implementa o padrão Facade, coordenando:
    - Backend de treinamento
    - Infraestrutura de execução
    - Gerenciamento de checkpoints
    - Avaliação do modelo
    - Logging estruturado
    
    Attributes:
        config: Configuração completa do treinamento
        backend: Backend de treinamento
        infrastructure: Infraestrutura de execução
        checkpoint_manager: Gerenciador de checkpoints
        evaluator: Avaliador de modelo
        logger: Logger estruturado
    
    Example:
        >>> config = ConfigLoader().load("config.yaml")
        >>> trainer = ModelTrainer(config)
        >>> result = trainer.train()
        >>> print(result.metrics)
    """
    
    def __init__(
        self,
        config: Config,
        backend: Optional[BackendBase] = None,
        infrastructure: Optional[InfrastructureBase] = None,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        """
        Inicializa o trainer.
        
        Args:
            config: Configuração completa
            backend: Backend de treinamento (criado automaticamente se não fornecido)
            infrastructure: Infraestrutura (criada automaticamente se não fornecida)
            logger: Logger estruturado
        """
        self._config = config
        self._logger = logger or StructuredLogger(
            "trainer",
            level=config.logging.level,
            json_format=config.logging.format == "json"
        )
        
        # Gerenciador de credenciais
        self._credential_manager = CredentialManager()
        self._credentials = self._credential_manager.load_credentials(
            config.credentials
        )
        
        # Cria backend se não fornecido
        if backend is None:
            backend = BackendFactory.create_backend(
                config.model.repository.value,
                self._credentials,
                logger=self._logger
            )
        self._backend = backend
        
        # Cria infraestrutura se não fornecida
        if infrastructure is None:
            infrastructure = InfrastructureFactory.create_infrastructure(
                config.infrastructure.type.value,
                config.infrastructure
            )
        self._infrastructure = infrastructure
        
        # Gerenciador de checkpoints
        self._checkpoint_manager = CheckpointManager(
            config.checkpoints,
            logger=self._logger
        )
        
        # Estado interno
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._train_dataset: Optional[Any] = None
        self._eval_dataset: Optional[Any] = None
        self._trainer: Optional[Any] = None
        self._is_setup = False
    
    @property
    def config(self) -> Config:
        """Retorna a configuração."""
        return self._config
    
    @property
    def backend(self) -> BackendBase:
        """Retorna o backend."""
        return self._backend
    
    @property
    def model(self) -> Optional[Any]:
        """Retorna o modelo carregado."""
        return self._model
    
    @property
    def tokenizer(self) -> Optional[Any]:
        """Retorna o tokenizer carregado."""
        return self._tokenizer
    
    def setup(self) -> None:
        """
        Configura o ambiente de treinamento.
        
        Carrega modelo, tokenizer e dataset, preparando tudo para o treinamento.
        
        Raises:
            TrainingError: Se a configuração falhar
        """
        if self._is_setup:
            self._logger.warning("Trainer já configurado, pulando setup")
            return
        
        self._logger.info("Iniciando setup do trainer")
        
        try:
            # Configura infraestrutura
            self._infrastructure.setup(self._config.infrastructure)
            self._logger.info(
                "Infraestrutura configurada",
                type=self._config.infrastructure.type.value,
                device=str(self._infrastructure.get_device())
            )
            
            # Carrega tokenizer
            self._tokenizer = self._backend.load_tokenizer(self._config.model)
            
            # Carrega modelo
            self._model = self._backend.load_model(self._config.model)
            
            # Aplica LoRA se configurado
            if self._config.lora and self._config.lora.enabled:
                self._model = self._backend.apply_lora(
                    self._model,
                    self._config.lora
                )
                # Garante que o modelo com LoRA está na GPU
                device = self._infrastructure.get_device()
                if str(device).startswith("cuda"):
                    self._model = self._model.to(device)
                    import torch
                    if next(self._model.parameters()).device.type == "cuda":
                        self._logger.info("Modelo com LoRA está na GPU")
            
            # Aplica gradient checkpointing se configurado
            if self._config.training.gradient_checkpointing:
                if hasattr(self._model, "gradient_checkpointing_enable"):
                    self._model.gradient_checkpointing_enable()
                    self._logger.info("Gradient checkpointing ativado")
                else:
                    self._logger.warning(
                        "Modelo não suporta gradient checkpointing"
                    )
            
            # Carrega dataset
            dataset = self._backend.load_dataset(self._config.dataset)
            
            # Prepara dataset
            processed_dataset = self._backend.prepare_dataset(
                dataset,
                self._tokenizer,
                self._config.dataset
            )
            
            # Separa splits
            train_split = self._config.dataset.splits.train
            val_split = self._config.dataset.splits.validation
            
            self._train_dataset = processed_dataset[train_split]
            
            if val_split and val_split in processed_dataset:
                self._eval_dataset = processed_dataset[val_split]
            
            self._is_setup = True
            self._logger.info("Setup concluído com sucesso")
            
        except Exception as e:
            raise TrainingError(
                f"Falha no setup do trainer: {e}",
                original_exception=e
            )
    
    def train(
        self,
        resume_from_checkpoint: Optional[str] = None
    ) -> TrainingResult:
        """
        Executa o treinamento completo.
        
        Args:
            resume_from_checkpoint: Caminho para checkpoint para resumir
            
        Returns:
            TrainingResult com métricas e informações do treinamento
            
        Raises:
            TrainingError: Se o treinamento falhar
        """
        if not self._is_setup:
            self.setup()
        
        self._logger.info(
            "Iniciando treinamento",
            epochs=self._config.training.epochs,
            batch_size=self._config.training.batch_size,
            learning_rate=self._config.training.learning_rate
        )
        
        try:
            # Cria output dir
            output_dir = Path(self._config.checkpoints.save_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Cria trainer do backend
            self._trainer = self._backend.create_trainer(
                model=self._model,
                dataset=self._train_dataset,
                training_config=self._config.training,
                eval_dataset=self._eval_dataset,
                eval_config=self._config.evaluation,
                output_dir=str(output_dir),
            )
            
            # Executa treinamento
            train_output = self._backend.train(
                self._trainer,
                resume_from_checkpoint=resume_from_checkpoint
            )
            
            # Coleta resultados
            result = TrainingResult(
                metrics=train_output.get("metrics", {}),
                total_steps=train_output.get("global_step", 0),
                epochs_completed=self._config.training.epochs,
                final_loss=train_output.get("metrics", {}).get("train_loss"),
            )
            
            # Salva modelo final
            final_model_path = output_dir / "final_model"
            self._backend.save_model(
                self._model,
                self._tokenizer,
                str(final_model_path)
            )
            result.best_checkpoint = str(final_model_path)
            
            # Push para Hub se configurado
            if self._config.versioning.push_to_hub and self._config.versioning.hub_name:
                self._push_to_hub()
            
            self._logger.info(
                "Treinamento concluído",
                metrics=result.metrics,
                checkpoint=result.best_checkpoint
            )
            
            return result
            
        except Exception as e:
            raise TrainingError(
                f"Erro durante treinamento: {e}",
                original_exception=e
            )
    
    def evaluate(self, dataset: Optional[Any] = None) -> Dict[str, float]:
        """
        Avalia o modelo.
        
        Args:
            dataset: Dataset de avaliação (usa eval_dataset se não fornecido)
            
        Returns:
            Dict com métricas de avaliação
        """
        if not self._is_setup:
            self.setup()
        
        eval_data = dataset or self._eval_dataset
        
        if eval_data is None:
            raise TrainingError("Nenhum dataset de avaliação disponível")
        
        self._logger.info("Iniciando avaliação")
        
        metrics = self._backend.evaluate(self._trainer, eval_data)
        
        self._logger.log_evaluation(
            epoch=self._config.training.epochs,
            metrics=metrics
        )
        
        return metrics
    
    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """
        Salva checkpoint do modelo atual.
        
        Args:
            path: Caminho para salvar (usa config se não fornecido)
            
        Returns:
            Caminho onde foi salvo
        """
        if path is None:
            path = self._config.checkpoints.save_dir
        
        self._logger.info("Salvando checkpoint", path=path)
        
        saved_path = self._backend.save_model(
            self._model,
            self._tokenizer,
            path
        )
        
        return saved_path
    
    def _push_to_hub(self) -> str:
        """
        Faz push do modelo para o Hugging Face Hub.
        
        Returns:
            URL do modelo no Hub
        """
        hub_name = self._config.versioning.hub_name
        private = self._config.versioning.private
        
        self._logger.info(
            "Fazendo push para o Hub",
            repo=hub_name,
            private=private
        )
        
        url = self._backend.push_to_hub(
            self._model,
            self._tokenizer,
            hub_name,
            private=private
        )
        
        return url
    
    def cleanup(self) -> None:
        """Limpa recursos alocados."""
        self._logger.info("Limpando recursos")
        
        self._backend.cleanup()
        self._model = None
        self._tokenizer = None
        self._train_dataset = None
        self._eval_dataset = None
        self._trainer = None
        self._is_setup = False
    
    def __enter__(self) -> "ModelTrainer":
        """Context manager entry."""
        self.setup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.cleanup()
    
    def __repr__(self) -> str:
        """Representação string do trainer."""
        return (
            f"ModelTrainer("
            f"model='{self._config.model.name}', "
            f"backend='{self._config.model.repository.value}', "
            f"is_setup={self._is_setup})"
        )

