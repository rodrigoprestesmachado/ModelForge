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
        
        # Configura padrões globais de logging
        from modelforge.utils.logging import set_logging_defaults
        set_logging_defaults(
            json_format=config.logging.format == "json",
            level=config.logging.level
        )
        
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
                config.infrastructure,
                logger=self._logger
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
            
            # Carrega modelo com otimização de memória (FP16 se configurado)
            use_fp16 = self._config.training.fp16 if self._config.training else None
            
            # IMPORTANTE: Desabilita device_map quando LoRA está habilitado
            # device_map pode causar problemas com treinamento de modelos LoRA
            use_device_map = None  # Auto-detect por padrão
            if self._config.lora and self._config.lora.enabled:
                use_device_map = False
                self._logger.info(
                    "LoRA habilitado: desabilitando device_map para evitar problemas com treinamento"
                )
            
            self._model = self._backend.load_model(
                self._config.model, 
                use_fp16=use_fp16,
                use_device_map=use_device_map
            )
            
            # Aplica LoRA se configurado
            if self._config.lora and self._config.lora.enabled:
                self._model = self._backend.apply_lora(
                    self._model,
                    self._config.lora
                )
                # O apply_lora já move o modelo para o device correto
                # Apenas verifica se está tudo OK
                device = self._infrastructure.get_device()
                try:
                    actual_device = next(self._model.parameters()).device
                    self._logger.info(
                        "Modelo com LoRA configurado",
                        expected_device=str(device),
                        actual_device=str(actual_device),
                        is_on_gpu=(actual_device.type == "cuda" if str(device).startswith("cuda") else True)
                    )
                except StopIteration:
                    self._logger.warning("Não foi possível verificar device do modelo com LoRA")
            
            # IMPORTANTE: Garante que o modelo está em modo de treinamento
            # e que os parâmetros têm requires_grad=True antes de aplicar gradient checkpointing
            self._model.train()
            
            # Verifica e habilita requires_grad para parâmetros treináveis
            # Para modelos LoRA/PEFT, apenas os parâmetros LoRA devem ter requires_grad=True
            # Mas precisamos garantir que pelo menos alguns parâmetros estejam treináveis
            trainable_params = [name for name, param in self._model.named_parameters() if param.requires_grad]
            is_peft_model = hasattr(self._model, "peft_config") or hasattr(self._model, "get_peft_model")
            
            if len(trainable_params) == 0:
                self._logger.warning(
                    "Nenhum parâmetro está treinável antes de aplicar gradient checkpointing. "
                    "Tentando habilitar requires_grad..."
                )
                if is_peft_model:
                    # Para modelos PEFT, tenta habilitar requires_grad para parâmetros LoRA
                    for name, param in self._model.named_parameters():
                        if "lora" in name.lower():
                            param.requires_grad = True
                            trainable_params.append(name)
                    # Se ainda não encontrou, tenta enable_input_require_grads
                    if len(trainable_params) == 0 and hasattr(self._model, "enable_input_require_grads"):
                        self._model.enable_input_require_grads()
                else:
                    # Para modelos normais, habilita requires_grad para todos os parâmetros
                    for name, param in self._model.named_parameters():
                        param.requires_grad = True
                        trainable_params.append(name)
            
            if len(trainable_params) > 0:
                self._logger.info(
                    "Parâmetros treináveis configurados",
                    count=len(trainable_params),
                    is_peft_model=is_peft_model
                )
            else:
                # Se ainda não há parâmetros treináveis, isso é um problema crítico
                self._logger.error(
                    "ERRO CRÍTICO: Nenhum parâmetro treinável após todas as tentativas!"
                )
                # Lista alguns parâmetros para debug
                param_names = [name for name, _ in list(self._model.named_parameters())[:10]]
                self._logger.debug(f"Primeiros 10 parâmetros: {param_names}")
            
            # Aplica gradient checkpointing se configurado
            # IMPORTANTE: Gradient checkpointing deve ser aplicado DEPOIS de garantir
            # que os parâmetros têm requires_grad=True
            if self._config.training.gradient_checkpointing:
                if hasattr(self._model, "gradient_checkpointing_enable"):
                    # IMPORTANTE: Para modelos PEFT com gradient_checkpointing, precisamos
                    # chamar enable_input_require_grads() ANTES de habilitar gradient_checkpointing.
                    # Isso faz com que os embeddings tenham requires_grad=True, que é necessário
                    # para que gradient_checkpointing funcione (evita o erro:
                    # "None of the inputs have requires_grad=True")
                    if is_peft_model and hasattr(self._model, "enable_input_require_grads"):
                        self._model.enable_input_require_grads()
                        self._logger.info(
                            "enable_input_require_grads() chamado para compatibilidade "
                            "LoRA + gradient_checkpointing"
                        )
                    
                    # Para modelos PEFT, precisamos garantir que o modelo base também
                    # está configurado corretamente para gradient checkpointing
                    if is_peft_model and hasattr(self._model, "base_model"):
                        # Aplica gradient checkpointing no modelo base
                        if hasattr(self._model.base_model, "gradient_checkpointing_enable"):
                            self._model.base_model.gradient_checkpointing_enable()
                    self._model.gradient_checkpointing_enable()
                    
                    # IMPORTANTE: Desabilita use_cache quando gradient checkpointing está ativo
                    # use_cache=True é incompatível com gradient checkpointing
                    if hasattr(self._model, "config"):
                        if hasattr(self._model.config, "use_cache"):
                            self._model.config.use_cache = False
                            self._logger.info("use_cache desabilitado (incompatível com gradient checkpointing)")
                        # Também verifica generation_config
                        if hasattr(self._model, "generation_config") and self._model.generation_config is not None:
                            if hasattr(self._model.generation_config, "use_cache"):
                                self._model.generation_config.use_cache = False
                    
                    self._logger.info("Gradient checkpointing ativado")
                else:
                    self._logger.warning(
                        "Modelo não suporta gradient checkpointing"
                    )
            
            # Carrega dataset
            dataset = self._backend.load_dataset(self._config.dataset)
            
            # Prepara dataset (passa a task do modelo para criar labels corretamente)
            processed_dataset = self._backend.prepare_dataset(
                dataset,
                self._tokenizer,
                self._config.dataset,
                model_task=self._config.model.task
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
        
        # Limpa cache CUDA antes do treinamento para liberar memória
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                self._logger.info(
                    "Cache CUDA limpo antes do treinamento",
                    memory_allocated=f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                    memory_reserved=f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
                )
        except Exception as e:
            self._logger.warning(
                "Não foi possível limpar cache CUDA",
                error=str(e)
            )
        
        try:
            # Validação final antes de iniciar o treinamento
            self._validate_before_training()
            
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
    
    def _validate_before_training(self) -> None:
        """
        Valida que tudo está configurado corretamente antes de iniciar o treinamento.
        
        Raises:
            TrainingError: Se alguma validação falhar
        """
        self._logger.info("Validando configuração antes do treinamento")
        
        # Valida modelo
        if self._model is None:
            raise TrainingError("Modelo não foi carregado")
        
        # Valida tokenizer
        if self._tokenizer is None:
            raise TrainingError("Tokenizer não foi carregado")
        
        # Valida dataset
        if self._train_dataset is None:
            raise TrainingError("Dataset de treinamento não foi carregado")
        
        # Valida que o modelo está em modo de treinamento
        if not self._model.training:
            self._logger.warning("Modelo não está em modo de treinamento, corrigindo...")
            self._model.train()
        
        # Valida que há parâmetros treináveis
        import torch
        trainable_params = [name for name, param in self._model.named_parameters() if param.requires_grad]
        if len(trainable_params) == 0:
            raise TrainingError(
                "Nenhum parâmetro treinável encontrado. "
                "Verifique se LoRA está configurado corretamente ou se o modelo foi congelado."
            )
        
        # Valida device (se GPU está disponível, modelo deve estar na GPU)
        device = self._infrastructure.get_device()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            try:
                sample_param = next(self._model.parameters())
                if sample_param.device.type != "cuda":
                    self._logger.warning(
                        f"Modelo não está na GPU. Esperado: cuda, Atual: {sample_param.device.type}. "
                        "Tentando mover para GPU..."
                    )
                    self._model = self._model.to(device)
                    sample_param = next(self._model.parameters())
                    if sample_param.device.type != "cuda":
                        raise TrainingError(
                            f"Falha ao mover modelo para GPU. Device atual: {sample_param.device}"
                        )
            except StopIteration:
                self._logger.warning("Não foi possível verificar device do modelo")
        
        # Valida dataset não está vazio
        try:
            dataset_size = len(self._train_dataset)
            if dataset_size == 0:
                raise TrainingError("Dataset de treinamento está vazio")
            self._logger.info(f"Dataset de treinamento tem {dataset_size} exemplos")
        except (TypeError, AttributeError):
            # Dataset pode ser streaming ou não ter __len__
            self._logger.info("Dataset de treinamento carregado (tamanho não disponível)")
        
        # Valida configuração de gradient checkpointing
        if self._config.training.gradient_checkpointing:
            # Verifica se use_cache está desabilitado
            if hasattr(self._model, "config") and hasattr(self._model.config, "use_cache"):
                if self._model.config.use_cache:
                    self._logger.warning(
                        "use_cache está habilitado com gradient checkpointing. Desabilitando..."
                    )
                    self._model.config.use_cache = False
        
        self._logger.info(
            "Validação concluída",
            trainable_params=len(trainable_params),
            model_training=self._model.training,
            device=str(device)
        )
    
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

