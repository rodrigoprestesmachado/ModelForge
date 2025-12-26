"""
Classe HuggingFaceBackend para treinamento com Hugging Face Transformers.

Este módulo implementa o backend para modelos do Hugging Face Hub,
usando a biblioteca transformers e datasets.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from modelforge.backends.base import BackendBase
from modelforge.config.schema import (
    DatasetConfig,
    EvaluationConfig,
    LoRAConfig,
    ModelConfig,
    TrainingConfig,
)
from modelforge.utils.exceptions import BackendError, DatasetError
from modelforge.utils.logging import StructuredLogger


class HuggingFaceBackend(BackendBase):
    """
    Backend para treinamento com Hugging Face Transformers.
    
    Implementa todas as operações necessárias para:
    - Carregar modelos e tokenizers do Hub
    - Carregar e processar datasets
    - Treinar modelos com Trainer
    - Avaliar e salvar modelos
    
    Attributes:
        credentials: Credenciais (incluindo HF_TOKEN)
        logger: Logger estruturado
    
    Example:
        >>> backend = HuggingFaceBackend({"HF_TOKEN": "hf_..."})
        >>> model = backend.load_model(model_config)
        >>> tokenizer = backend.load_tokenizer(model_config)
    """
    
    def __init__(
        self,
        credentials: Optional[Dict[str, str]] = None,
        logger: Optional[StructuredLogger] = None
    ) -> None:
        """
        Inicializa o backend Hugging Face.
        
        Args:
            credentials: Dicionário com credenciais (HF_TOKEN)
            logger: Logger para mensagens
        """
        super().__init__(credentials)
        self._logger = logger or StructuredLogger("huggingface_backend")
        self._device = None
        self._training_args = None
        
        # Configura autenticação
        self._setup_auth()
    
    def _setup_auth(self) -> None:
        """Configura autenticação com o Hugging Face Hub."""
        token = self._credentials.get("HF_TOKEN") or self._credentials.get("HUGGINGFACE_TOKEN")
        
        if token:
            try:
                from huggingface_hub import login
                login(token=token, add_to_git_credential=False)
                self._logger.info("Autenticado no Hugging Face Hub")
            except Exception as e:
                self._logger.warning(
                    "Falha na autenticação do HF Hub",
                    error=str(e)
                )
    
    def load_model(self, config: ModelConfig) -> Any:
        """
        Carrega um modelo do Hugging Face Hub.
        
        Args:
            config: Configuração do modelo
            
        Returns:
            Modelo PreTrainedModel
            
        Raises:
            BackendError: Se o modelo não puder ser carregado
        """
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoModelForCausalLM,
                AutoModel
            )
            
            self._logger.info(
                "Carregando modelo",
                model_name=config.name,
                task=config.task
            )
            
            model_kwargs: Dict[str, Any] = {
                "pretrained_model_name_or_path": config.name,
            }
            
            if config.revision:
                model_kwargs["revision"] = config.revision
            
            token = self._credentials.get("HF_TOKEN")
            if token:
                model_kwargs["token"] = token
            
            # Obtém dispositivo - sempre move explicitamente para garantir uso de GPU
            device = self.get_device()
            import torch
            
            # Configura device_map para carregar diretamente na GPU quando disponível
            # Isso evita carregar na CPU primeiro e depois mover, economizando memória
            use_device_map = False
            if str(device).startswith("cuda") and torch.cuda.is_available():
                # Carrega diretamente na GPU usando device_map
                # Isso é mais eficiente que carregar na CPU e depois mover
                model_kwargs["device_map"] = "auto"
                model_kwargs["low_cpu_mem_usage"] = True
                use_device_map = True
                self._logger.info(
                    "Carregando modelo diretamente na GPU usando device_map",
                    device_map="auto"
                )
            else:
                # CPU ou MPS - não usa device_map
                model_kwargs["low_cpu_mem_usage"] = True
                self._logger.info("Carregando modelo na CPU/MPS")
            
            # Seleciona a classe de modelo baseado na tarefa
            if config.task == "text-generation" or config.task == "causal-lm":
                # Modelos de geração de texto (causal language models)
                model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            elif config.task == "text-classification" or config.num_labels:
                # Modelos de classificação
                if config.num_labels:
                    model_kwargs["num_labels"] = config.num_labels
                model = AutoModelForSequenceClassification.from_pretrained(
                    **model_kwargs
                )
            else:
                # Modelo genérico como fallback
                model = AutoModel.from_pretrained(**model_kwargs)
            
            # Se não usou device_map, move para o dispositivo explicitamente
            if not use_device_map:
                model = model.to(device)
            
            # Verifica se está realmente na GPU (quando usando device_map, verifica de forma diferente)
            if str(device).startswith("cuda"):
                if use_device_map:
                    # Quando usando device_map, verifica através do hf_device_map
                    if hasattr(model, "hf_device_map"):
                        device_map_info = model.hf_device_map
                        self._logger.info(
                            "Modelo carregado com device_map",
                            device_map=str(device_map_info)[:200]  # Primeiros 200 chars
                        )
                    # Tenta verificar através dos parâmetros
                    try:
                        sample_param = next(model.parameters())
                        actual_device = sample_param.device
                        if actual_device.type == "cuda":
                            self._logger.info(
                                "Modelo está na GPU (device_map)",
                                device=str(actual_device)
                            )
                    except StopIteration:
                        # Modelo sem parâmetros? Improvável mas possível
                        self._logger.warning("Não foi possível verificar device do modelo")
                else:
                    # Verificação normal para modelos sem device_map
                    actual_device = next(model.parameters()).device
                    if actual_device.type != "cuda":
                        self._logger.warning(
                            "Modelo não está na GPU, tentando mover novamente..."
                        )
                        model = model.to(device)
                        actual_device = next(model.parameters()).device
                        if actual_device.type == "cuda":
                            self._logger.info("Modelo agora está na GPU")
                        else:
                            self._logger.error(
                                "Falha ao mover modelo para GPU!",
                                actual_device=str(actual_device)
                            )
            
            self._model = model
            
            # Log detalhado sobre o device
            try:
                actual_device = next(model.parameters()).device
                self._logger.info(
                    "Modelo carregado com sucesso",
                    model_name=config.name,
                    task=config.task,
                    target_device=str(device),
                    actual_device=str(actual_device),
                    on_gpu=(actual_device.type == "cuda"),
                    used_device_map=use_device_map
                )
            except StopIteration:
                # Modelo sem parâmetros (improvável)
                self._logger.info(
                    "Modelo carregado com sucesso",
                    model_name=config.name,
                    task=config.task,
                    target_device=str(device),
                    used_device_map=use_device_map
                )
            
            return model
            
        except Exception as e:
            raise BackendError(
                f"Falha ao carregar modelo '{config.name}': {e}",
                backend_type="huggingface",
                operation="load_model",
                original_exception=e
            )
    
    def apply_lora(
        self,
        model: Any,
        lora_config: LoRAConfig
    ) -> Any:
        """
        Aplica LoRA (Low-Rank Adaptation) ao modelo.
        
        Args:
            model: Modelo a ser modificado
            lora_config: Configuração de LoRA
            
        Returns:
            Modelo com LoRA aplicado
        """
        if not lora_config.enabled:
            return model
        
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            self._logger.info(
                "Aplicando LoRA ao modelo",
                r=lora_config.r,
                alpha=lora_config.lora_alpha,
                dropout=lora_config.lora_dropout
            )
            
            # Determina o task_type
            task_type_str = lora_config.task_type
            if task_type_str == "CAUSAL_LM" or task_type_str is None:
                task_type = TaskType.CAUSAL_LM
            elif task_type_str == "SEQ_2_SEQ_LM":
                task_type = TaskType.SEQ_2_SEQ_LM
            else:
                task_type = TaskType.CAUSAL_LM  # Default
                self._logger.warning(
                    f"Task type '{task_type_str}' não reconhecido, usando CAUSAL_LM"
                )
            
            # Cria configuração LoRA
            peft_config = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=lora_config.lora_dropout,
                bias=lora_config.bias,
                task_type=task_type,
            )
            
            # Aplica LoRA ao modelo
            model = get_peft_model(model, peft_config)
            
            self._logger.info("LoRA aplicado com sucesso")
            
            # Imprime estatísticas do modelo
            model.print_trainable_parameters()
            
            return model
            
        except ImportError:
            raise BackendError(
                "Biblioteca 'peft' não encontrada. Instale com: pip install peft",
                backend_type="huggingface",
                operation="apply_lora"
            )
        except Exception as e:
            raise BackendError(
                f"Falha ao aplicar LoRA: {e}",
                backend_type="huggingface",
                operation="apply_lora",
                original_exception=e
            )
    
    def load_tokenizer(self, config: ModelConfig) -> Any:
        """
        Carrega o tokenizer do modelo.
        
        Args:
            config: Configuração do modelo
            
        Returns:
            Tokenizer PreTrainedTokenizer
        """
        try:
            from transformers import AutoTokenizer
            
            self._logger.info("Carregando tokenizer", model_name=config.name)
            
            tokenizer_kwargs = {
                "pretrained_model_name_or_path": config.name,
                "revision": config.revision,
            }
            
            token = self._credentials.get("HF_TOKEN")
            if token:
                tokenizer_kwargs["token"] = token
            
            tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
            
            # Garante que o tokenizer tem pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self._tokenizer = tokenizer
            self._logger.info("Tokenizer carregado com sucesso")
            
            return tokenizer
            
        except Exception as e:
            raise BackendError(
                f"Falha ao carregar tokenizer para '{config.name}': {e}",
                backend_type="huggingface",
                operation="load_tokenizer",
                original_exception=e
            )
    
    def load_dataset(self, config: DatasetConfig) -> Any:
        """
        Carrega um dataset do Hugging Face Hub.
        
        Args:
            config: Configuração do dataset
            
        Returns:
            DatasetDict carregado
        """
        try:
            from datasets import load_dataset
            
            self._logger.info(
                "Carregando dataset",
                dataset_name=config.name,
                subset=config.subset
            )
            
            dataset_kwargs: Dict[str, Any] = {
                "path": config.name,
            }
            
            if config.subset:
                dataset_kwargs["name"] = config.subset
            
            if config.streaming:
                dataset_kwargs["streaming"] = True
            
            token = self._credentials.get("HF_TOKEN")
            if token:
                dataset_kwargs["token"] = token
            
            dataset = load_dataset(**dataset_kwargs)
            
            self._dataset = dataset
            self._logger.info(
                "Dataset carregado com sucesso",
                splits=list(dataset.keys()) if hasattr(dataset, 'keys') else "streaming"
            )
            
            return dataset
            
        except Exception as e:
            raise DatasetError(
                f"Falha ao carregar dataset '{config.name}': {e}",
                dataset_name=config.name,
                original_exception=e
            )
    
    def prepare_dataset(
        self,
        dataset: Any,
        tokenizer: Any,
        config: DatasetConfig
    ) -> Any:
        """
        Prepara o dataset aplicando tokenização.
        
        Args:
            dataset: Dataset bruto
            tokenizer: Tokenizer para processar
            config: Configuração do dataset
            
        Returns:
            Dataset processado
        """
        self._logger.info("Preparando dataset para treinamento")
        
        text_column = config.columns.text
        label_column = config.columns.label
        text_pair_column = config.columns.text_pair
        max_length = config.preprocessing.max_length
        
        def tokenize_function(examples: Dict[str, List]) -> Dict[str, Any]:
            """Função de tokenização para map."""
            if text_pair_column and text_pair_column in examples:
                return tokenizer(
                    examples[text_column],
                    examples[text_pair_column],
                    padding=config.preprocessing.padding,
                    truncation=config.preprocessing.truncation,
                    max_length=max_length,
                )
            else:
                return tokenizer(
                    examples[text_column],
                    padding=config.preprocessing.padding,
                    truncation=config.preprocessing.truncation,
                    max_length=max_length,
                )
        
        # Aplica tokenização
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[
                col for col in dataset[config.splits.train].column_names
                if col not in ["input_ids", "attention_mask", "token_type_ids", label_column]
            ] if label_column else None
        )
        
        # Renomeia coluna de label se necessário
        if label_column and label_column != "labels":
            tokenized_dataset = tokenized_dataset.rename_column(label_column, "labels")
        
        # Define formato para PyTorch
        tokenized_dataset.set_format("torch")
        
        self._logger.info("Dataset preparado com sucesso")
        
        return tokenized_dataset
    
    def create_trainer(
        self,
        model: Any,
        dataset: Any,
        training_config: TrainingConfig,
        eval_dataset: Optional[Any] = None,
        eval_config: Optional[EvaluationConfig] = None,
        output_dir: str = "./output",
        compute_metrics: Optional[Callable] = None,
    ) -> Any:
        """
        Cria um Trainer do Hugging Face configurado.
        
        Args:
            model: Modelo a treinar
            dataset: Dataset de treinamento
            training_config: Configuração de treinamento
            eval_dataset: Dataset de validação
            eval_config: Configuração de avaliação
            output_dir: Diretório de saída
            compute_metrics: Função para calcular métricas
            
        Returns:
            Trainer configurado
        """
        from transformers import Trainer, TrainingArguments
        
        self._logger.info("Criando Trainer")
        
        # Obtém device para garantir uso de GPU
        device = self.get_device()
        
        # Configura TrainingArguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config.epochs,
            per_device_train_batch_size=training_config.batch_size,
            per_device_eval_batch_size=training_config.batch_size,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            warmup_steps=training_config.warmup_steps,
            warmup_ratio=training_config.warmup_ratio,
            lr_scheduler_type=training_config.scheduler.value,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            fp16=training_config.fp16,
            bf16=training_config.bf16,
            max_grad_norm=training_config.max_grad_norm,
            seed=training_config.seed,
            gradient_checkpointing=training_config.gradient_checkpointing,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            eval_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="epoch",
            load_best_model_at_end=eval_dataset is not None,
            report_to="none",  # Desabilita integração com wandb por padrão
        )
        
        # Log sobre device usado no treinamento
        self._logger.info(
            "TrainingArguments criado",
            device=str(device),
            fp16=training_config.fp16,
            bf16=training_config.bf16,
            gradient_checkpointing=training_config.gradient_checkpointing
        )
        
        self._training_args = training_args
        
        # Função de métricas padrão se não fornecida
        if compute_metrics is None and eval_config:
            compute_metrics = self._create_compute_metrics(eval_config.metrics)
        
        # Cria o Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            tokenizer=self._tokenizer,
            compute_metrics=compute_metrics,
        )
        
        self._trainer = trainer
        self._logger.info("Trainer criado com sucesso")
        
        return trainer
    
    def _create_compute_metrics(self, metrics: List[str]) -> Callable:
        """
        Cria função de compute_metrics para o Trainer.
        
        Args:
            metrics: Lista de métricas a calcular
            
        Returns:
            Função compute_metrics
        """
        def compute_metrics(eval_pred) -> Dict[str, float]:
            import evaluate
            
            predictions, labels = eval_pred
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            
            # Para classificação, pega o argmax
            if len(predictions.shape) > 1:
                predictions = np.argmax(predictions, axis=1)
            
            results = {}
            
            for metric_name in metrics:
                try:
                    metric = evaluate.load(metric_name)
                    result = metric.compute(
                        predictions=predictions,
                        references=labels
                    )
                    if isinstance(result, dict):
                        results.update(result)
                    else:
                        results[metric_name] = result
                except Exception as e:
                    self._logger.warning(
                        f"Falha ao calcular métrica {metric_name}",
                        error=str(e)
                    )
            
            return results
        
        return compute_metrics
    
    def train(
        self,
        trainer: Any,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executa o treinamento.
        
        Args:
            trainer: Trainer configurado
            resume_from_checkpoint: Caminho para checkpoint
            
        Returns:
            Dict com resultados do treinamento
        """
        self._logger.info("Iniciando treinamento")
        
        try:
            train_result = trainer.train(
                resume_from_checkpoint=resume_from_checkpoint
            )
            
            metrics = train_result.metrics
            self._logger.info(
                "Treinamento concluído",
                metrics=metrics
            )
            
            return {
                "metrics": metrics,
                "global_step": train_result.global_step,
            }
            
        except Exception as e:
            raise BackendError(
                f"Erro durante treinamento: {e}",
                backend_type="huggingface",
                operation="train",
                original_exception=e
            )
    
    def evaluate(self, trainer: Any, dataset: Any) -> Dict[str, float]:
        """
        Avalia o modelo no dataset.
        
        Args:
            trainer: Trainer com modelo
            dataset: Dataset de avaliação
            
        Returns:
            Dict com métricas
        """
        self._logger.info("Avaliando modelo")
        
        try:
            metrics = trainer.evaluate(eval_dataset=dataset)
            self._logger.info("Avaliação concluída", metrics=metrics)
            return metrics
            
        except Exception as e:
            raise BackendError(
                f"Erro durante avaliação: {e}",
                backend_type="huggingface",
                operation="evaluate",
                original_exception=e
            )
    
    def save_model(
        self,
        model: Any,
        tokenizer: Any,
        output_dir: str
    ) -> str:
        """
        Salva o modelo e tokenizer.
        
        Args:
            model: Modelo a salvar
            tokenizer: Tokenizer a salvar
            output_dir: Diretório de saída
            
        Returns:
            Caminho onde foi salvo
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self._logger.info("Salvando modelo", output_dir=output_dir)
        
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        self._logger.info("Modelo salvo com sucesso")
        
        return str(output_path)
    
    def push_to_hub(
        self,
        model: Any,
        tokenizer: Any,
        repo_name: str,
        private: bool = False
    ) -> str:
        """
        Faz push do modelo para o Hugging Face Hub.
        
        Args:
            model: Modelo a enviar
            tokenizer: Tokenizer a enviar
            repo_name: Nome do repositório
            private: Se é privado
            
        Returns:
            URL do modelo
        """
        self._logger.info(
            "Fazendo push para o Hub",
            repo_name=repo_name,
            private=private
        )
        
        try:
            model.push_to_hub(repo_name, private=private)
            tokenizer.push_to_hub(repo_name, private=private)
            
            url = f"https://huggingface.co/{repo_name}"
            self._logger.info("Push concluído", url=url)
            
            return url
            
        except Exception as e:
            raise BackendError(
                f"Erro ao fazer push para o Hub: {e}",
                backend_type="huggingface",
                operation="push_to_hub",
                original_exception=e
            )
    
    def get_framework(self) -> str:
        """Retorna o framework usado."""
        return "pytorch"
    
    def get_device(self) -> Any:
        """
        Retorna o dispositivo de computação.
        
        Returns:
            torch.device
        """
        if self._device is None:
            import torch
            
            # Verifica CUDA disponível e funcional
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                self._logger.info(
                    "Dispositivo CUDA detectado",
                    device=str(self._device),
                    gpu_count=torch.cuda.device_count(),
                    gpu_name=torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "unknown"
                )
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
                self._logger.info("Dispositivo MPS detectado", device=str(self._device))
            else:
                self._device = torch.device("cpu")
                self._logger.warning(
                    "Nenhuma GPU detectada, usando CPU",
                    device=str(self._device)
                )
        
        return self._device
    
    def load_from_checkpoint(self, checkpoint_path: str, config: ModelConfig) -> Any:
        """
        Carrega modelo de um checkpoint local.
        
        Args:
            checkpoint_path: Caminho do checkpoint
            config: Configuração do modelo
            
        Returns:
            Modelo carregado
        """
        from transformers import AutoModelForSequenceClassification, AutoModel
        
        self._logger.info("Carregando de checkpoint", path=checkpoint_path)
        
        model_kwargs = {
            "pretrained_model_name_or_path": checkpoint_path,
        }
        
        if config.num_labels:
            model_kwargs["num_labels"] = config.num_labels
            model = AutoModelForSequenceClassification.from_pretrained(**model_kwargs)
        else:
            model = AutoModel.from_pretrained(**model_kwargs)
        
        device = self.get_device()
        model = model.to(device)
        
        self._model = model
        return model

