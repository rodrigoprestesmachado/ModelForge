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
    
    def load_model(
        self, 
        config: ModelConfig, 
        use_fp16: Optional[bool] = None,
        use_device_map: Optional[bool] = None
    ) -> Any:
        """
        Carrega um modelo do Hugging Face Hub.
        
        Args:
            config: Configuração do modelo
            use_fp16: Se deve carregar o modelo em FP16 (None = auto-detect)
            use_device_map: Se deve usar device_map (None = auto-detect, False = desabilitado)
            
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
                task=config.task,
                use_fp16=use_fp16
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
            
            # Configura torch_dtype para FP16 se solicitado (economiza memória)
            if use_fp16 and str(device).startswith("cuda") and torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16
                self._logger.info(
                    "Carregando modelo em FP16 para economizar memória",
                    torch_dtype="float16"
                )
            
            # Configura device_map para carregar diretamente na GPU quando disponível
            # AVISO: device_map="auto" pode causar problemas com treinamento, especialmente com LoRA
            # Se use_device_map=False, não usa device_map mesmo que GPU esteja disponível
            use_device_map_flag = False
            if use_device_map is False:
                # Explicitamente desabilitado
                use_device_map_flag = False
                self._logger.info("device_map desabilitado explicitamente")
            elif use_device_map is True:
                # Explicitamente habilitado
                if str(device).startswith("cuda") and torch.cuda.is_available():
                    use_device_map_flag = True
            else:
                # Auto-detect: usa device_map se GPU disponível (comportamento padrão)
                if str(device).startswith("cuda") and torch.cuda.is_available():
                    use_device_map_flag = True
            
            if use_device_map_flag:
                # Carrega diretamente na GPU usando device_map
                # Isso é mais eficiente que carregar na CPU e depois mover
                # NOTA: Pode causar problemas com treinamento, especialmente com LoRA
                model_kwargs["device_map"] = "auto"
                model_kwargs["low_cpu_mem_usage"] = True
                
                # Configura max_memory para evitar OOM (deixa ~1GB livre)
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    # Reserva ~1GB para operações e buffers
                    max_memory_gb = max(1, (gpu_memory / 1024**3) - 1)
                    model_kwargs["max_memory"] = {0: f"{max_memory_gb:.0f}GB"}
                    self._logger.info(
                        "Configurando max_memory para evitar OOM",
                        max_memory_gb=f"{max_memory_gb:.2f}",
                        total_gpu_memory_gb=f"{gpu_memory / 1024**3:.2f}"
                    )
                except Exception as e:
                    self._logger.warning(
                        "Não foi possível configurar max_memory",
                        error=str(e)
                    )
                
                self._logger.info(
                    "Carregando modelo diretamente na GPU usando device_map",
                    device_map="auto"
                )
            else:
                # CPU, MPS ou device_map desabilitado - não usa device_map
                model_kwargs["low_cpu_mem_usage"] = True
                if not str(device).startswith("cuda") or not torch.cuda.is_available():
                    self._logger.info("Carregando modelo na CPU/MPS")
                else:
                    self._logger.info("Carregando modelo na GPU sem device_map")
            
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
            if not use_device_map_flag:
                model = model.to(device)
            
            # Verifica se está realmente na GPU (quando usando device_map, verifica de forma diferente)
            if str(device).startswith("cuda"):
                if use_device_map_flag:
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
            
            # IMPORTANTE: Garante que o modelo está em modo de treinamento
            # Isso é crítico quando usando device_map="auto" que pode carregar em modo de inferência
            model.train()
            
            # Quando usando device_map="auto", o modelo pode ser carregado em modo de inferência
            # Precisamos garantir que os parâmetros possam calcular gradientes
            # NOTA: Se LoRA for aplicado depois, o PEFT vai gerenciar requires_grad automaticamente
            # Mas para o modelo base, precisamos garantir que está configurado para treinamento
            trainable_count = 0
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    # Habilita requires_grad para parâmetros que não são buffers
                    # Buffers geralmente não precisam de gradientes (ex: running_mean em BatchNorm)
                    if "buffer" not in name.lower() and "running" not in name.lower():
                        param.requires_grad = True
                        trainable_count += 1
            
            if trainable_count > 0:
                self._logger.info(
                    f"Habilitado requires_grad para {trainable_count} parâmetros após carregamento com device_map"
                )
            
            self._model = model
            
            # Log detalhado sobre o device e parâmetros treináveis
            try:
                actual_device = next(model.parameters()).device
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                
                self._logger.info(
                    "Modelo carregado com sucesso",
                    model_name=config.name,
                    task=config.task,
                    target_device=str(device),
                    actual_device=str(actual_device),
                    on_gpu=(actual_device.type == "cuda"),
                    used_device_map=use_device_map,
                    trainable_params=trainable_params,
                    total_params=total_params,
                    trainable_ratio=f"{trainable_params/total_params*100:.2f}%" if total_params > 0 else "0%"
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
    
    def _detect_target_modules(self, model: Any) -> List[str]:
        """
        Detecta automaticamente os módulos alvo para LoRA baseado na arquitetura do modelo.
        
        Args:
            model: Modelo a ser analisado
            
        Returns:
            Lista de nomes de módulos alvo
        """
        # Obtém a arquitetura do modelo
        model_type = getattr(model.config, "model_type", None) if hasattr(model, "config") else None
        
        # Tenta obter o nome do modelo do config para detecção adicional
        model_name = None
        if hasattr(model, "config") and hasattr(model.config, "name_or_path"):
            model_name = model.config.name_or_path
        elif hasattr(model, "name_or_path"):
            model_name = model.name_or_path
        
        # Padrões comuns de módulos para diferentes arquiteturas
        module_patterns = {
            "gemma": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "phi": ["q_proj", "k_proj", "v_proj", "dense"],
            "gpt2": ["c_attn", "c_proj"],
            "gpt_neox": ["query_key_value", "dense"],
            "opt": ["q_proj", "k_proj", "v_proj", "out_proj"],
            "bloom": ["query_key_value", "dense_h_to_4h", "dense_4h_to_h"],
            "bert": ["query", "key", "value", "dense"],
            "roberta": ["query", "key", "value", "dense"],
        }
        
        # Verifica se o nome do modelo contém "gemma" (para Gemma 3 ou outras variantes)
        if model_name and "gemma" in model_name.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            self._logger.info(
                "Módulos LoRA detectados por nome do modelo (Gemma)",
                model_name=model_name,
                target_modules=target_modules
            )
            return target_modules
        
        # Tenta usar padrão específico da arquitetura
        if model_type and model_type.lower() in module_patterns:
            target_modules = module_patterns[model_type.lower()]
            self._logger.info(
                "Módulos LoRA detectados por arquitetura",
                model_type=model_type,
                target_modules=target_modules
            )
            return target_modules
        
        # Se não encontrou padrão específico, tenta detectar automaticamente
        # Procura por módulos comuns de attention
        all_module_names = [name for name, _ in model.named_modules()]
        
        # Para modelos com language_model (como Gemma 3 IT), procura dentro do language_model
        language_model_modules = [
            name for name in all_module_names
            if "language_model" in name
        ]
        
        # Padrões de busca para módulos de attention
        attention_patterns = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # LLaMA, Gemma, Mistral
            "query", "key", "value", "dense",  # BERT, RoBERTa
            "c_attn", "c_proj",  # GPT-2
            "query_key_value",  # GPT-NeoX, Bloom
            "out_proj",  # OPT
        ]
        
        # Encontra módulos que correspondem aos padrões
        # Usa apenas os nomes finais dos módulos (ex: "q_proj" em vez de "model.language_model.layers.0.q_proj")
        found_module_patterns = set()
        found_modules = []
        
        for pattern in attention_patterns:
            for module_name in all_module_names:
                # Para modelos com language_model, prioriza módulos dentro do language_model
                if language_model_modules:
                    if pattern in module_name and "language_model" in module_name:
                        # Adiciona o padrão (nome final) em vez do nome completo
                        if pattern not in found_module_patterns:
                            found_module_patterns.add(pattern)
                            found_modules.append(pattern)
                else:
                    # Se não tem language_model, usa qualquer módulo que corresponda
                    if pattern in module_name:
                        if pattern not in found_module_patterns:
                            found_module_patterns.add(pattern)
                            found_modules.append(pattern)
        
        # Se encontrou módulos, retorna
        if found_modules:
            self._logger.info(
                "Módulos LoRA detectados automaticamente",
                target_modules=found_modules
            )
            return found_modules
        
        # Fallback: tenta usar todos os módulos lineares
        linear_modules = []
        for name, module in model.named_modules():
            if hasattr(module, "weight") and len(module.weight.shape) == 2:
                # É uma camada linear
                linear_modules.append(name)
        
        if linear_modules:
            # Limita a módulos de attention (geralmente contêm "attn" ou "attention")
            attention_linear = [
                name for name in linear_modules
                if "attn" in name.lower() or "attention" in name.lower()
            ]
            if attention_linear:
                self._logger.info(
                    "Módulos LoRA detectados (camadas lineares de attention)",
                    target_modules=attention_linear[:8]  # Limita a 8 módulos
                )
                return attention_linear[:8]
        
        # Último fallback: módulos padrão para modelos transformer
        default_modules = ["q_proj", "v_proj"]
        self._logger.warning(
            "Não foi possível detectar módulos automaticamente, usando padrão",
            target_modules=default_modules
        )
        return default_modules
    
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
            import torch
            
            # IMPORTANTE: Consolida o modelo em um único dispositivo antes de aplicar LoRA
            # device_map pode causar problemas com LoRA durante treinamento
            device = self.get_device()
            
            # Verifica se o modelo está usando device_map
            has_device_map = hasattr(model, "hf_device_map")
            
            if has_device_map:
                self._logger.warning(
                    "Modelo está usando device_map. Consolidando em um único dispositivo "
                    "antes de aplicar LoRA para evitar problemas com treinamento..."
                )
                try:
                    # Para modelos com device_map, precisamos desabilitar o device_map
                    # e mover todos os parâmetros para um único dispositivo
                    # Primeiro, tenta desabilitar o device_map através do accelerate
                    try:
                        from accelerate import dispatch_model, infer_auto_device_map
                        from accelerate.utils import get_balanced_memory
                        
                        # Tenta obter o modelo base se for um modelo wrapped
                        base_model = model
                        if hasattr(model, "base_model"):
                            base_model = model.base_model
                        elif hasattr(model, "model"):
                            base_model = model.model
                        
                        # Move todos os parâmetros para o dispositivo
                        for param in base_model.parameters():
                            if param.device != device:
                                param.data = param.data.to(device)
                        
                        # Remove device_map se existir
                        if hasattr(model, "hf_device_map"):
                            delattr(model, "hf_device_map")
                        
                        self._logger.info(
                            "Modelo consolidado em um único dispositivo",
                            device=str(device)
                        )
                    except ImportError:
                        # Se accelerate não estiver disponível, tenta método simples
                        self._logger.info("Accelerate não disponível, usando método simples")
                        model = model.to(device)
                        if hasattr(model, "hf_device_map"):
                            delattr(model, "hf_device_map")
                        self._logger.info("Modelo movido para dispositivo", device=str(device))
                except Exception as e:
                    self._logger.warning(
                        f"Erro ao consolidar modelo: {e}. "
                        "Continuando, mas pode haver problemas com treinamento..."
                    )
            
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
            
            # Detecta target_modules se não especificado
            target_modules = lora_config.target_modules
            if target_modules is None or (isinstance(target_modules, list) and len(target_modules) == 0):
                target_modules = self._detect_target_modules(model)
                self._logger.info(
                    "target_modules detectado automaticamente",
                    target_modules=target_modules
                )
            
            # Cria configuração LoRA
            peft_config = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_config.lora_dropout,
                bias=lora_config.bias,
                task_type=task_type,
            )
            
            # Aplica LoRA ao modelo
            model = get_peft_model(model, peft_config)
            
            # IMPORTANTE: Garante que o modelo está em modo de treinamento
            model.train()
            
            # IMPORTANTE: Para modelos PEFT, precisamos chamar enable_input_require_grads()
            # para garantir compatibilidade com gradient_checkpointing.
            # Isso faz com que os embeddings (saídas da camada de embedding) tenham
            # requires_grad=True, o que é necessário para que gradient_checkpointing funcione.
            # Chamamos isso cedo, independente de gradient_checkpointing estar ativo,
            # pois não causa problemas e previne erros caso seja habilitado depois.
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
                self._logger.debug(
                    "enable_input_require_grads() chamado para compatibilidade com gradient_checkpointing"
                )
            
            # IMPORTANTE: Move o modelo para o dispositivo correto após aplicar LoRA
            # Isso garante que todos os parâmetros (incluindo LoRA) estão no device correto
            try:
                model = model.to(device)
                # Verifica se está realmente no device correto
                sample_param = next(model.parameters())
                if sample_param.device != device:
                    self._logger.warning(
                        f"Modelo não está no device esperado. Esperado: {device}, Atual: {sample_param.device}"
                    )
                    # Tenta mover novamente
                    model = model.to(device)
            except Exception as e:
                self._logger.warning(
                    f"Erro ao mover modelo para device após LoRA: {e}"
                )
            
            # Verifica se há parâmetros treináveis
            trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
            if len(trainable_params) == 0:
                self._logger.error(
                    "CRÍTICO: Nenhum parâmetro treinável encontrado após aplicar LoRA!"
                )
                # Tenta corrigir habilitando requires_grad para parâmetros LoRA
                for name, param in model.named_parameters():
                    if "lora" in name.lower():
                        param.requires_grad = True
                        trainable_params.append(name)
                        self._logger.info(f"Forçando requires_grad=True para: {name}")
                
                # Se ainda não encontrou, tenta enable_input_require_grads
                if len(trainable_params) == 0 and hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
                    self._logger.info("Usando enable_input_require_grads()")
                
                if len(trainable_params) == 0:
                    raise BackendError(
                        "Falha crítica: Nenhum parâmetro treinável após aplicar LoRA. "
                        "Verifique a configuração do modelo e LoRA.",
                        backend_type="huggingface",
                        operation="apply_lora"
                    )
            else:
                self._logger.info(
                    "LoRA aplicado com sucesso",
                    trainable_params=len(trainable_params),
                    device=str(device)
                )
            
            # Imprime estatísticas do modelo
            model.print_trainable_parameters()
            
            # Verificação final: garante que o modelo está pronto para treinamento
            try:
                sample_param = next(model.parameters())
                actual_device = sample_param.device
                self._logger.info(
                    "Validação final do modelo com LoRA",
                    device=str(actual_device),
                    is_training=model.training,
                    trainable_count=len(trainable_params)
                )
            except Exception as e:
                self._logger.warning(f"Erro na validação final: {e}")
            
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
        config: DatasetConfig,
        model_task: Optional[str] = None
    ) -> Any:
        """
        Prepara o dataset aplicando tokenização.
        
        Args:
            dataset: Dataset bruto
            tokenizer: Tokenizer para processar
            config: Configuração do dataset
            model_task: Tarefa do modelo (text-generation, text-classification, etc.)
            
        Returns:
            Dataset processado
        """
        self._logger.info("Preparando dataset para treinamento")
        
        text_column = config.columns.text
        label_column = config.columns.label
        text_pair_column = config.columns.text_pair
        max_length = config.preprocessing.max_length
        
        # Detecta se é uma tarefa de geração de texto (causal LM)
        is_causal_lm = model_task in ["text-generation", "causal-lm"] or (
            model_task is None and hasattr(self, "_model") and 
            hasattr(self._model, "config") and
            getattr(self._model.config, "model_type", None) in ["gemma", "llama", "gpt2"]
        )
        
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
        
        # Para modelos de causal LM (text-generation), cria labels a partir dos input_ids
        # se não houver uma coluna de label explícita
        if is_causal_lm and not label_column:
            def create_labels(examples: Dict[str, List]) -> Dict[str, Any]:
                """Cria labels para causal language modeling."""
                labels = []
                input_ids_list = examples["input_ids"]
                attention_mask_list = examples["attention_mask"]
                
                for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
                    # Converte para lista se necessário e copia
                    if isinstance(input_ids, list):
                        label = input_ids.copy()
                    else:
                        # Se for tensor ou array, converte para lista
                        label = list(input_ids)
                    
                    # Converte attention_mask para lista se necessário
                    if not isinstance(attention_mask, list):
                        attention_mask = list(attention_mask)
                    
                    # Define -100 para tokens de padding (ignorados no cálculo da loss)
                    # -100 é o valor padrão usado pelo PyTorch para ignorar tokens
                    for i, mask in enumerate(attention_mask):
                        if mask == 0:
                            label[i] = -100
                    
                    labels.append(label)
                return {"labels": labels}
            
            tokenized_dataset = tokenized_dataset.map(
                create_labels,
                batched=True,
                desc="Criando labels para causal LM"
            )
            self._logger.info("Labels criados para causal language modeling")
        elif label_column and label_column != "labels":
            # Renomeia coluna de label se necessário
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
        
        # IMPORTANTE: Se gradient_checkpointing está ativo, garante que use_cache está desabilitado
        # Isso evita o warning e problemas com gradientes
        if training_config.gradient_checkpointing:
            if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            if hasattr(model, "generation_config") and model.generation_config is not None:
                if hasattr(model.generation_config, "use_cache"):
                    model.generation_config.use_cache = False
            # Para modelos PEFT, também verifica o modelo base
            if hasattr(model, "base_model"):
                if hasattr(model.base_model, "config") and hasattr(model.base_model.config, "use_cache"):
                    model.base_model.config.use_cache = False
        
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
        
        # Garante que o modelo está em modo de treinamento
        # Isso é crítico para que os gradientes sejam calculados
        import torch
        model.train()
        
        # IMPORTANTE: Garante que os parâmetros têm requires_grad=True antes de criar o Trainer
        # Isso é especialmente crítico quando usando gradient checkpointing
        is_peft_model = hasattr(model, "peft_config") or hasattr(model, "get_peft_model")
        
        # Verifica se há parâmetros treináveis
        trainable_params_before = [name for name, param in model.named_parameters() if param.requires_grad]
        
        if len(trainable_params_before) == 0:
            self._logger.warning(
                "Nenhum parâmetro está treinável antes de criar Trainer. "
                "Tentando habilitar requires_grad..."
            )
            if is_peft_model:
                # Para modelos PEFT, habilita requires_grad para parâmetros LoRA
                for name, param in model.named_parameters():
                    if "lora" in name.lower():
                        param.requires_grad = True
                # Tenta enable_input_require_grads se disponível
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
            else:
                # Para modelos normais, habilita requires_grad para todos
                for name, param in model.named_parameters():
                    param.requires_grad = True
        
        # Verifica novamente após tentar habilitar
        trainable_params_after = [name for name, param in model.named_parameters() if param.requires_grad]
        if len(trainable_params_after) == 0:
            self._logger.error(
                "CRÍTICO: Nenhum parâmetro está treinável após tentar habilitar requires_grad!"
            )
        
        # Verifica se o modelo está usando device_map (pode causar problemas com treinamento)
        has_device_map = hasattr(model, "hf_device_map")
        
        # PROBLEMA CONHECIDO: device_map="auto" pode interferir com treinamento
        # Quando usando device_map, o modelo pode estar distribuído de forma que
        # o Trainer não consegue calcular gradientes corretamente
        # Vamos tentar remover o device_map se necessário
        if has_device_map:
            self._logger.warning(
                "Modelo carregado com device_map. "
                "Isso pode causar problemas com treinamento. "
                "Tentando consolidar modelo em um único dispositivo..."
            )
            try:
                # Tenta mover o modelo para um único dispositivo
                # Isso remove o device_map e consolida o modelo
                device = self.get_device()
                if str(device).startswith("cuda"):
                    # Move todos os parâmetros para a GPU
                    model = model.to(device)
                    # Remove o device_map se possível
                    if hasattr(model, "hf_device_map"):
                        delattr(model, "hf_device_map")
                    self._logger.info("Modelo consolidado em um único dispositivo para treinamento")
            except Exception as e:
                self._logger.warning(
                    f"Não foi possível consolidar modelo: {e}. "
                    "Continuando com device_map..."
                )
        
        # Verifica e habilita requires_grad para parâmetros treináveis
        # Quando usando LoRA/PEFT, apenas alguns parâmetros devem ter requires_grad=True
        # Mas precisamos garantir que pelo menos alguns parâmetros estejam treináveis
        trainable_params = trainable_params_after
        total_params = sum(1 for _ in model.named_parameters())
        
        # IMPORTANTE: Para modelos PEFT com gradient_checkpointing, precisamos SEMPRE chamar
        # enable_input_require_grads() para que os embeddings tenham requires_grad=True.
        # Isso é necessário porque gradient_checkpointing precisa que as entradas das camadas
        # internas tenham gradientes para funcionar corretamente.
        # Sem isso, ocorre o erro: "None of the inputs have requires_grad=True"
        if is_peft_model and training_config.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
                self._logger.info(
                    "enable_input_require_grads() chamado para compatibilidade "
                    "LoRA + gradient_checkpointing"
                )
        
        if len(trainable_params) == 0:
            # Se nenhum parâmetro está treinável, isso é um problema
            self._logger.warning(
                "Nenhum parâmetro está treinável! Tentando corrigir..."
            )
            
            if is_peft_model:
                # Modelo PEFT - os parâmetros LoRA devem estar treináveis
                # PEFT deve configurar isso automaticamente, mas vamos verificar
                try:
                    # Tenta habilitar requires_grad para parâmetros LoRA
                    for name, param in model.named_parameters():
                        if "lora" in name.lower():
                            param.requires_grad = True
                            trainable_params.append(name)
                            self._logger.info(f"Habilitando gradientes para parâmetro LoRA: {name}")
                    
                    # Se ainda não encontrou, tenta usar enable_input_require_grads
                    if len(trainable_params) == 0 and hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                        self._logger.info("Habilitando requires_grad para inputs do modelo PEFT")
                        
                        # Verifica novamente
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                trainable_params.append(name)
                except Exception as e:
                    self._logger.warning(
                        f"Erro ao configurar gradientes para modelo PEFT: {e}"
                    )
            else:
                # Modelo normal - todos os parâmetros devem estar treináveis
                for name, param in model.named_parameters():
                    param.requires_grad = True
                    trainable_params.append(name)
                self._logger.info("Habilitando gradientes para todos os parâmetros")
        
        # Verifica se há pelo menos alguns parâmetros treináveis
        if len(trainable_params) == 0:
            # Última tentativa: lista alguns parâmetros para debug
            all_param_names = [name for name, _ in list(model.named_parameters())[:20]]
            self._logger.error(
                "ERRO CRÍTICO: Nenhum parâmetro treinável!",
                total_params=total_params,
                is_peft_model=is_peft_model,
                sample_params=all_param_names[:10]
            )
            raise BackendError(
                "Nenhum parâmetro do modelo está configurado para treinamento. "
                "Verifique se o modelo foi configurado corretamente (LoRA, etc.). "
                f"Total de parâmetros: {total_params}, Modelo PEFT: {is_peft_model}",
                backend_type="huggingface",
                operation="create_trainer"
            )
        
        # Validação final antes de criar o Trainer
        try:
            sample_param = next(model.parameters())
            actual_device = sample_param.device
            self._logger.info(
                "Modelo configurado para treinamento",
                trainable_params=len(trainable_params),
                total_params=total_params,
                trainable_ratio=f"{len(trainable_params)/total_params*100:.2f}%" if total_params > 0 else "0%",
                is_peft_model=is_peft_model,
                device=str(actual_device),
                is_training=model.training,
                on_gpu=(actual_device.type == "cuda" if str(device).startswith("cuda") else False)
            )
        except Exception as e:
            self._logger.warning(f"Erro na validação final: {e}")
            self._logger.info(
                "Modelo configurado para treinamento",
                trainable_params=len(trainable_params),
                total_params=total_params,
                is_peft_model=is_peft_model
            )
        
        # Função de métricas padrão se não fornecida
        if compute_metrics is None and eval_config:
            compute_metrics = self._create_compute_metrics(eval_config.metrics)
        
        # Validação final antes de criar o Trainer
        # Garante que o modelo está pronto
        model.train()  # Garante modo de treinamento
        try:
            # Testa se consegue acessar um parâmetro
            sample_param = next(model.parameters())
            if not sample_param.requires_grad and len(trainable_params) > 0:
                # Se o sample não tem requires_grad mas há parâmetros treináveis, está OK
                # (pode ser que o sample seja um buffer)
                pass
        except Exception as e:
            self._logger.warning(f"Erro ao validar modelo antes de criar Trainer: {e}")
        
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
        self._logger.info(
            "Trainer criado com sucesso",
            trainable_params=len(trainable_params),
            gradient_checkpointing=training_config.gradient_checkpointing
        )
        
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
        
        # Log de memória antes do treinamento
        try:
            import torch
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self._logger.info(
                    "Estado de memória CUDA antes do treinamento",
                    allocated_gb=f"{memory_allocated:.2f}",
                    reserved_gb=f"{memory_reserved:.2f}",
                    total_gb=f"{memory_total:.2f}",
                    free_gb=f"{memory_total - memory_reserved:.2f}"
                )
        except Exception as e:
            self._logger.debug(f"Erro ao verificar memória: {e}")
        
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

