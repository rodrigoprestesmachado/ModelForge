"""
Classe ModelInference para inferência com modelos treinados.

Este módulo implementa a lógica de carregamento e inferência de modelos,
permitindo testar a qualidade das respostas antes do deploy.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from modelforge.utils.exceptions import ModelForgeException
from modelforge.utils.logging import StructuredLogger


class InferenceError(ModelForgeException):
    """Erro durante inferência."""
    
    def __init__(
        self,
        message: str,
        model_path: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ) -> None:
        details = {}
        if model_path:
            details["model_path"] = model_path
        super().__init__(message, details, original_exception)
        self.model_path = model_path


class ModelInference:
    """
    Classe para inferência com modelos treinados.
    
    Encapsula a lógica de carregamento de modelo e geração de respostas,
    suportando tanto modelos de geração (causal LM) quanto classificação.
    
    Attributes:
        model_path: Caminho do modelo
        model: Modelo carregado
        tokenizer: Tokenizer carregado
        logger: Logger estruturado
    
    Example:
        >>> inference = ModelInference("./checkpoints/final_model")
        >>> response = inference.generate("Hello, how are you?", max_tokens=100)
        >>> print(response)
    """
    
    def __init__(
        self,
        model_path: str,
        logger: Optional[StructuredLogger] = None
    ) -> None:
        """
        Inicializa o objeto de inferência.
        
        Args:
            model_path: Caminho do modelo treinado
            logger: Logger estruturado (opcional)
        """
        self._model_path = model_path
        self._logger = logger or StructuredLogger("inference")
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._device: Optional[Any] = None
        self._is_generative: bool = False
    
    @property
    def model(self) -> Optional[Any]:
        """Retorna o modelo carregado."""
        return self._model
    
    @property
    def tokenizer(self) -> Optional[Any]:
        """Retorna o tokenizer carregado."""
        return self._tokenizer
    
    @property
    def is_loaded(self) -> bool:
        """Verifica se o modelo está carregado."""
        return self._model is not None and self._tokenizer is not None
    
    def load(self) -> None:
        """
        Carrega o modelo e tokenizer.
        
        Raises:
            InferenceError: Se o modelo não puder ser carregado
        """
        if self.is_loaded:
            self._logger.warning("Modelo já carregado")
            return
        
        path = Path(self._model_path)
        
        if not path.exists():
            raise InferenceError(
                f"Modelo não encontrado: {self._model_path}",
                model_path=self._model_path
            )
        
        self._logger.info("Carregando modelo", path=self._model_path)
        
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSequenceClassification,
                AutoTokenizer,
                AutoConfig,
            )
            
            # Carrega tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Detecta tipo de modelo pela config
            config = AutoConfig.from_pretrained(self._model_path)
            
            # Tenta carregar como modelo causal primeiro
            try:
                self._model = AutoModelForCausalLM.from_pretrained(self._model_path)
                self._is_generative = True
                self._logger.info("Modelo carregado como CausalLM (generativo)")
            except Exception:
                # Tenta como modelo de classificação
                try:
                    self._model = AutoModelForSequenceClassification.from_pretrained(
                        self._model_path
                    )
                    self._is_generative = False
                    self._logger.info("Modelo carregado como classificação")
                except Exception as e:
                    raise InferenceError(
                        f"Não foi possível carregar o modelo: {e}",
                        model_path=self._model_path,
                        original_exception=e
                    )
            
            # Move para dispositivo apropriado
            self._device = self._get_device()
            self._model = self._model.to(self._device)
            self._model.eval()
            
            self._logger.info(
                "Modelo carregado com sucesso",
                device=str(self._device),
                is_generative=self._is_generative
            )
            
        except InferenceError:
            raise
        except Exception as e:
            raise InferenceError(
                f"Erro ao carregar modelo: {e}",
                model_path=self._model_path,
                original_exception=e
            )
    
    def _get_device(self) -> Any:
        """
        Detecta o dispositivo de computação disponível.
        
        Returns:
            torch.device
        """
        import torch
        
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Gera uma resposta para o prompt fornecido.
        
        Args:
            prompt: Texto de entrada
            max_tokens: Número máximo de tokens a gerar
            temperature: Temperatura de sampling (0 = determinístico)
            
        Returns:
            Dict com a resposta e metadados:
                - response: Texto gerado ou classe predita
                - type: "generation" ou "classification"
                - tokens_generated: Número de tokens gerados (para geração)
                - predicted_class: Classe predita (para classificação)
                - probabilities: Probabilidades por classe (para classificação)
                
        Raises:
            InferenceError: Se a inferência falhar
        """
        if not self.is_loaded:
            self.load()
        
        self._logger.debug(
            "Gerando resposta",
            prompt_length=len(prompt),
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        try:
            import torch
            
            # Tokeniza entrada
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # Move para dispositivo
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if self._is_generative:
                    return self._generate_text(inputs, max_tokens, temperature)
                else:
                    return self._classify(inputs)
                    
        except Exception as e:
            raise InferenceError(
                f"Erro durante inferência: {e}",
                model_path=self._model_path,
                original_exception=e
            )
    
    def _generate_text(
        self,
        inputs: Dict[str, Any],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """
        Gera texto usando o modelo.
        
        Args:
            inputs: Inputs tokenizados
            max_tokens: Número máximo de tokens
            temperature: Temperatura de sampling
            
        Returns:
            Dict com resposta e metadados
        """
        import torch
        
        # Configuração de geração
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False
        
        # Gera
        outputs = self._model.generate(**inputs, **gen_kwargs)
        
        # Decodifica apenas os tokens novos
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        
        response_text = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return {
            "response": response_text.strip(),
            "type": "generation",
            "tokens_generated": len(generated_ids),
        }
    
    def _classify(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifica o texto de entrada.
        
        Args:
            inputs: Inputs tokenizados
            
        Returns:
            Dict com classe predita e probabilidades
        """
        import torch
        
        outputs = self._model(**inputs)
        logits = outputs.logits
        
        # Calcula probabilidades
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_class = torch.argmax(logits, dim=-1).item()
        
        # Tenta obter labels do modelo
        label_mapping = {}
        if hasattr(self._model.config, "id2label"):
            label_mapping = self._model.config.id2label
        
        predicted_label = label_mapping.get(predicted_class, f"Classe {predicted_class}")
        
        # Formata probabilidades
        probs_dict = {}
        for i, prob in enumerate(probabilities.tolist()):
            label = label_mapping.get(i, f"Classe {i}")
            probs_dict[label] = round(prob, 4)
        
        return {
            "response": predicted_label,
            "type": "classification",
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "confidence": round(probabilities[predicted_class].item(), 4),
            "probabilities": probs_dict,
        }
    
    def cleanup(self) -> None:
        """Limpa recursos alocados."""
        self._model = None
        self._tokenizer = None
        self._device = None
        self._logger.info("Recursos liberados")
    
    def __enter__(self) -> "ModelInference":
        """Context manager entry."""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.cleanup()
    
    def __repr__(self) -> str:
        """Representação string."""
        return (
            f"ModelInference("
            f"model_path='{self._model_path}', "
            f"is_loaded={self.is_loaded}, "
            f"is_generative={self._is_generative})"
        )

