"""
Classe abstrata BackendBase para backends de treinamento.

Este módulo define a interface base que todos os backends devem implementar,
seguindo o padrão Strategy para permitir diferentes implementações intercambiáveis.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from modelforge.config.schema import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)


class BackendBase(ABC):
    """
    Classe abstrata base para backends de treinamento.
    
    Define a interface que todos os backends devem implementar.
    Segue o padrão Strategy, permitindo trocar backends de forma
    transparente sem alterar o código do trainer.
    
    Attributes:
        credentials: Dicionário com credenciais de autenticação
        model: Modelo carregado
        tokenizer: Tokenizer carregado
        dataset: Dataset carregado
    
    Example:
        >>> class MyBackend(BackendBase):
        ...     def load_model(self, config):
        ...         # Implementação específica
        ...         pass
    """
    
    def __init__(self, credentials: Optional[Dict[str, str]] = None) -> None:
        """
        Inicializa o backend.
        
        Args:
            credentials: Dicionário com credenciais de autenticação
        """
        self._credentials = credentials or {}
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._dataset: Optional[Any] = None
        self._trainer: Optional[Any] = None
    
    @property
    def model(self) -> Optional[Any]:
        """Retorna o modelo carregado."""
        return self._model
    
    @property
    def tokenizer(self) -> Optional[Any]:
        """Retorna o tokenizer carregado."""
        return self._tokenizer
    
    @property
    def dataset(self) -> Optional[Any]:
        """Retorna o dataset carregado."""
        return self._dataset
    
    @property
    def trainer(self) -> Optional[Any]:
        """Retorna o trainer configurado."""
        return self._trainer
    
    @abstractmethod
    def load_model(self, config: ModelConfig) -> Any:
        """
        Carrega um modelo do repositório.
        
        Args:
            config: Configuração do modelo
            
        Returns:
            Modelo carregado
            
        Raises:
            BackendError: Se o modelo não puder ser carregado
        """
        pass
    
    @abstractmethod
    def load_tokenizer(self, config: ModelConfig) -> Any:
        """
        Carrega o tokenizer do modelo.
        
        Args:
            config: Configuração do modelo
            
        Returns:
            Tokenizer carregado
            
        Raises:
            BackendError: Se o tokenizer não puder ser carregado
        """
        pass
    
    @abstractmethod
    def load_dataset(self, config: DatasetConfig) -> Any:
        """
        Carrega um dataset do repositório.
        
        Args:
            config: Configuração do dataset
            
        Returns:
            Dataset carregado
            
        Raises:
            DatasetError: Se o dataset não puder ser carregado
        """
        pass
    
    @abstractmethod
    def prepare_dataset(
        self,
        dataset: Any,
        tokenizer: Any,
        config: DatasetConfig,
        model_task: Optional[str] = None
    ) -> Any:
        """
        Prepara o dataset para treinamento.
        
        Aplica tokenização e pré-processamento.
        
        Args:
            dataset: Dataset bruto
            tokenizer: Tokenizer para processar textos
            config: Configuração do dataset
            model_task: Tarefa do modelo (text-generation, text-classification, etc.)
            
        Returns:
            Dataset processado
        """
        pass
    
    @abstractmethod
    def create_trainer(
        self,
        model: Any,
        dataset: Any,
        training_config: TrainingConfig,
        eval_dataset: Optional[Any] = None,
    ) -> Any:
        """
        Cria um trainer configurado.
        
        Args:
            model: Modelo a treinar
            dataset: Dataset de treinamento
            training_config: Configuração de treinamento
            eval_dataset: Dataset de validação (opcional)
            
        Returns:
            Trainer configurado
        """
        pass
    
    @abstractmethod
    def train(
        self,
        trainer: Any,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executa o treinamento.
        
        Args:
            trainer: Trainer configurado
            resume_from_checkpoint: Caminho para checkpoint para resumir
            
        Returns:
            Dict com métricas de treinamento
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        trainer: Any,
        dataset: Any
    ) -> Dict[str, float]:
        """
        Avalia o modelo em um dataset.
        
        Args:
            trainer: Trainer com modelo
            dataset: Dataset de avaliação
            
        Returns:
            Dict com métricas de avaliação
        """
        pass
    
    @abstractmethod
    def save_model(
        self,
        model: Any,
        tokenizer: Any,
        output_dir: str
    ) -> str:
        """
        Salva o modelo treinado.
        
        Args:
            model: Modelo a salvar
            tokenizer: Tokenizer a salvar
            output_dir: Diretório de saída
            
        Returns:
            Caminho onde o modelo foi salvo
        """
        pass
    
    @abstractmethod
    def push_to_hub(
        self,
        model: Any,
        tokenizer: Any,
        repo_name: str,
        private: bool = False
    ) -> str:
        """
        Faz push do modelo para o Hub.
        
        Args:
            model: Modelo a enviar
            tokenizer: Tokenizer a enviar
            repo_name: Nome do repositório (user/model)
            private: Se o repositório deve ser privado
            
        Returns:
            URL do modelo no Hub
        """
        pass
    
    @abstractmethod
    def get_framework(self) -> str:
        """
        Retorna o nome do framework usado.
        
        Returns:
            Nome do framework (pytorch, tensorflow, etc.)
        """
        pass
    
    @abstractmethod
    def get_device(self) -> Any:
        """
        Retorna o dispositivo de computação.
        
        Returns:
            Dispositivo (cuda, cpu, etc.)
        """
        pass
    
    def setup(self, model_config: ModelConfig, dataset_config: DatasetConfig) -> None:
        """
        Configura o backend carregando modelo, tokenizer e dataset.
        
        Args:
            model_config: Configuração do modelo
            dataset_config: Configuração do dataset
        """
        self._tokenizer = self.load_tokenizer(model_config)
        self._model = self.load_model(model_config)
        self._dataset = self.load_dataset(dataset_config)
        self._dataset = self.prepare_dataset(
            self._dataset,
            self._tokenizer,
            dataset_config
        )
    
    def cleanup(self) -> None:
        """Limpa recursos alocados pelo backend."""
        self._model = None
        self._tokenizer = None
        self._dataset = None
        self._trainer = None
    
    def __repr__(self) -> str:
        """Representação string do backend."""
        return (
            f"{self.__class__.__name__}("
            f"framework='{self.get_framework()}', "
            f"model_loaded={self._model is not None})"
        )

