"""
Classes Pydantic para validação e tipagem de configurações YAML.

Este módulo define todas as classes de configuração usando Pydantic,
garantindo validação automática e type hints completos.
"""

from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


class FrameworkType(str, Enum):
    """Tipos de frameworks suportados."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


class RepositoryType(str, Enum):
    """Tipos de repositórios suportados."""
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    CUSTOM = "custom"


class InfrastructureType(str, Enum):
    """Tipos de infraestrutura suportados."""
    LOCAL = "local"
    COLAB = "colab"
    CLOUD = "cloud"
    CONTAINER = "container"


class SchedulerType(str, Enum):
    """Tipos de scheduler de learning rate."""
    LINEAR = "linear"
    COSINE = "cosine"
    CONSTANT = "constant"
    POLYNOMIAL = "polynomial"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"


class SaveStrategyType(str, Enum):
    """Estratégias de salvamento de checkpoints."""
    EPOCH = "epoch"
    STEPS = "steps"
    NO = "no"


class ModelConfig(BaseModel):
    """
    Configuração do modelo base para fine-tuning.
    
    Attributes:
        name: Nome do modelo (ex: 'bert-base-uncased')
        version: Versão do modelo (ex: 'latest', 'v1.0')
        repository: Repositório de origem do modelo
        type: Tipo do modelo (ex: 'transformer', 'cnn')
        framework: Framework de deep learning
        task: Tarefa do modelo (ex: 'text-classification')
        num_labels: Número de classes para classificação
        revision: Revisão específica do modelo no Hub
    """
    name: str = Field(..., description="Nome do modelo no repositório")
    version: str = Field(default="latest", description="Versão do modelo")
    repository: RepositoryType = Field(
        default=RepositoryType.HUGGINGFACE,
        description="Repositório de origem"
    )
    type: str = Field(default="transformer", description="Tipo do modelo")
    framework: FrameworkType = Field(
        default=FrameworkType.PYTORCH,
        description="Framework de deep learning"
    )
    task: Optional[str] = Field(
        default=None,
        description="Tarefa do modelo (text-classification, etc.)"
    )
    num_labels: Optional[int] = Field(
        default=None,
        description="Número de labels para classificação"
    )
    revision: Optional[str] = Field(
        default=None,
        description="Revisão específica do modelo"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Valida que o nome do modelo não está vazio."""
        if not v or not v.strip():
            raise ValueError("Nome do modelo não pode ser vazio")
        return v.strip()


class SplitsConfig(BaseModel):
    """Configuração de splits do dataset."""
    train: str = Field(default="train", description="Nome do split de treino")
    validation: Optional[str] = Field(
        default=None,
        description="Nome do split de validação"
    )
    test: Optional[str] = Field(
        default=None,
        description="Nome do split de teste"
    )


class ColumnsConfig(BaseModel):
    """Configuração de colunas do dataset."""
    text: str = Field(..., description="Coluna de texto/entrada")
    label: Optional[str] = Field(
        default=None,
        description="Coluna de labels"
    )
    text_pair: Optional[str] = Field(
        default=None,
        description="Segunda coluna de texto para tarefas de pares"
    )


class PreprocessingConfig(BaseModel):
    """Configuração de pré-processamento do dataset."""
    max_length: int = Field(
        default=512,
        ge=1,
        le=8192,
        description="Comprimento máximo de tokens"
    )
    truncation: bool = Field(default=True, description="Truncar sequências longas")
    padding: str = Field(default="max_length", description="Estratégia de padding")
    return_tensors: str = Field(default="pt", description="Formato dos tensors")


class DatasetConfig(BaseModel):
    """
    Configuração do dataset para treinamento.
    
    Attributes:
        name: Nome do dataset
        repository: Repositório de origem
        splits: Configuração de splits
        columns: Mapeamento de colunas
        preprocessing: Configurações de pré-processamento
        subset: Subset específico do dataset
        streaming: Usar streaming para datasets grandes
    """
    name: str = Field(..., description="Nome do dataset")
    repository: RepositoryType = Field(
        default=RepositoryType.HUGGINGFACE,
        description="Repositório de origem"
    )
    splits: SplitsConfig = Field(
        default_factory=SplitsConfig,
        description="Configuração de splits"
    )
    columns: ColumnsConfig = Field(..., description="Mapeamento de colunas")
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="Configurações de pré-processamento"
    )
    subset: Optional[str] = Field(
        default=None,
        description="Subset específico do dataset"
    )
    streaming: bool = Field(
        default=False,
        description="Usar streaming para datasets grandes"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Valida que o nome do dataset não está vazio."""
        if not v or not v.strip():
            raise ValueError("Nome do dataset não pode ser vazio")
        return v.strip()


class TrainingConfig(BaseModel):
    """
    Configuração de parâmetros de treinamento.
    
    Attributes:
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado
        epochs: Número de épocas
        scheduler: Tipo de scheduler de LR
        warmup_steps: Passos de warmup
        weight_decay: Weight decay para regularização
        gradient_accumulation_steps: Passos de acumulação de gradiente
        fp16: Usar precisão mista FP16
        bf16: Usar precisão mista BF16
        max_grad_norm: Norma máxima de gradiente
        seed: Seed para reprodutibilidade
    """
    batch_size: int = Field(default=16, ge=1, description="Tamanho do batch")
    learning_rate: float = Field(
        default=2e-5,
        gt=0,
        description="Taxa de aprendizado"
    )
    epochs: int = Field(default=3, ge=1, description="Número de épocas")
    scheduler: SchedulerType = Field(
        default=SchedulerType.LINEAR,
        description="Tipo de scheduler"
    )
    warmup_steps: int = Field(default=0, ge=0, description="Passos de warmup")
    warmup_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ratio de warmup"
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        description="Weight decay"
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        description="Passos de acumulação"
    )
    fp16: bool = Field(default=False, description="Usar FP16")
    bf16: bool = Field(default=False, description="Usar BF16")
    max_grad_norm: float = Field(
        default=1.0,
        gt=0,
        description="Norma máxima de gradiente"
    )
    seed: int = Field(default=42, description="Seed para reprodutibilidade")
    
    @model_validator(mode="after")
    def validate_precision(self) -> "TrainingConfig":
        """Valida que apenas um tipo de precisão mista está ativo."""
        if self.fp16 and self.bf16:
            raise ValueError("Não é possível usar FP16 e BF16 simultaneamente")
        return self


class EvaluationConfig(BaseModel):
    """
    Configuração de avaliação do modelo.
    
    Attributes:
        metrics: Lista de métricas para avaliar
        save_strategy: Estratégia de salvamento
        eval_steps: Passos entre avaliações
        load_best_model_at_end: Carregar melhor modelo ao final
        metric_for_best_model: Métrica para selecionar melhor modelo
        greater_is_better: Se maior valor é melhor para a métrica
    """
    metrics: List[str] = Field(
        default=["accuracy"],
        description="Métricas de avaliação"
    )
    save_strategy: SaveStrategyType = Field(
        default=SaveStrategyType.EPOCH,
        description="Estratégia de salvamento"
    )
    eval_steps: Optional[int] = Field(
        default=None,
        ge=1,
        description="Passos entre avaliações"
    )
    load_best_model_at_end: bool = Field(
        default=True,
        description="Carregar melhor modelo ao final"
    )
    metric_for_best_model: str = Field(
        default="eval_loss",
        description="Métrica para melhor modelo"
    )
    greater_is_better: bool = Field(
        default=False,
        description="Maior valor é melhor"
    )


class CheckpointConfig(BaseModel):
    """
    Configuração de checkpoints.
    
    Attributes:
        save_dir: Diretório para salvar checkpoints
        save_steps: Passos entre salvamentos
        max_to_keep: Número máximo de checkpoints a manter
        save_total_limit: Limite total de checkpoints
    """
    save_dir: str = Field(
        default="./checkpoints",
        description="Diretório de checkpoints"
    )
    save_steps: Optional[int] = Field(
        default=None,
        ge=1,
        description="Passos entre salvamentos"
    )
    max_to_keep: int = Field(
        default=3,
        ge=1,
        description="Máximo de checkpoints a manter"
    )
    save_total_limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Limite total de checkpoints"
    )


class VersioningConfig(BaseModel):
    """
    Configuração de versionamento e upload ao Hub.
    
    Attributes:
        hub_name: Nome do modelo no Hub (username/model-name)
        push_to_hub: Fazer push ao Hugging Face Hub
        private: Modelo privado no Hub
        hub_strategy: Estratégia de push ao Hub
    """
    hub_name: Optional[str] = Field(
        default=None,
        description="Nome no Hub (username/model-name)"
    )
    push_to_hub: bool = Field(
        default=False,
        description="Fazer push ao Hub"
    )
    private: bool = Field(
        default=False,
        description="Modelo privado"
    )
    hub_strategy: str = Field(
        default="every_save",
        description="Estratégia de push"
    )


class ResourcesConfig(BaseModel):
    """Configuração de recursos computacionais."""
    gpu: bool = Field(default=True, description="Usar GPU")
    gpu_count: int = Field(default=1, ge=0, description="Número de GPUs")
    cpu_count: Optional[int] = Field(default=None, description="Número de CPUs")
    memory_gb: Optional[int] = Field(default=None, description="Memória em GB")


class InfrastructureConfig(BaseModel):
    """
    Configuração de infraestrutura de execução.
    
    Attributes:
        type: Tipo de infraestrutura
        resources: Configuração de recursos
        provider: Provedor cloud (aws, gcp, azure)
    """
    type: InfrastructureType = Field(
        default=InfrastructureType.LOCAL,
        description="Tipo de infraestrutura"
    )
    resources: ResourcesConfig = Field(
        default_factory=ResourcesConfig,
        description="Configuração de recursos"
    )
    provider: Optional[str] = Field(
        default=None,
        description="Provedor cloud"
    )


class CredentialsConfig(BaseModel):
    """
    Configuração de credenciais (suporta variáveis de ambiente).
    
    Attributes:
        huggingface_token: Token do Hugging Face Hub
        wandb_api_key: Chave da API do W&B
        additional: Credenciais adicionais
    """
    huggingface_token: Optional[str] = Field(
        default=None,
        description="Token do HF Hub (use ${HF_TOKEN})"
    )
    wandb_api_key: Optional[str] = Field(
        default=None,
        description="Chave da API do W&B"
    )
    additional: Dict[str, str] = Field(
        default_factory=dict,
        description="Credenciais adicionais"
    )


class APIConfig(BaseModel):
    """Configuração da API de inferência."""
    framework: str = Field(default="flask", description="Framework da API")
    endpoints: List[str] = Field(
        default=["chat/completions", "completions"],
        description="Endpoints a expor"
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Porta da API"
    )
    host: str = Field(default="0.0.0.0", description="Host da API")


class ExportConfig(BaseModel):
    """
    Configuração de exportação do modelo.
    
    Attributes:
        format: Formato de exportação (docker, onnx, etc.)
        output_dir: Diretório de saída
        api: Configuração da API
        image_name: Nome da imagem Docker
        image_tag: Tag da imagem Docker
        registry: Registry para push da imagem
    """
    format: str = Field(default="docker", description="Formato de exportação")
    output_dir: str = Field(
        default="./output",
        description="Diretório de saída"
    )
    api: APIConfig = Field(
        default_factory=APIConfig,
        description="Configuração da API"
    )
    image_name: Optional[str] = Field(
        default=None,
        description="Nome da imagem Docker"
    )
    image_tag: str = Field(default="latest", description="Tag da imagem")
    registry: Optional[str] = Field(
        default=None,
        description="Registry para push"
    )


class LoggingConfig(BaseModel):
    """Configuração de logging."""
    level: str = Field(default="INFO", description="Nível de log")
    format: str = Field(default="json", description="Formato de log")
    file: Optional[str] = Field(default=None, description="Arquivo de log")


class Config(BaseModel):
    """
    Configuração principal do ModelForge.
    
    Agrega todas as configurações do sistema em uma única classe.
    Esta é a classe raiz que representa um arquivo YAML completo.
    """
    model: ModelConfig = Field(..., description="Configuração do modelo")
    dataset: DatasetConfig = Field(..., description="Configuração do dataset")
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Configuração de treinamento"
    )
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
        description="Configuração de avaliação"
    )
    checkpoints: CheckpointConfig = Field(
        default_factory=CheckpointConfig,
        description="Configuração de checkpoints"
    )
    versioning: VersioningConfig = Field(
        default_factory=VersioningConfig,
        description="Configuração de versionamento"
    )
    infrastructure: InfrastructureConfig = Field(
        default_factory=InfrastructureConfig,
        description="Configuração de infraestrutura"
    )
    credentials: CredentialsConfig = Field(
        default_factory=CredentialsConfig,
        description="Configuração de credenciais"
    )
    export: ExportConfig = Field(
        default_factory=ExportConfig,
        description="Configuração de exportação"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Configuração de logging"
    )

    class Config:
        """Configuração do modelo Pydantic."""
        extra = "allow"  # Permite campos adicionais
        validate_assignment = True  # Valida em atribuições

