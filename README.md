# ModelForge

Sistema completo para fine-tuning de modelos de Machine Learning, orientado por configuração YAML, com CLI intuitivo e deploy via Docker com API compatível OpenAI.

## Características

- **Configuração via YAML**: Toda a configuração do treinamento em um único arquivo
- **Arquitetura OOP**: Código modular e extensível baseado em classes
- **CLI Intuitivo**: Comandos simples para todas as operações
- **Hugging Face Integration**: Suporte nativo a modelos e datasets do Hub
- **Multi-infraestrutura**: Execute em local, Colab, Cloud ou Containers
- **Deploy Pronto**: Exporte modelos como containers Docker com API REST
- **API OpenAI-Compatible**: Endpoints `/v1/chat/completions` e `/v1/completions`

## Instalação

### Requisitos

- Python 3.9+
- pip
- Docker (para deploy)
- GPU recomendada (CUDA ou MPS)

### Instalação via pip

```bash
# Clone o repositório
git clone https://github.com/modelforge/modelforge.git
cd modelforge

# Instale o pacote
pip install -e .

# Ou instale com dependências de desenvolvimento
pip install -e ".[dev]"
```

### Instalação das dependências

```bash
pip install -r requirements.txt
```

## Início Rápido

### 1. Criar um novo projeto

```bash
modelforge init my-classifier
cd my-classifier
```

### 2. Configurar credenciais

```bash
# Copie o exemplo de .env
cp .env.example .env

# Edite .env e adicione seu token do Hugging Face
# HF_TOKEN=hf_your_token_here
```

### 3. Editar configuração

Edite `config.yaml` com suas configurações:

```yaml
model:
  name: "bert-base-uncased"
  task: "text-classification"
  num_labels: 2

dataset:
  name: "imdb"
  splits:
    train: "train"
    validation: "test"
  columns:
    text: "text"
    label: "label"

training:
  batch_size: 16
  learning_rate: 2.0e-5
  epochs: 3
```

### 4. Validar configuração

```bash
modelforge validate config.yaml
```

### 5. Treinar modelo

```bash
modelforge train config.yaml
```

### 6. Exportar para Docker

```bash
modelforge deploy config.yaml
```

### 7. Executar API

```bash
docker run -p 8000:8000 modelforge-model:latest
```

## Comandos CLI

| Comando | Descrição |
|---------|-----------|
| `modelforge init [nome]` | Cria novo projeto |
| `modelforge validate [config]` | Valida arquivo YAML |
| `modelforge train [config]` | Executa fine-tuning |
| `modelforge status` | Exibe status/checkpoints |
| `modelforge export [config]` | Exporta modelo |
| `modelforge deploy [config]` | Gera Docker image |
| `modelforge serve [model]` | Inicia API local |

### Exemplos de uso

```bash
# Inicializar projeto com template avançado
modelforge init my-project --template advanced

# Validar com detalhes
modelforge validate config.yaml --verbose

# Treinar resumindo de checkpoint
modelforge train config.yaml --resume ./checkpoints/checkpoint-epoch-2

# Exportar como API standalone
modelforge export config.yaml --format api --output ./deploy

# Deploy com push para registry
modelforge deploy config.yaml --push

# Servir modelo localmente
modelforge serve ./checkpoints/final_model --port 8080
```

## Arquivo de Configuração YAML

### Estrutura Completa

```yaml
# Configuração do modelo
model:
  name: "bert-base-uncased"      # Nome no Hub ou caminho local
  version: "latest"               # Versão do modelo
  repository: "huggingface"       # huggingface, local, custom
  framework: "pytorch"            # pytorch, tensorflow
  task: "text-classification"     # Tarefa do modelo
  num_labels: 2                   # Classes para classificação

# Configuração do dataset
dataset:
  name: "imdb"                    # Nome no Hub ou caminho
  repository: "huggingface"
  splits:
    train: "train"
    validation: "test"
  columns:
    text: "text"                  # Coluna de texto
    label: "label"                # Coluna de labels
  preprocessing:
    max_length: 512               # Tamanho máximo de tokens
    truncation: true
    padding: "max_length"

# Parâmetros de treinamento
training:
  batch_size: 16
  learning_rate: 2.0e-5
  epochs: 3
  scheduler: "linear"             # linear, cosine, constant
  warmup_steps: 500
  weight_decay: 0.01
  gradient_accumulation_steps: 1
  fp16: true                      # Precisão mista
  seed: 42

# Avaliação
evaluation:
  metrics: ["accuracy", "f1"]
  save_strategy: "epoch"
  load_best_model_at_end: true

# Checkpoints
checkpoints:
  save_dir: "./checkpoints"
  max_to_keep: 3

# Versionamento (Hugging Face Hub)
versioning:
  hub_name: "username/model-name"
  push_to_hub: false
  private: false

# Infraestrutura
infrastructure:
  type: "local"                   # local, colab, cloud, container
  resources:
    gpu: true
    gpu_count: 1

# Credenciais (use variáveis de ambiente)
credentials:
  huggingface_token: "${HF_TOKEN}"

# Exportação
export:
  format: "docker"
  output_dir: "./output"
  api:
    port: 8000
    endpoints: ["chat/completions", "completions"]
```

## Configuração de Credenciais

### Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```bash
# Hugging Face
HF_TOKEN=hf_your_token_here
HF_USERNAME=your_username

# Weights & Biases (opcional)
WANDB_API_KEY=your_wandb_key

# Docker Registry (opcional)
DOCKER_REGISTRY=docker.io
DOCKER_USERNAME=your_username
```

### Uso no YAML

Use a sintaxe `${VAR_NAME}` para referenciar variáveis:

```yaml
credentials:
  huggingface_token: "${HF_TOKEN}"
```

### Suporte a valores default

```yaml
credentials:
  huggingface_token: "${HF_TOKEN:-default_value}"
```

## API REST (Compatível OpenAI)

Após o deploy, a API expõe os seguintes endpoints:

### Health Check

```bash
curl http://localhost:8000/health
```

### Chat Completions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "modelforge-model",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
  }'
```

**Resposta:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1703520000,
  "model": "modelforge-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'm doing well, thank you for asking!"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15,
    "total_tokens": 25
  }
}
```

### Completions

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "modelforge-model",
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### Listar Modelos

```bash
curl http://localhost:8000/v1/models
```

## Arquitetura OOP

O ModelForge é construído com arquitetura orientada a objetos:

### Classes Principais

```
modelforge/
├── config/
│   ├── schema.py          # Classes Pydantic (Config, ModelConfig, etc.)
│   ├── loader.py          # ConfigLoader
│   └── security.py        # CredentialManager (Singleton)
├── core/
│   ├── trainer.py         # ModelTrainer (Facade)
│   ├── evaluator.py       # ModelEvaluator
│   └── checkpoint.py      # CheckpointManager
├── backends/
│   ├── base.py            # BackendBase (ABC)
│   ├── huggingface.py     # HuggingFaceBackend
│   └── factory.py         # BackendFactory
├── infrastructure/
│   ├── base.py            # InfrastructureBase (ABC)
│   ├── colab.py           # ColabInfrastructure
│   ├── cloud.py           # CloudInfrastructure
│   ├── container.py       # ContainerInfrastructure
│   └── factory.py         # InfrastructureFactory
├── export/
│   ├── docker.py          # DockerBuilder (Builder)
│   ├── api.py             # ModelAPIServer, OpenAI handlers
│   └── exporter.py        # ModelExporter (Facade)
└── utils/
    ├── logging.py         # StructuredLogger
    └── exceptions.py      # Hierarquia de exceções
```

### Padrões de Design

- **Factory**: `BackendFactory`, `InfrastructureFactory`
- **Strategy**: Backends e infraestruturas intercambiáveis
- **Builder**: `DockerBuilder`
- **Facade**: `ModelTrainer`, `ModelExporter`
- **Singleton**: `CredentialManager`

### Extensibilidade

Adicione novos backends:

```python
from modelforge.backends.base import BackendBase
from modelforge.backends.factory import BackendFactory

class MyCustomBackend(BackendBase):
    def load_model(self, config):
        # Implementação
        pass
    
    # ... outros métodos

# Registre o backend
BackendFactory.register_backend("custom", MyCustomBackend)
```

## Deploy com Docker

### Build manual

```bash
# Prepare o contexto
modelforge export config.yaml --format docker --output ./deploy

# Build da imagem
cd ./deploy/docker-build
docker build -t my-model-api:latest .
```

### Docker Compose

```yaml
version: '3.8'
services:
  model-api:
    build: ./deploy/docker-build
    ports:
      - "8000:8000"
    environment:
      - API_PORT=8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
docker-compose up -d
```

### Com GPU (nvidia-docker)

```bash
docker run --gpus all -p 8000:8000 my-model-api:latest
```

## Exemplos

### Fine-tuning BERT para Classificação

```yaml
model:
  name: "bert-base-uncased"
  task: "text-classification"
  num_labels: 2

dataset:
  name: "imdb"
  columns:
    text: "text"
    label: "label"

training:
  batch_size: 16
  learning_rate: 2e-5
  epochs: 3
  fp16: true
```

### Fine-tuning GPT-2 para Geração

```yaml
model:
  name: "gpt2"
  task: "text-generation"

dataset:
  name: "wikitext"
  subset: "wikitext-2-raw-v1"
  columns:
    text: "text"

training:
  batch_size: 4
  learning_rate: 5e-5
  epochs: 2
  gradient_accumulation_steps: 8
```

### Treinamento no Google Colab

```yaml
infrastructure:
  type: "colab"
  resources:
    gpu: true

training:
  batch_size: 8  # Menor para GPUs gratuitas
  fp16: true     # Economiza memória
```

## Troubleshooting

### Erro de memória GPU

```yaml
training:
  batch_size: 8  # Reduza o batch size
  gradient_accumulation_steps: 4  # Compense com acumulação
  fp16: true  # Use precisão mista
```

### Token do Hugging Face inválido

1. Verifique se o token está correto em `.env`
2. Certifique-se que a variável está sendo carregada:
   ```bash
   echo $HF_TOKEN
   ```
3. Gere um novo token em: https://huggingface.co/settings/tokens

### Docker build falha

1. Verifique se Docker está instalado:
   ```bash
   docker --version
   ```
2. Verifique espaço em disco
3. Execute com logs:
   ```bash
   docker build --no-cache -t my-model .
   ```

## Contribuindo

1. Fork o repositório
2. Crie uma branch: `git checkout -b feature/nova-funcionalidade`
3. Commit suas mudanças: `git commit -m 'Adiciona nova funcionalidade'`
4. Push para a branch: `git push origin feature/nova-funcionalidade`
5. Abra um Pull Request

## Licença

MIT License - veja [LICENSE](LICENSE) para detalhes.

## Links

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [PyTorch](https://pytorch.org/)
- [Docker](https://docs.docker.com/)

