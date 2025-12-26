"""
Testes para o módulo de backends.
"""

from unittest.mock import MagicMock, patch

import pytest

from modelforge.backends.base import BackendBase
from modelforge.backends.factory import BackendFactory
from modelforge.backends.huggingface import HuggingFaceBackend
from modelforge.config.schema import ModelConfig, DatasetConfig, ColumnsConfig
from modelforge.utils.exceptions import BackendError


class TestBackendBase:
    """Testes para BackendBase."""
    
    def test_cannot_instantiate_directly(self):
        """Testa que BackendBase não pode ser instanciado diretamente."""
        with pytest.raises(TypeError):
            BackendBase()
    
    def test_subclass_must_implement_abstract_methods(self):
        """Testa que subclasses devem implementar métodos abstratos."""
        
        class IncompleteBackend(BackendBase):
            pass
        
        with pytest.raises(TypeError):
            IncompleteBackend()
    
    def test_complete_subclass_can_be_instantiated(self):
        """Testa que subclasse completa pode ser instanciada."""
        
        class CompleteBackend(BackendBase):
            def load_model(self, config):
                return MagicMock()
            
            def load_tokenizer(self, config):
                return MagicMock()
            
            def load_dataset(self, config):
                return MagicMock()
            
            def prepare_dataset(self, dataset, tokenizer, config):
                return dataset
            
            def create_trainer(self, model, dataset, training_config, eval_dataset=None):
                return MagicMock()
            
            def train(self, trainer, resume_from_checkpoint=None):
                return {"metrics": {}}
            
            def evaluate(self, trainer, dataset):
                return {}
            
            def save_model(self, model, tokenizer, output_dir):
                return output_dir
            
            def push_to_hub(self, model, tokenizer, repo_name, private=False):
                return f"https://huggingface.co/{repo_name}"
            
            def get_framework(self):
                return "test"
            
            def get_device(self):
                return "cpu"
        
        backend = CompleteBackend()
        assert backend.get_framework() == "test"


class TestBackendFactory:
    """Testes para BackendFactory."""
    
    def test_create_huggingface_backend(self):
        """Testa criação de backend HuggingFace."""
        backend = BackendFactory.create_backend("huggingface")
        
        assert isinstance(backend, HuggingFaceBackend)
    
    def test_create_hf_alias(self):
        """Testa criação de backend usando alias 'hf'."""
        backend = BackendFactory.create_backend("hf")
        
        assert isinstance(backend, HuggingFaceBackend)
    
    def test_create_unknown_backend_raises_error(self):
        """Testa que backend desconhecido gera erro."""
        with pytest.raises(BackendError):
            BackendFactory.create_backend("unknown_backend")
    
    def test_case_insensitive(self):
        """Testa que nome do backend é case insensitive."""
        backend1 = BackendFactory.create_backend("HUGGINGFACE")
        backend2 = BackendFactory.create_backend("HuggingFace")
        
        assert isinstance(backend1, HuggingFaceBackend)
        assert isinstance(backend2, HuggingFaceBackend)
    
    def test_list_backends(self):
        """Testa listagem de backends."""
        backends = BackendFactory.list_backends()
        
        assert "huggingface" in backends
        assert "hf" in backends
    
    def test_is_backend_available(self):
        """Testa verificação de disponibilidade."""
        assert BackendFactory.is_backend_available("huggingface") is True
        assert BackendFactory.is_backend_available("unknown") is False
    
    def test_register_custom_backend(self):
        """Testa registro de backend customizado."""
        
        class CustomBackend(BackendBase):
            def load_model(self, config):
                return MagicMock()
            
            def load_tokenizer(self, config):
                return MagicMock()
            
            def load_dataset(self, config):
                return MagicMock()
            
            def prepare_dataset(self, dataset, tokenizer, config):
                return dataset
            
            def create_trainer(self, model, dataset, training_config, eval_dataset=None):
                return MagicMock()
            
            def train(self, trainer, resume_from_checkpoint=None):
                return {"metrics": {}}
            
            def evaluate(self, trainer, dataset):
                return {}
            
            def save_model(self, model, tokenizer, output_dir):
                return output_dir
            
            def push_to_hub(self, model, tokenizer, repo_name, private=False):
                return f"https://huggingface.co/{repo_name}"
            
            def get_framework(self):
                return "custom"
            
            def get_device(self):
                return "cpu"
        
        BackendFactory.register_backend("custom", CustomBackend)
        
        assert BackendFactory.is_backend_available("custom") is True
        
        backend = BackendFactory.create_backend("custom")
        assert isinstance(backend, CustomBackend)
        
        # Limpa
        BackendFactory.unregister_backend("custom")
    
    def test_register_invalid_class_raises_error(self):
        """Testa que registrar classe inválida gera erro."""
        
        class NotABackend:
            pass
        
        with pytest.raises(ValueError):
            BackendFactory.register_backend("invalid", NotABackend)


class TestHuggingFaceBackend:
    """Testes para HuggingFaceBackend."""
    
    def test_initialization(self):
        """Testa inicialização do backend."""
        backend = HuggingFaceBackend()
        
        assert backend.get_framework() == "pytorch"
        assert backend.model is None
        assert backend.tokenizer is None
    
    def test_initialization_with_credentials(self):
        """Testa inicialização com credenciais."""
        credentials = {"HF_TOKEN": "hf_test_token"}
        backend = HuggingFaceBackend(credentials=credentials)
        
        assert backend._credentials == credentials
    
    @patch("modelforge.backends.huggingface.AutoTokenizer")
    def test_load_tokenizer(self, mock_tokenizer):
        """Testa carregamento de tokenizer."""
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        backend = HuggingFaceBackend()
        config = ModelConfig(name="bert-base-uncased")
        
        tokenizer = backend.load_tokenizer(config)
        
        mock_tokenizer.from_pretrained.assert_called_once()
        assert backend.tokenizer is not None
    
    @patch("modelforge.backends.huggingface.torch")
    def test_get_device_cuda(self, mock_torch):
        """Testa detecção de dispositivo CUDA."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = "cuda"
        
        backend = HuggingFaceBackend()
        device = backend.get_device()
        
        mock_torch.device.assert_called_with("cuda")
    
    @patch("modelforge.backends.huggingface.torch")
    def test_get_device_cpu(self, mock_torch):
        """Testa fallback para CPU."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        
        backend = HuggingFaceBackend()
        device = backend.get_device()
        
        mock_torch.device.assert_called_with("cpu")
    
    def test_repr(self):
        """Testa representação string."""
        backend = HuggingFaceBackend()
        
        repr_str = repr(backend)
        
        assert "HuggingFaceBackend" in repr_str
        assert "pytorch" in repr_str


class TestBackendIntegration:
    """Testes de integração para backends (requer dependências)."""
    
    @pytest.mark.skip(reason="Requer conexão com HuggingFace Hub")
    def test_load_model_from_hub(self):
        """Testa carregamento de modelo real do Hub."""
        backend = HuggingFaceBackend()
        config = ModelConfig(
            name="bert-base-uncased",
            task="text-classification",
            num_labels=2,
        )
        
        model = backend.load_model(config)
        
        assert model is not None
    
    @pytest.mark.skip(reason="Requer conexão com HuggingFace Hub")
    def test_load_dataset_from_hub(self):
        """Testa carregamento de dataset real do Hub."""
        backend = HuggingFaceBackend()
        config = DatasetConfig(
            name="imdb",
            columns=ColumnsConfig(text="text", label="label"),
        )
        
        dataset = backend.load_dataset(config)
        
        assert dataset is not None
        assert "train" in dataset

