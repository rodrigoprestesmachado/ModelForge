"""
Testes para o módulo de configuração.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from modelforge.config.loader import ConfigLoader
from modelforge.config.schema import (
    Config,
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    ColumnsConfig,
    SplitsConfig,
)
from modelforge.config.security import CredentialManager
from modelforge.utils.exceptions import ConfigValidationError


class TestModelConfig:
    """Testes para ModelConfig."""
    
    def test_valid_model_config(self):
        """Testa criação de ModelConfig válido."""
        config = ModelConfig(
            name="bert-base-uncased",
            version="latest",
            repository="huggingface",
            framework="pytorch",
        )
        
        assert config.name == "bert-base-uncased"
        assert config.version == "latest"
        assert config.repository.value == "huggingface"
        assert config.framework.value == "pytorch"
    
    def test_empty_name_raises_error(self):
        """Testa que nome vazio gera erro."""
        with pytest.raises(ValueError):
            ModelConfig(
                name="",
                version="latest",
            )
    
    def test_whitespace_name_raises_error(self):
        """Testa que nome apenas com espaços gera erro."""
        with pytest.raises(ValueError):
            ModelConfig(
                name="   ",
                version="latest",
            )


class TestDatasetConfig:
    """Testes para DatasetConfig."""
    
    def test_valid_dataset_config(self):
        """Testa criação de DatasetConfig válido."""
        config = DatasetConfig(
            name="imdb",
            repository="huggingface",
            splits=SplitsConfig(train="train", validation="test"),
            columns=ColumnsConfig(text="text", label="label"),
        )
        
        assert config.name == "imdb"
        assert config.splits.train == "train"
        assert config.columns.text == "text"


class TestTrainingConfig:
    """Testes para TrainingConfig."""
    
    def test_valid_training_config(self):
        """Testa criação de TrainingConfig válido."""
        config = TrainingConfig(
            batch_size=16,
            learning_rate=2e-5,
            epochs=3,
        )
        
        assert config.batch_size == 16
        assert config.learning_rate == 2e-5
        assert config.epochs == 3
    
    def test_fp16_and_bf16_exclusive(self):
        """Testa que fp16 e bf16 não podem ser usados juntos."""
        with pytest.raises(ValueError):
            TrainingConfig(
                batch_size=16,
                learning_rate=2e-5,
                epochs=3,
                fp16=True,
                bf16=True,
            )
    
    def test_negative_batch_size_raises_error(self):
        """Testa que batch_size negativo gera erro."""
        with pytest.raises(ValueError):
            TrainingConfig(
                batch_size=-1,
                learning_rate=2e-5,
                epochs=3,
            )


class TestConfigLoader:
    """Testes para ConfigLoader."""
    
    def test_load_valid_yaml(self):
        """Testa carregamento de YAML válido."""
        config_dict = {
            "model": {
                "name": "bert-base-uncased",
                "version": "latest",
            },
            "dataset": {
                "name": "imdb",
                "columns": {
                    "text": "text",
                    "label": "label",
                },
            },
        }
        
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_dict, f)
            f.flush()
            
            loader = ConfigLoader()
            config = loader.load(f.name)
            
            assert config.model.name == "bert-base-uncased"
            assert config.dataset.name == "imdb"
        
        os.unlink(f.name)
    
    def test_load_nonexistent_file_raises_error(self):
        """Testa que arquivo inexistente gera erro."""
        loader = ConfigLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load("/path/to/nonexistent.yaml")
    
    def test_load_invalid_yaml_raises_error(self):
        """Testa que YAML inválido gera erro."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("invalid: yaml: content: :")
            f.flush()
            
            loader = ConfigLoader()
            
            with pytest.raises(ConfigValidationError):
                loader.load(f.name)
        
        os.unlink(f.name)
    
    def test_resolve_env_variables(self):
        """Testa resolução de variáveis de ambiente."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            loader = ConfigLoader()
            
            config = {"key": "${TEST_VAR}"}
            resolved = loader._resolve_env_variables(config)
            
            assert resolved["key"] == "test_value"
    
    def test_resolve_env_with_default(self):
        """Testa resolução de variável com valor default."""
        loader = ConfigLoader()
        
        config = {"key": "${NONEXISTENT_VAR:-default}"}
        resolved = loader._resolve_env_variables(config)
        
        assert resolved["key"] == "default"
    
    def test_merge_configs(self):
        """Testa merge de configurações."""
        loader = ConfigLoader()
        
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 10}, "e": 5}
        
        merged = loader.merge_configs(base, override)
        
        assert merged["a"] == 1
        assert merged["b"]["c"] == 10
        assert merged["b"]["d"] == 3
        assert merged["e"] == 5
    
    def test_validate_valid_config(self):
        """Testa validação de config válido."""
        loader = ConfigLoader()
        
        config = {
            "model": {
                "name": "bert-base-uncased",
            },
            "dataset": {
                "name": "imdb",
                "columns": {
                    "text": "text",
                    "label": "label",
                },
            },
        }
        
        assert loader.validate(config) is True
    
    def test_validate_invalid_config_raises_error(self):
        """Testa validação de config inválido."""
        loader = ConfigLoader()
        
        config = {
            "model": {
                "name": "",  # Nome vazio é inválido
            },
        }
        
        with pytest.raises(ConfigValidationError):
            loader.validate(config)
    
    def test_get_template(self):
        """Testa obtenção de template."""
        template = ConfigLoader.get_template()
        
        assert "model" in template
        assert "dataset" in template
        assert "training" in template
        assert template["model"]["name"] == "bert-base-uncased"


class TestCredentialManager:
    """Testes para CredentialManager."""
    
    def setup_method(self):
        """Reset singleton antes de cada teste."""
        CredentialManager.reset()
    
    def test_singleton_pattern(self):
        """Testa que CredentialManager é singleton."""
        manager1 = CredentialManager()
        manager2 = CredentialManager()
        
        assert manager1 is manager2
    
    def test_get_secret_from_env(self):
        """Testa obtenção de secret de variável de ambiente."""
        with patch.dict(os.environ, {"TEST_SECRET": "secret_value"}):
            CredentialManager.reset()
            manager = CredentialManager()
            
            secret = manager.get_secret("TEST_SECRET")
            
            assert secret == "secret_value"
    
    def test_get_secret_default(self):
        """Testa valor default quando secret não existe."""
        manager = CredentialManager()
        
        secret = manager.get_secret("NONEXISTENT", default="default_value")
        
        assert secret == "default_value"
    
    def test_set_secret(self):
        """Testa definição de secret."""
        CredentialManager.reset()
        manager = CredentialManager()
        
        manager.set_secret("CUSTOM_KEY", "custom_value")
        
        assert manager.get_secret("CUSTOM_KEY") == "custom_value"
    
    def test_has_credentials_for_huggingface(self):
        """Testa verificação de credenciais para HuggingFace."""
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}):
            CredentialManager.reset()
            manager = CredentialManager()
            
            assert manager.has_credentials_for("huggingface") is True
    
    def test_clear(self):
        """Testa limpeza de credenciais."""
        CredentialManager.reset()
        manager = CredentialManager()
        manager.set_secret("TEST", "value")
        
        manager.clear()
        
        assert manager.get_secret("TEST") is None


class TestConfigIntegration:
    """Testes de integração para configuração."""
    
    def test_full_config_creation(self):
        """Testa criação de Config completo."""
        config_dict = {
            "model": {
                "name": "bert-base-uncased",
                "version": "latest",
                "repository": "huggingface",
                "framework": "pytorch",
                "task": "text-classification",
                "num_labels": 2,
            },
            "dataset": {
                "name": "imdb",
                "repository": "huggingface",
                "splits": {
                    "train": "train",
                    "validation": "test",
                },
                "columns": {
                    "text": "text",
                    "label": "label",
                },
                "preprocessing": {
                    "max_length": 512,
                },
            },
            "training": {
                "batch_size": 16,
                "learning_rate": 2e-5,
                "epochs": 3,
                "scheduler": "linear",
            },
            "evaluation": {
                "metrics": ["accuracy", "f1"],
            },
            "checkpoints": {
                "save_dir": "./checkpoints",
                "max_to_keep": 3,
            },
            "infrastructure": {
                "type": "local",
                "resources": {
                    "gpu": True,
                    "gpu_count": 1,
                },
            },
        }
        
        config = Config(**config_dict)
        
        assert config.model.name == "bert-base-uncased"
        assert config.dataset.name == "imdb"
        assert config.training.batch_size == 16
        assert config.evaluation.metrics == ["accuracy", "f1"]
        assert config.infrastructure.resources.gpu is True

