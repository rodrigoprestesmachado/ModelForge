"""
Configuração de fixtures para testes pytest.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml


@pytest.fixture
def sample_config_dict():
    """Retorna um dicionário de configuração válido para testes."""
    return {
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


@pytest.fixture
def sample_config_file(sample_config_dict, tmp_path):
    """Cria um arquivo de configuração YAML temporário."""
    config_path = tmp_path / "config.yaml"
    
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)
    
    return config_path


@pytest.fixture
def mock_model():
    """Retorna um mock de modelo PyTorch."""
    model = MagicMock()
    model.parameters.return_value = iter([MagicMock()])
    model.save_pretrained = MagicMock()
    model.to = MagicMock(return_value=model)
    return model


@pytest.fixture
def mock_tokenizer():
    """Retorna um mock de tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "[EOS]"
    tokenizer.save_pretrained = MagicMock()
    return tokenizer


@pytest.fixture
def mock_dataset():
    """Retorna um mock de dataset."""
    dataset = MagicMock()
    dataset.__getitem__ = MagicMock(return_value=MagicMock())
    dataset.keys.return_value = ["train", "test"]
    return dataset


@pytest.fixture
def checkpoint_dir(tmp_path):
    """Cria um diretório de checkpoints temporário."""
    checkpoint_path = tmp_path / "checkpoints"
    checkpoint_path.mkdir()
    return checkpoint_path


@pytest.fixture
def output_dir(tmp_path):
    """Cria um diretório de saída temporário."""
    output_path = tmp_path / "output"
    output_path.mkdir()
    return output_path


@pytest.fixture(autouse=True)
def reset_credential_manager():
    """Reset do CredentialManager antes de cada teste."""
    from modelforge.config.security import CredentialManager
    CredentialManager.reset()
    yield
    CredentialManager.reset()


@pytest.fixture
def env_with_hf_token(monkeypatch):
    """Configura variável de ambiente HF_TOKEN."""
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_12345")


@pytest.fixture
def mock_torch():
    """Mock do módulo torch."""
    mock = MagicMock()
    mock.cuda.is_available.return_value = False
    mock.backends.mps.is_available.return_value = False
    mock.device.return_value = MagicMock()
    return mock

