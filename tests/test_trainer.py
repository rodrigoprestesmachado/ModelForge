"""
Testes para o módulo de treinamento.
"""

from unittest.mock import MagicMock, patch
from dataclasses import asdict

import pytest

from modelforge.core.trainer import ModelTrainer, TrainingResult
from modelforge.core.evaluator import ModelEvaluator, EvaluationResult
from modelforge.core.checkpoint import CheckpointManager, CheckpointInfo
from modelforge.config.schema import (
    Config,
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    CheckpointConfig,
    ColumnsConfig,
)


class TestTrainingResult:
    """Testes para TrainingResult."""
    
    def test_default_values(self):
        """Testa valores default."""
        result = TrainingResult()
        
        assert result.metrics == {}
        assert result.best_checkpoint is None
        assert result.total_steps == 0
        assert result.epochs_completed == 0
    
    def test_to_dict(self):
        """Testa conversão para dicionário."""
        result = TrainingResult(
            metrics={"loss": 0.5},
            best_checkpoint="/path/to/checkpoint",
            total_steps=1000,
            epochs_completed=3,
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["metrics"] == {"loss": 0.5}
        assert result_dict["best_checkpoint"] == "/path/to/checkpoint"
        assert result_dict["total_steps"] == 1000


class TestModelTrainer:
    """Testes para ModelTrainer."""
    
    def create_mock_config(self):
        """Cria configuração mock."""
        return Config(
            model=ModelConfig(
                name="bert-base-uncased",
                task="text-classification",
                num_labels=2,
            ),
            dataset=DatasetConfig(
                name="imdb",
                columns=ColumnsConfig(text="text", label="label"),
            ),
            training=TrainingConfig(
                batch_size=16,
                learning_rate=2e-5,
                epochs=3,
            ),
            checkpoints=CheckpointConfig(
                save_dir="./checkpoints",
            ),
        )
    
    @patch("modelforge.core.trainer.BackendFactory")
    @patch("modelforge.core.trainer.InfrastructureFactory")
    @patch("modelforge.core.trainer.CredentialManager")
    def test_initialization(
        self, mock_cred, mock_infra_factory, mock_backend_factory
    ):
        """Testa inicialização do trainer."""
        config = self.create_mock_config()
        
        mock_cred.return_value.load_credentials.return_value = {}
        mock_backend_factory.create_backend.return_value = MagicMock()
        mock_infra_factory.create_infrastructure.return_value = MagicMock()
        
        trainer = ModelTrainer(config)
        
        assert trainer.config == config
        assert trainer.model is None
        assert not trainer._is_setup
    
    @patch("modelforge.core.trainer.BackendFactory")
    @patch("modelforge.core.trainer.InfrastructureFactory")
    @patch("modelforge.core.trainer.CredentialManager")
    def test_setup_loads_components(
        self, mock_cred, mock_infra_factory, mock_backend_factory
    ):
        """Testa que setup carrega componentes."""
        config = self.create_mock_config()
        
        mock_cred.return_value.load_credentials.return_value = {}
        
        mock_backend = MagicMock()
        mock_backend.load_tokenizer.return_value = MagicMock()
        mock_backend.load_model.return_value = MagicMock()
        mock_backend.load_dataset.return_value = {"train": [], "test": []}
        mock_backend.prepare_dataset.return_value = {"train": [], "test": []}
        mock_backend_factory.create_backend.return_value = mock_backend
        
        mock_infra = MagicMock()
        mock_infra_factory.create_infrastructure.return_value = mock_infra
        
        trainer = ModelTrainer(config)
        trainer.setup()
        
        assert trainer._is_setup
        mock_backend.load_tokenizer.assert_called_once()
        mock_backend.load_model.assert_called_once()
        mock_infra.setup.assert_called_once()
    
    @patch("modelforge.core.trainer.BackendFactory")
    @patch("modelforge.core.trainer.InfrastructureFactory")
    @patch("modelforge.core.trainer.CredentialManager")
    def test_context_manager(
        self, mock_cred, mock_infra_factory, mock_backend_factory
    ):
        """Testa uso como context manager."""
        config = self.create_mock_config()
        
        mock_cred.return_value.load_credentials.return_value = {}
        
        mock_backend = MagicMock()
        mock_backend.load_tokenizer.return_value = MagicMock()
        mock_backend.load_model.return_value = MagicMock()
        mock_backend.load_dataset.return_value = {"train": [], "test": []}
        mock_backend.prepare_dataset.return_value = {"train": [], "test": []}
        mock_backend_factory.create_backend.return_value = mock_backend
        
        mock_infra = MagicMock()
        mock_infra_factory.create_infrastructure.return_value = mock_infra
        
        with ModelTrainer(config) as trainer:
            assert trainer._is_setup
        
        # Cleanup deve ter sido chamado
        mock_backend.cleanup.assert_called()


class TestEvaluationResult:
    """Testes para EvaluationResult."""
    
    def test_default_values(self):
        """Testa valores default."""
        result = EvaluationResult()
        
        assert result.metrics == {}
        assert result.split == "eval"
        assert result.num_samples == 0
    
    def test_to_dict(self):
        """Testa conversão para dicionário."""
        result = EvaluationResult(
            metrics={"accuracy": 0.95},
            split="test",
            num_samples=1000,
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["metrics"] == {"accuracy": 0.95}
        assert result_dict["split"] == "test"
    
    def test_repr(self):
        """Testa representação string."""
        result = EvaluationResult(
            metrics={"accuracy": 0.95},
            split="test",
        )
        
        repr_str = repr(result)
        
        assert "test" in repr_str
        assert "accuracy" in repr_str


class TestModelEvaluator:
    """Testes para ModelEvaluator."""
    
    def test_initialization(self):
        """Testa inicialização do evaluator."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        evaluator = ModelEvaluator(
            mock_model,
            mock_tokenizer,
            metrics=["accuracy", "f1"]
        )
        
        assert evaluator.model == mock_model
        assert evaluator.metrics == ["accuracy", "f1"]
        assert evaluator.evaluation_history == []
    
    def test_add_custom_metric(self):
        """Testa adição de métrica customizada."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        evaluator = ModelEvaluator(mock_model, mock_tokenizer)
        
        def custom_metric(predictions, labels):
            return 0.5
        
        evaluator.add_metric("custom", custom_metric)
        
        assert "custom" in evaluator._custom_metrics
    
    def test_generate_report(self):
        """Testa geração de relatório."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        evaluator = ModelEvaluator(mock_model, mock_tokenizer)
        
        result = EvaluationResult(
            metrics={"accuracy": 0.95, "f1": 0.93},
            split="test",
            num_samples=1000,
        )
        
        report = evaluator.generate_report(result)
        
        assert "accuracy" in report
        assert "0.95" in report
        assert "test" in report
    
    def test_compare_evaluations(self):
        """Testa comparação de avaliações."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        evaluator = ModelEvaluator(mock_model, mock_tokenizer)
        
        result1 = EvaluationResult(metrics={"accuracy": 0.90})
        result2 = EvaluationResult(metrics={"accuracy": 0.95})
        
        diff = evaluator.compare_evaluations(result1, result2)
        
        assert "accuracy_diff" in diff
        assert diff["accuracy_diff"] == 0.05


class TestCheckpointInfo:
    """Testes para CheckpointInfo."""
    
    def test_creation(self):
        """Testa criação de CheckpointInfo."""
        info = CheckpointInfo(
            path="/path/to/checkpoint",
            epoch=1,
            step=1000,
            metrics={"loss": 0.5},
        )
        
        assert info.path == "/path/to/checkpoint"
        assert info.epoch == 1
        assert info.step == 1000
    
    def test_to_dict(self):
        """Testa conversão para dicionário."""
        info = CheckpointInfo(
            path="/path/to/checkpoint",
            epoch=1,
        )
        
        info_dict = info.to_dict()
        
        assert info_dict["path"] == "/path/to/checkpoint"
        assert info_dict["epoch"] == 1
    
    def test_from_dict(self):
        """Testa criação a partir de dicionário."""
        data = {
            "path": "/path/to/checkpoint",
            "epoch": 2,
            "step": 500,
            "metrics": {"loss": 0.3},
            "timestamp": "2024-01-01T00:00:00",
        }
        
        info = CheckpointInfo.from_dict(data)
        
        assert info.path == "/path/to/checkpoint"
        assert info.epoch == 2
        assert info.step == 500


class TestCheckpointManager:
    """Testes para CheckpointManager."""
    
    def test_initialization(self, tmp_path):
        """Testa inicialização do manager."""
        config = CheckpointConfig(
            save_dir=str(tmp_path / "checkpoints"),
            max_to_keep=3,
        )
        
        manager = CheckpointManager(config)
        
        assert manager.save_dir.exists()
        assert manager.checkpoints == []
    
    def test_save_checkpoint(self, tmp_path):
        """Testa salvamento de checkpoint."""
        config = CheckpointConfig(
            save_dir=str(tmp_path / "checkpoints"),
            max_to_keep=3,
        )
        
        manager = CheckpointManager(config)
        
        mock_model = MagicMock()
        mock_model.save_pretrained = MagicMock()
        
        path = manager.save(
            model=mock_model,
            epoch=1,
            metrics={"loss": 0.5},
        )
        
        assert path is not None
        assert len(manager.checkpoints) == 1
        assert manager.latest_checkpoint.epoch == 1
    
    def test_cleanup_old_checkpoints(self, tmp_path):
        """Testa limpeza de checkpoints antigos."""
        config = CheckpointConfig(
            save_dir=str(tmp_path / "checkpoints"),
            max_to_keep=2,
        )
        
        manager = CheckpointManager(config)
        
        mock_model = MagicMock()
        mock_model.save_pretrained = MagicMock()
        
        # Salva 3 checkpoints (max é 2)
        for epoch in range(1, 4):
            manager.save(model=mock_model, epoch=epoch)
        
        # Deve manter apenas 2
        assert len(manager.checkpoints) == 2
    
    def test_get_best_checkpoint(self, tmp_path):
        """Testa obtenção do melhor checkpoint."""
        config = CheckpointConfig(
            save_dir=str(tmp_path / "checkpoints"),
            max_to_keep=5,
        )
        
        manager = CheckpointManager(config)
        
        mock_model = MagicMock()
        mock_model.save_pretrained = MagicMock()
        
        # Salva checkpoints com diferentes métricas
        manager.save(model=mock_model, epoch=1, metrics={"accuracy": 0.80})
        manager.save(model=mock_model, epoch=2, metrics={"accuracy": 0.95})
        manager.save(model=mock_model, epoch=3, metrics={"accuracy": 0.85})
        
        best = manager.get_best_checkpoint("accuracy", higher_is_better=True)
        
        assert best.epoch == 2
        assert best.metrics["accuracy"] == 0.95
    
    def test_repr(self, tmp_path):
        """Testa representação string."""
        config = CheckpointConfig(
            save_dir=str(tmp_path / "checkpoints"),
            max_to_keep=3,
        )
        
        manager = CheckpointManager(config)
        
        repr_str = repr(manager)
        
        assert "CheckpointManager" in repr_str
        assert "max_to_keep=3" in repr_str

