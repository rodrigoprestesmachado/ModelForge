"""
Testes para o CLI.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml
from click.testing import CliRunner

from modelforge.cli import cli, CLI


class TestCLI:
    """Testes para a classe CLI."""
    
    def test_cli_instance_creation(self):
        """Testa criação de instância CLI."""
        cli_instance = CLI()
        assert cli_instance is not None


class TestInitCommand:
    """Testes para o comando init."""
    
    def test_init_creates_project(self):
        """Testa que init cria projeto."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init", "test-project"])
            
            assert result.exit_code == 0
            assert Path("test-project").exists()
            assert Path("test-project/config.yaml").exists()
            assert Path("test-project/checkpoints").exists()
    
    def test_init_creates_config_file(self):
        """Testa que init cria arquivo de configuração."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init", "test-project"])
            
            assert result.exit_code == 0
            
            config_path = Path("test-project/config.yaml")
            assert config_path.exists()
            
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            assert "model" in config
            assert "dataset" in config
    
    def test_init_with_advanced_template(self):
        """Testa init com template avançado."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli, ["init", "test-project", "--template", "advanced"]
            )
            
            assert result.exit_code == 0
            
            with open("test-project/config.yaml") as f:
                config = yaml.safe_load(f)
            
            # Template avançado tem gradient_accumulation_steps
            assert config["training"]["gradient_accumulation_steps"] == 4
    
    def test_init_with_output_dir(self):
        """Testa init com diretório de saída customizado."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli, ["init", "my-project", "--output", "custom-dir"]
            )
            
            assert result.exit_code == 0
            assert Path("custom-dir/my-project").exists()


class TestValidateCommand:
    """Testes para o comando validate."""
    
    def test_validate_valid_config(self):
        """Testa validação de config válido."""
        runner = CliRunner()
        
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
        
        with runner.isolated_filesystem():
            with open("config.yaml", "w") as f:
                yaml.dump(config, f)
            
            result = runner.invoke(cli, ["validate", "config.yaml"])
            
            assert result.exit_code == 0
            assert "válida" in result.output.lower() or "valid" in result.output.lower()
    
    def test_validate_invalid_config(self):
        """Testa validação de config inválido."""
        runner = CliRunner()
        
        config = {
            "model": {
                "name": "",  # Nome vazio é inválido
            },
        }
        
        with runner.isolated_filesystem():
            with open("config.yaml", "w") as f:
                yaml.dump(config, f)
            
            result = runner.invoke(cli, ["validate", "config.yaml"])
            
            assert result.exit_code != 0
    
    def test_validate_nonexistent_file(self):
        """Testa validação de arquivo inexistente."""
        runner = CliRunner()
        
        result = runner.invoke(cli, ["validate", "nonexistent.yaml"])
        
        assert result.exit_code != 0
    
    def test_validate_verbose(self):
        """Testa validação com flag verbose."""
        runner = CliRunner()
        
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
        
        with runner.isolated_filesystem():
            with open("config.yaml", "w") as f:
                yaml.dump(config, f)
            
            result = runner.invoke(cli, ["validate", "config.yaml", "--verbose"])
            
            assert result.exit_code == 0
            # Verbose mostra mais detalhes
            assert "bert-base-uncased" in result.output


class TestTrainCommand:
    """Testes para o comando train."""
    
    def test_train_dry_run(self):
        """Testa train em modo dry-run."""
        runner = CliRunner()
        
        config = {
            "model": {
                "name": "bert-base-uncased",
                "task": "text-classification",
                "num_labels": 2,
            },
            "dataset": {
                "name": "imdb",
                "columns": {
                    "text": "text",
                    "label": "label",
                },
            },
            "training": {
                "batch_size": 16,
                "epochs": 1,
            },
        }
        
        with runner.isolated_filesystem():
            with open("config.yaml", "w") as f:
                yaml.dump(config, f)
            
            result = runner.invoke(cli, ["train", "config.yaml", "--dry-run"])
            
            assert result.exit_code == 0
            assert "dry-run" in result.output.lower()


class TestStatusCommand:
    """Testes para o comando status."""
    
    def test_status_no_checkpoints(self):
        """Testa status sem checkpoints."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["status", "--checkpoints"])
            
            assert result.exit_code == 0
    
    def test_status_with_checkpoints(self):
        """Testa status com checkpoints existentes."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Cria diretório de checkpoints
            os.makedirs("checkpoints/checkpoint-epoch-1")
            
            # Cria arquivo dummy
            with open("checkpoints/checkpoint-epoch-1/config.json", "w") as f:
                f.write("{}")
            
            result = runner.invoke(cli, ["status", "--checkpoints"])
            
            assert result.exit_code == 0
            assert "checkpoint-epoch-1" in result.output


class TestExportCommand:
    """Testes para o comando export."""
    
    def test_export_missing_model(self):
        """Testa export sem modelo treinado."""
        runner = CliRunner()
        
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
            "checkpoints": {
                "save_dir": "./checkpoints",
            },
        }
        
        with runner.isolated_filesystem():
            with open("config.yaml", "w") as f:
                yaml.dump(config, f)
            
            result = runner.invoke(cli, ["export", "config.yaml"])
            
            # Deve falhar porque não há modelo treinado
            assert result.exit_code != 0


class TestServeCommand:
    """Testes para o comando serve."""
    
    def test_serve_nonexistent_model(self):
        """Testa serve com modelo inexistente."""
        runner = CliRunner()
        
        result = runner.invoke(cli, ["serve", "/nonexistent/model"])
        
        # Deve falhar porque o caminho não existe
        assert result.exit_code != 0


class TestCLIHelp:
    """Testes para mensagens de ajuda."""
    
    def test_main_help(self):
        """Testa ajuda principal."""
        runner = CliRunner()
        
        result = runner.invoke(cli, ["--help"])
        
        assert result.exit_code == 0
        assert "ModelForge" in result.output
        assert "init" in result.output
        assert "train" in result.output
        assert "validate" in result.output
    
    def test_init_help(self):
        """Testa ajuda do comando init."""
        runner = CliRunner()
        
        result = runner.invoke(cli, ["init", "--help"])
        
        assert result.exit_code == 0
        assert "project" in result.output.lower()
    
    def test_train_help(self):
        """Testa ajuda do comando train."""
        runner = CliRunner()
        
        result = runner.invoke(cli, ["train", "--help"])
        
        assert result.exit_code == 0
        assert "config" in result.output.lower()
    
    def test_version(self):
        """Testa exibição de versão."""
        runner = CliRunner()
        
        result = runner.invoke(cli, ["--version"])
        
        assert result.exit_code == 0
        assert "modelforge" in result.output.lower()

