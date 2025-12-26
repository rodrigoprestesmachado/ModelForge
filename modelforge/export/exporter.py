"""
Classe ModelExporter para exportação de modelos.

Este módulo implementa o padrão Facade para orquestrar
o processo de exportação de modelos.
"""

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from modelforge.backends.base import BackendBase
from modelforge.config.schema import ExportConfig
from modelforge.export.docker import DockerBuilder
from modelforge.utils.exceptions import ExportError
from modelforge.utils.logging import StructuredLogger


class ModelExporter:
    """
    Orquestrador de exportação de modelos.
    
    Implementa o padrão Facade para coordenar:
    - Preparação do modelo para exportação
    - Build de imagens Docker
    - Geração de API
    - Push para registries
    
    Attributes:
        config: Configuração de exportação
        backend: Backend do modelo
        logger: Logger estruturado
    
    Example:
        >>> exporter = ModelExporter(config, backend)
        >>> output = exporter.export(model_path, output_dir)
        >>> docker_image = exporter.export_to_docker(model_path)
    """
    
    def __init__(
        self,
        config: ExportConfig,
        backend: Optional[BackendBase] = None,
        logger: Optional[StructuredLogger] = None
    ) -> None:
        """
        Inicializa o exportador.
        
        Args:
            config: Configuração de exportação
            backend: Backend do modelo
            logger: Logger estruturado
        """
        self._config = config
        self._backend = backend
        self._logger = logger or StructuredLogger("exporter")
        self._docker_builder = DockerBuilder(config, logger)
    
    @property
    def config(self) -> ExportConfig:
        """Retorna configuração."""
        return self._config
    
    def export(
        self,
        model_path: str,
        output_dir: str,
        include_tokenizer: bool = True
    ) -> str:
        """
        Exporta o modelo para um diretório.
        
        Args:
            model_path: Caminho do modelo treinado
            output_dir: Diretório de saída
            include_tokenizer: Incluir tokenizer
            
        Returns:
            Caminho do modelo exportado
        """
        self._logger.info(
            "Exportando modelo",
            source=model_path,
            destination=output_dir
        )
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_source = Path(model_path)
        
        if not model_source.exists():
            raise ExportError(
                f"Modelo não encontrado: {model_path}",
                export_format=self._config.format,
                stage="export"
            )
        
        # Copia modelo
        model_dest = output_path / "model"
        if model_dest.exists():
            shutil.rmtree(model_dest)
        
        shutil.copytree(model_source, model_dest)
        
        self._logger.info(
            "Modelo exportado com sucesso",
            path=str(model_dest)
        )
        
        return str(model_dest)
    
    def export_to_docker(
        self,
        model_path: str,
        image_name: Optional[str] = None,
        tag: str = "latest",
        push: bool = False
    ) -> str:
        """
        Exporta modelo como imagem Docker.
        
        Args:
            model_path: Caminho do modelo
            image_name: Nome da imagem
            tag: Tag da imagem
            push: Se deve fazer push para registry
            
        Returns:
            Nome completo da imagem
        """
        self._logger.info(
            "Exportando modelo para Docker",
            model_path=model_path
        )
        
        # Usa nome da config ou gera um
        image_name = image_name or self._config.image_name or "modelforge-model"
        
        # Prepara contexto de build
        output_dir = Path(self._config.output_dir) / "docker-build"
        build_context = self._docker_builder.prepare_build_context(
            model_path,
            str(output_dir)
        )
        
        # Constrói imagem
        full_image_name = self._docker_builder.build_image(
            image_name,
            tag,
            build_context
        )
        
        # Push se solicitado
        if push and self._config.registry:
            self._docker_builder.push_image(
                full_image_name,
                self._config.registry
            )
        
        self._logger.info(
            "Imagem Docker criada",
            image=full_image_name
        )
        
        return full_image_name
    
    def _prepare_model_for_export(self, model_path: str) -> str:
        """
        Prepara modelo para exportação.
        
        Args:
            model_path: Caminho do modelo
            
        Returns:
            Caminho do modelo preparado
        """
        model_source = Path(model_path)
        
        if not model_source.exists():
            raise ExportError(
                f"Modelo não encontrado: {model_path}",
                export_format=self._config.format,
                stage="prepare"
            )
        
        # Valida que tem os arquivos necessários
        required_files = ["config.json"]
        
        for filename in required_files:
            if not (model_source / filename).exists():
                self._logger.warning(
                    f"Arquivo recomendado não encontrado: {filename}"
                )
        
        return model_path
    
    def generate_api_standalone(
        self,
        model_path: str,
        output_dir: str
    ) -> str:
        """
        Gera código da API standalone (sem Docker).
        
        Args:
            model_path: Caminho do modelo
            output_dir: Diretório de saída
            
        Returns:
            Caminho do código gerado
        """
        self._logger.info(
            "Gerando API standalone",
            model_path=model_path
        )
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        api_dir = output_path / "api"
        api_dir.mkdir(exist_ok=True)
        
        # Usa o DockerBuilder para gerar código da API
        self._docker_builder._generate_api_code(api_dir)
        
        # Copia modelo
        model_dest = output_path / "model"
        if not model_dest.exists():
            shutil.copytree(model_path, model_dest)
        
        # Gera requirements
        requirements_content = self._docker_builder.REQUIREMENTS_TEMPLATE
        (output_path / "requirements.txt").write_text(requirements_content)
        
        # Gera script de execução
        run_script = '''#!/bin/bash
# Script para executar a API localmente

# Instala dependências
pip install -r requirements.txt

# Define variáveis de ambiente
export MODEL_PATH="./model"
export API_PORT=8000

# Executa API
python -m api.app
'''
        
        (output_path / "run.sh").write_text(run_script)
        
        self._logger.info(
            "API standalone gerada",
            path=str(output_path)
        )
        
        return str(output_path)
    
    def export_to_onnx(
        self,
        model_path: str,
        output_path: str
    ) -> str:
        """
        Exporta modelo para formato ONNX.
        
        Args:
            model_path: Caminho do modelo
            output_path: Caminho de saída ONNX
            
        Returns:
            Caminho do arquivo ONNX
        """
        self._logger.info(
            "Exportando para ONNX",
            model_path=model_path
        )
        
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            
            # Carrega modelo
            model = AutoModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Cria inputs dummy
            dummy_input = tokenizer(
                "Dummy input for export",
                return_tensors="pt"
            )
            
            # Exporta para ONNX
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            torch.onnx.export(
                model,
                (dummy_input["input_ids"], dummy_input["attention_mask"]),
                str(output_file),
                input_names=["input_ids", "attention_mask"],
                output_names=["output"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "output": {0: "batch", 1: "sequence"}
                },
                opset_version=14
            )
            
            self._logger.info("Modelo exportado para ONNX", path=str(output_file))
            
            return str(output_file)
            
        except Exception as e:
            raise ExportError(
                f"Falha ao exportar para ONNX: {e}",
                export_format="onnx",
                stage="export",
                original_exception=e
            )
    
    def get_export_info(self, model_path: str) -> Dict[str, Any]:
        """
        Obtém informações sobre o modelo para exportação.
        
        Args:
            model_path: Caminho do modelo
            
        Returns:
            Dict com informações do modelo
        """
        model_source = Path(model_path)
        
        info = {
            "model_path": model_path,
            "exists": model_source.exists(),
            "files": [],
            "size_mb": 0,
        }
        
        if model_source.exists():
            info["files"] = [f.name for f in model_source.iterdir()]
            info["size_mb"] = sum(
                f.stat().st_size for f in model_source.rglob("*") if f.is_file()
            ) / (1024 * 1024)
        
        return info
    
    def __repr__(self) -> str:
        """Representação string."""
        return (
            f"ModelExporter("
            f"format='{self._config.format}', "
            f"output_dir='{self._config.output_dir}')"
        )

