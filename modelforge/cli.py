"""
CLI do ModelForge para operações de fine-tuning.

Este módulo implementa a interface de linha de comando usando click,
fornecendo comandos intuitivos para todas as operações do sistema.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from modelforge import __version__


# Console Rich para output formatado
console = Console()


class CLI:
    """
    Classe principal do CLI.
    
    Gerencia todos os comandos e operações do ModelForge.
    """
    
    def __init__(self):
        """Inicializa o CLI."""
        self._console = console
    
    def print_banner(self) -> None:
        """Exibe banner do ModelForge."""
        banner = f"""
[bold cyan]╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   [bold white]ModelForge[/bold white] - Fine-tuning de Modelos ML            ║
║   Versão: {__version__:<44}║
║                                                       ║
╚═══════════════════════════════════════════════════════╝[/bold cyan]
"""
        self._console.print(banner)


# Instância global do CLI
cli_instance = CLI()


@click.group()
@click.version_option(version=__version__, prog_name="modelforge")
def cli():
    """
    ModelForge - Sistema de Fine-tuning de Modelos ML
    
    Execute modelforge COMANDO --help para mais informações sobre cada comando.
    """
    pass


@cli.command()
@click.argument("project_name", default="my-project")
@click.option(
    "--template", "-t",
    type=click.Choice(["basic", "advanced"]),
    default="basic",
    help="Template de configuração a usar"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=".",
    help="Diretório de saída"
)
def init(project_name: str, template: str, output: str):
    """
    Inicializa um novo projeto ModelForge.
    
    Cria a estrutura de diretórios e arquivo de configuração YAML.
    
    Exemplo:
        modelforge init my-classifier
        modelforge init my-project --template advanced
    """
    from modelforge.config.loader import ConfigLoader
    import yaml
    
    cli_instance.print_banner()
    
    output_path = Path(output) / project_name
    
    with console.status(f"[bold green]Criando projeto {project_name}..."):
        try:
            # Cria diretório
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Cria subdiretórios
            (output_path / "checkpoints").mkdir(exist_ok=True)
            (output_path / "output").mkdir(exist_ok=True)
            (output_path / "logs").mkdir(exist_ok=True)
            
            # Gera configuração
            config_template = ConfigLoader.get_template()
            
            if template == "advanced":
                # Adiciona configurações avançadas
                config_template["training"]["gradient_accumulation_steps"] = 4
                config_template["training"]["fp16"] = True
                config_template["evaluation"]["metrics"].extend(["precision", "recall"])
                config_template["versioning"]["push_to_hub"] = True
            
            # Salva config.yaml
            config_path = output_path / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_template, f, default_flow_style=False, allow_unicode=True)
            
            # Cria .env.example
            env_example = """# Variáveis de ambiente para o projeto
HF_TOKEN=hf_your_token_here
"""
            (output_path / ".env.example").write_text(env_example)
            
            # Cria .gitignore
            gitignore = """# ModelForge
checkpoints/
output/
logs/
.env
*.pt
*.pth
*.bin
__pycache__/
"""
            (output_path / ".gitignore").write_text(gitignore)
            
            console.print(f"\n[bold green]✓[/bold green] Projeto '{project_name}' criado com sucesso!")
            console.print(f"\n[dim]Estrutura criada:[/dim]")
            console.print(f"  {project_name}/")
            console.print(f"  ├── config.yaml")
            console.print(f"  ├── .env.example")
            console.print(f"  ├── .gitignore")
            console.print(f"  ├── checkpoints/")
            console.print(f"  ├── output/")
            console.print(f"  └── logs/")
            
            console.print(f"\n[bold]Próximos passos:[/bold]")
            console.print(f"  1. cd {project_name}")
            console.print(f"  2. Edite config.yaml com suas configurações")
            console.print(f"  3. Copie .env.example para .env e adicione suas credenciais")
            console.print(f"  4. Execute: modelforge validate config.yaml")
            console.print(f"  5. Execute: modelforge train config.yaml")
            
        except Exception as e:
            console.print(f"[bold red]✗ Erro ao criar projeto:[/bold red] {e}")
            sys.exit(1)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Exibe detalhes da validação")
def validate(config_path: str, verbose: bool):
    """
    Valida um arquivo de configuração YAML.
    
    Verifica se todas as configurações estão corretas
    e se as credenciais são válidas.
    
    Exemplo:
        modelforge validate config.yaml
        modelforge validate config.yaml --verbose
    """
    from modelforge.config.loader import ConfigLoader
    from modelforge.utils.exceptions import ConfigValidationError
    
    cli_instance.print_banner()
    console.print(f"\n[bold]Validando:[/bold] {config_path}\n")
    
    loader = ConfigLoader()
    
    try:
        with console.status("[bold green]Validando configuração..."):
            config = loader.load(config_path)
        
        console.print("[bold green]✓[/bold green] Configuração válida!\n")
        
        if verbose:
            # Exibe detalhes da configuração
            table = Table(title="Configuração Carregada")
            table.add_column("Seção", style="cyan")
            table.add_column("Valor", style="white")
            
            table.add_row("Modelo", config.model.name)
            table.add_row("Dataset", config.dataset.name)
            table.add_row("Epochs", str(config.training.epochs))
            table.add_row("Batch Size", str(config.training.batch_size))
            table.add_row("Learning Rate", str(config.training.learning_rate))
            table.add_row("Infraestrutura", config.infrastructure.type.value)
            table.add_row("GPU", str(config.infrastructure.resources.gpu))
            
            console.print(table)
        
    except ConfigValidationError as e:
        console.print(f"[bold red]✗ Configuração inválida:[/bold red]\n")
        console.print(f"  {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]✗ Erro ao validar:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--resume", "-r", type=click.Path(), help="Resumir de checkpoint")
@click.option("--dry-run", is_flag=True, help="Simula treinamento sem executar")
def train(config_path: str, resume: Optional[str], dry_run: bool):
    """
    Executa o fine-tuning do modelo.
    
    Carrega a configuração e inicia o processo de treinamento.
    
    Exemplo:
        modelforge train config.yaml
        modelforge train config.yaml --resume checkpoint-epoch-2
    """
    from modelforge.config.loader import ConfigLoader
    from modelforge.core.trainer import ModelTrainer
    
    cli_instance.print_banner()
    console.print(f"\n[bold]Iniciando treinamento com:[/bold] {config_path}\n")
    
    try:
        # Carrega configuração
        loader = ConfigLoader()
        config = loader.load(config_path)
        
        console.print(f"[dim]Modelo:[/dim] {config.model.name}")
        console.print(f"[dim]Dataset:[/dim] {config.dataset.name}")
        console.print(f"[dim]Epochs:[/dim] {config.training.epochs}")
        console.print(f"[dim]Batch Size:[/dim] {config.training.batch_size}")
        console.print()
        
        if dry_run:
            console.print("[yellow]Modo dry-run: treinamento não será executado[/yellow]")
            console.print("[bold green]✓[/bold green] Configuração válida para treinamento")
            return
        
        # Cria trainer
        trainer = ModelTrainer(config)
        
        # Executa treinamento
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Configurando ambiente...", total=None)
            trainer.setup()
            
            progress.update(task, description="[cyan]Treinando modelo...")
            result = trainer.train(resume_from_checkpoint=resume)
        
        console.print("\n[bold green]✓ Treinamento concluído![/bold green]\n")
        
        # Exibe resultados
        table = Table(title="Resultados do Treinamento")
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", style="white")
        
        for metric, value in result.metrics.items():
            table.add_row(metric, f"{value:.4f}" if isinstance(value, float) else str(value))
        
        table.add_row("Checkpoint", result.best_checkpoint or "N/A")
        table.add_row("Steps Totais", str(result.total_steps))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Erro durante treinamento:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("run_id", required=False)
@click.option("--checkpoints", "-c", is_flag=True, help="Lista checkpoints")
def status(run_id: Optional[str], checkpoints: bool):
    """
    Exibe o status do treinamento.
    
    Mostra informações sobre treinamentos em andamento ou concluídos.
    
    Exemplo:
        modelforge status
        modelforge status --checkpoints
    """
    cli_instance.print_banner()
    
    if checkpoints:
        # Lista checkpoints
        console.print("\n[bold]Checkpoints disponíveis:[/bold]\n")
        
        checkpoint_dir = Path("./checkpoints")
        if checkpoint_dir.exists():
            checkpoints_list = list(checkpoint_dir.iterdir())
            if checkpoints_list:
                table = Table()
                table.add_column("Checkpoint", style="cyan")
                table.add_column("Tamanho", style="white")
                table.add_column("Data", style="dim")
                
                for cp in sorted(checkpoints_list):
                    if cp.is_dir():
                        size = sum(f.stat().st_size for f in cp.rglob("*") if f.is_file())
                        size_mb = size / (1024 * 1024)
                        mtime = cp.stat().st_mtime
                        from datetime import datetime
                        date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                        table.add_row(cp.name, f"{size_mb:.1f} MB", date)
                
                console.print(table)
            else:
                console.print("[dim]Nenhum checkpoint encontrado[/dim]")
        else:
            console.print("[dim]Diretório de checkpoints não existe[/dim]")
    else:
        console.print("\n[dim]Use --checkpoints para listar checkpoints disponíveis[/dim]")


@cli.command(name="export")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="./output", help="Diretório de saída")
@click.option("--model-path", "-m", type=click.Path(), help="Caminho do modelo treinado")
@click.option("--format", "-f", type=click.Choice(["docker", "api", "onnx"]), default="docker")
def export_cmd(config_path: str, output: str, model_path: Optional[str], format: str):
    """
    Exporta o modelo treinado.
    
    Gera artefatos de deploy como imagens Docker ou código de API.
    
    Exemplo:
        modelforge export config.yaml --output ./deploy
        modelforge export config.yaml --format docker --model-path ./checkpoints/final
    """
    from modelforge.config.loader import ConfigLoader
    from modelforge.export.exporter import ModelExporter
    
    cli_instance.print_banner()
    console.print(f"\n[bold]Exportando modelo...[/bold]\n")
    
    try:
        # Carrega configuração
        loader = ConfigLoader()
        config = loader.load(config_path)
        
        # Determina caminho do modelo
        if model_path is None:
            model_path = str(Path(config.checkpoints.save_dir) / "final_model")
        
        if not Path(model_path).exists():
            console.print(f"[bold red]✗ Modelo não encontrado:[/bold red] {model_path}")
            console.print("[dim]Execute o treinamento primeiro ou especifique --model-path[/dim]")
            sys.exit(1)
        
        # Cria exportador
        exporter = ModelExporter(config.export)
        
        with console.status(f"[bold green]Exportando para {format}..."):
            if format == "docker":
                result = exporter.export_to_docker(model_path)
                console.print(f"\n[bold green]✓[/bold green] Imagem Docker criada: {result}")
                console.print(f"\n[bold]Para executar:[/bold]")
                console.print(f"  docker run -p 8000:8000 {result}")
                
            elif format == "api":
                result = exporter.generate_api_standalone(model_path, output)
                console.print(f"\n[bold green]✓[/bold green] API gerada em: {result}")
                console.print(f"\n[bold]Para executar:[/bold]")
                console.print(f"  cd {result}")
                console.print(f"  pip install -r requirements.txt")
                console.print(f"  python -m api.app")
                
            elif format == "onnx":
                result = exporter.export_to_onnx(
                    model_path,
                    str(Path(output) / "model.onnx")
                )
                console.print(f"\n[bold green]✓[/bold green] Modelo ONNX exportado: {result}")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Erro ao exportar:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--model-path", "-m", type=click.Path(), help="Caminho do modelo")
@click.option("--build-only", is_flag=True, help="Apenas constrói a imagem")
@click.option("--push", is_flag=True, help="Faz push para registry")
def deploy(config_path: str, model_path: Optional[str], build_only: bool, push: bool):
    """
    Gera imagem Docker e faz deploy do modelo.
    
    Constrói uma imagem Docker com API REST compatível com OpenAI.
    
    Exemplo:
        modelforge deploy config.yaml
        modelforge deploy config.yaml --push
    """
    from modelforge.config.loader import ConfigLoader
    from modelforge.export.exporter import ModelExporter
    
    cli_instance.print_banner()
    console.print(f"\n[bold]Iniciando deploy...[/bold]\n")
    
    try:
        # Carrega configuração
        loader = ConfigLoader()
        config = loader.load(config_path)
        
        # Determina caminho do modelo
        if model_path is None:
            model_path = str(Path(config.checkpoints.save_dir) / "final_model")
        
        if not Path(model_path).exists():
            console.print(f"[bold red]✗ Modelo não encontrado:[/bold red] {model_path}")
            sys.exit(1)
        
        # Cria exportador
        exporter = ModelExporter(config.export)
        
        with console.status("[bold green]Construindo imagem Docker..."):
            image_name = exporter.export_to_docker(
                model_path,
                push=push
            )
        
        console.print(f"\n[bold green]✓[/bold green] Imagem criada: {image_name}")
        
        if not build_only:
            console.print(f"\n[bold]Para executar localmente:[/bold]")
            console.print(f"  docker run -p {config.export.api.port}:{config.export.api.port} {image_name}")
            
            console.print(f"\n[bold]Endpoints disponíveis:[/bold]")
            console.print(f"  POST http://localhost:{config.export.api.port}/v1/chat/completions")
            console.print(f"  POST http://localhost:{config.export.api.port}/v1/completions")
            console.print(f"  GET  http://localhost:{config.export.api.port}/v1/models")
            console.print(f"  GET  http://localhost:{config.export.api.port}/health")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Erro no deploy:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--port", "-p", type=int, default=8000, help="Porta da API")
@click.option("--host", "-h", type=str, default="0.0.0.0", help="Host da API")
def serve(model_path: str, port: int, host: str):
    """
    Inicia servidor de API para o modelo.
    
    Executa uma API REST localmente para testes.
    
    Exemplo:
        modelforge serve ./checkpoints/final_model
        modelforge serve ./model --port 8080
    """
    from modelforge.config.schema import ExportConfig, APIConfig
    from modelforge.export.api import ModelAPIServer
    
    cli_instance.print_banner()
    console.print(f"\n[bold]Iniciando servidor de API...[/bold]\n")
    console.print(f"[dim]Modelo:[/dim] {model_path}")
    console.print(f"[dim]Endereço:[/dim] http://{host}:{port}")
    console.print()
    
    try:
        # Cria configuração
        api_config = APIConfig(port=port, host=host)
        export_config = ExportConfig(api=api_config)
        
        # Cria servidor
        server = ModelAPIServer(model_path, export_config)
        
        console.print("[bold green]Servidor iniciado![/bold green]")
        console.print("\n[bold]Endpoints:[/bold]")
        console.print(f"  POST http://{host}:{port}/v1/chat/completions")
        console.print(f"  POST http://{host}:{port}/v1/completions")
        console.print(f"  GET  http://{host}:{port}/v1/models")
        console.print(f"  GET  http://{host}:{port}/health")
        console.print("\n[dim]Pressione Ctrl+C para encerrar[/dim]\n")
        
        server.run(host=host, port=port)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Servidor encerrado[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]✗ Erro ao iniciar servidor:[/bold red] {e}")
        sys.exit(1)


def main():
    """Ponto de entrada principal do CLI."""
    cli()


if __name__ == "__main__":
    main()

