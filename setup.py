"""
ModelForge - Sistema de Fine-tuning de Modelos ML
Setup para instalação do pacote
"""

from setuptools import setup, find_packages
from pathlib import Path

# Lê o README para a descrição longa
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Lê as dependências do requirements.txt
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="modelforge",
    version="0.1.0",
    author="ModelForge Team",
    author_email="modelforge@example.com",
    description="Sistema de Fine-tuning de Modelos ML orientado por configuração YAML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/modelforge/modelforge",
    project_urls={
        "Bug Tracker": "https://github.com/modelforge/modelforge/issues",
        "Documentation": "https://github.com/modelforge/modelforge#readme",
        "Source Code": "https://github.com/modelforge/modelforge",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=24.1.0",
            "isort>=5.13.0",
            "mypy>=1.8.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "modelforge=modelforge.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "modelforge": [
            "templates/*.yaml",
            "templates/*.yml",
            "templates/Dockerfile",
        ],
    },
    zip_safe=False,
)

