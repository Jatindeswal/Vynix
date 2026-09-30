#!/usr/bin/env python3
"""
Project Vynix: Production-grade setup.py for packaging and distribution.
"""

from setuptools import setup
import os

here = os.path.abspath(os.path.dirname(__file__))

# Read README for long description
readme_path = os.path.join(here, "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="vynix",
    version="1.0.0",
    description="Few-Shot Human-Object Interaction (HOI) Detection with Geometric Hallucination Veto",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Project Vynix Team",
    author_email="contact@vynix.ai",
    url="https://github.com/Jatindeswal/Vynix",
    license="MIT",
    py_modules=[
        "vynix_cli",
        "vynix_fewshot_adapter",
        "train_vynix_full",
        "test_vynix_pretrained",
        "evaluate_vynix_full",
        "vynix_prototype",
    ],
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "transformers>=4.30.0",
        "opencv-python-headless>=4.8.0",
        "Pillow>=9.5.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        "datasets>=2.14.0",
        "huggingface-hub>=0.16.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vynix = vynix_cli:main",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
)
