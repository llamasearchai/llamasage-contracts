from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llamasage-contracts-llamasearch",
    version="0.1.0",
    author="LlamaSearch AI",
    author_email="nikjois@llamasearch.ai",
    description="A powerful library for AI-powered search and data processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://llamasearch.ai",
    project_urls={
        "Bug Tracker": "https://github.com/llamasearchai/llamasage-contracts/issues",
        "Documentation": "https://llamasearchai.github.io/llamasage-contracts/",
        "Source Code": "https://github.com/llamasearchai/llamasage-contracts",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(include=["llamasage-contracts", "llamasage-contracts.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "requests>=2.25.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "isort>=5.9.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
        ],
        "mlx": [
            "mlx>=0.0.5",
        ],
        "all": [
            "mlx>=0.0.5",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "scikit-learn>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llamasage-contracts=llamasage-contracts.cli:main",
        ],
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)

# Updated in commit 5 - 2025-04-04 17:36:07

# Updated in commit 13 - 2025-04-04 17:36:07

# Updated in commit 21 - 2025-04-04 17:36:08

# Updated in commit 29 - 2025-04-04 17:36:08

# Updated in commit 5 - 2025-04-05 14:38:07

# Updated in commit 13 - 2025-04-05 14:38:07

# Updated in commit 21 - 2025-04-05 14:38:07

# Updated in commit 29 - 2025-04-05 14:38:07

# Updated in commit 5 - 2025-04-05 15:24:34

# Updated in commit 13 - 2025-04-05 15:24:34

# Updated in commit 21 - 2025-04-05 15:24:34

# Updated in commit 29 - 2025-04-05 15:24:34

# Updated in commit 5 - 2025-04-05 16:00:16

# Updated in commit 13 - 2025-04-05 16:00:17

# Updated in commit 21 - 2025-04-05 16:00:17

# Updated in commit 29 - 2025-04-05 16:00:17

# Updated in commit 5 - 2025-04-05 17:05:34

# Updated in commit 13 - 2025-04-05 17:05:34

# Updated in commit 21 - 2025-04-05 17:05:34

# Updated in commit 29 - 2025-04-05 17:05:34

# Updated in commit 5 - 2025-04-05 17:37:40

# Updated in commit 13 - 2025-04-05 17:37:40

# Updated in commit 21 - 2025-04-05 17:37:40

# Updated in commit 29 - 2025-04-05 17:37:40

# Updated in commit 5 - 2025-04-05 18:24:26
