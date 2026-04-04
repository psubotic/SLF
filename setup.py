# MIT License
# Copyright (c) 2026 Pavle Subotic

from setuptools import find_packages, setup

setup(
    name="slf-trap-detector",
    version="0.1.0",
    description="Spotted Lanternfly detection in pheromone trap images",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
        "scikit-image>=0.21.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "ml": [
            "torch>=2.1.0",
            "torchvision>=0.16.0",
        ],
        "augmentation": [
            "albumentations>=1.3.1",
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "ipykernel>=6.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "slf-demo=scripts.run_demo:main",
            "slf-synth=scripts.generate_synthetic:main",
        ]
    },
)
