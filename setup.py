from setuptools import setup, find_packages

setup(
    name="OrthoSAM",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.12",
    entry_points={
        "console_scripts": [
            "orthosam-setup=OrthoSAM.setup:main",
        ]
    },
)

