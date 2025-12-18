from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import sys
import subprocess

class CustomInstallCommand(install):
    def run(self):
        install.run(self)

        base_dir = os.path.dirname(__file__)

        meta_dir = os.path.join(base_dir, "OrthoSAM", "MetaSAM")
        os.makedirs(meta_dir, exist_ok=True)

        urls = [
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        ]
        for url in urls:
            filename = url.split("/")[-1]
            filepath = os.path.join(meta_dir, filename)
            if not os.path.exists(filepath):
                subprocess.run(["wget", "-O", filepath, url], check=True)
        update_script = os.path.join(base_dir, "update_config.py")
        subprocess.run([sys.executable, update_script], check=True)


setup(
    name="OrthoSAM",
    version="0.1",
    packages=find_packages(),
    python_requires="==3.12",
    cmdclass={
        'install': CustomInstallCommand,
    },
)
