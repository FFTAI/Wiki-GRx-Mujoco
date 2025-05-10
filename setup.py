from setuptools import setup, find_packages

setup(
    name="wiki_grx_mujoco",
    version="1.0.0",
    description="Test Fourier robots' policy in MUJOCO environment",
    author="Jason Chen",
    author_email="xin.chen@fftai.com",
    license="LGPL-3.0",
    packages=find_packages(),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=2.3.1",

        "tqdm",
        "pynput",
        "matplotlib>=3.7.5",

        "mujoco>=3.0.0",
        # "mujoco-python-viewer",
    ]
)
