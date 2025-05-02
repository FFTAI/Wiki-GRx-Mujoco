from setuptools import setup, find_packages

setup(
    name='wiki-grx-mujoco',
    version='1.0.0',
    author='Xuanbo',
    author_email='xuanbo.wang@fftai.com',
    description='test policy in mujoco env',
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=2.3.1",
        "tqdm",
        "pynput",
        "matplotlib>=3.7.5",

        "mujoco",
        # "mujoco-python-viewer",
    ]
)
