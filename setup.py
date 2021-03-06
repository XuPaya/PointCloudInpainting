# install using 'pip install -e .'

from setuptools import setup

setup(name='pointnet',
      packages=['pointnet', 'PyTorchEMD'],
      package_dir={'pointnet': 'pointnet', 'PyTorchEMD': 'PyTorchEMD'},
      install_requires=['torch',
                        'tqdm',
                        'plyfile'],
    version='0.0.1')
