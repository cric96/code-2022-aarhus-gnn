#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='gnn-forecast',
    version='0.0.3',
    description='Forecasting of distributed spatio-temporal phenomena using Graph Neural Network + Recurrent Neural Networks',
    author='cric96',
    author_email='gianluca.aguzzi@unibo.it',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/cric96/code-2022-aarhus-gnn',
    install_requires=['pytorch-lightning', 'torch-geometric-temporal'],
    packages=find_packages('project'),
    package_dir={'':'project'}
)
