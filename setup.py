from setuptools import setup, find_packages

setup(
    name='FastQML',
    version='0.1',
    author='Mateusz Papierz',
    author_email='m2papierz@gmail.com',
    description='A Python library for Quantum Machine Learning, designed for easy and scalable '
                'implementation of QML models, optimized for CPU and GPU.',
    packages=find_packages(),
    install_requires=[
        'pytest',
        'numpy',
        'pennylane',
        'jax',
        'optax',
        'flax'
    ],
    url='https://github.com/m2papierz/FastQML',
    license='GNU General Public License v3.0'
)
