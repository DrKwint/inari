from setuptools import setup
import setuptools

version = "0.0.1"

setup(
    name='inari',
    version=version,
    packages=setuptools.find_packages(),
    setup_requires=[
        "setuptools", 
        "wheel"
    ],
    install_requires=[
        'jax>=0.3.5',
        'flax>=0.4.1',
        'gin-config>=0.5.0',
        'rlax>=0.1.2',
        'dopamine-rl>=4.0.2',
        'optax>=0.1.1',
        'tensorflow>=2.8.0',
        'gym>=0.23.1',
        'jinja2>=3.1.1',
    ],
)