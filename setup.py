
from setuptools import setup, find_packages
import os


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='neural_network_model',
    version='0.1.0',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],

    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'albumentations>=1.1.0',
        'opencv-python-headless>=4.5.5',
         'matplotlib>=3.5.0',
         'Pillow>=9.0.0',
         'pytorch-lightning>=2.0.0',
         'torchmetrics>=0.11.0',
         'albumentations>=1.3.0',
         'numpy>=1.21.0',
         'tensorboard>=2.10.0',

    ],

    include_package_data=True,
    license='MIT',
)
