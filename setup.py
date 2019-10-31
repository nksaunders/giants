#!/usr/bin/env python
import os
import sys
from setuptools import setup
from giants import *

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(name='giants',
      version='0.0.0',
      description="",
      author='Samuel Grunblatt, Nicholas Saunders',
      license='',
      package_dir={'giants': 'giants'},
      install_requires=install_requires,
      )
