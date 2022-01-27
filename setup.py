#!/usr/bin/env python
import os
import sys
from setuptools import setup

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(name='giants',
      version='0.0.1',
      description="",
      author='Samuel Grunblatt, Nicholas Saunders',
      license='',
      package_dir={'': 'src'},
      install_requires=install_requires,
      )
