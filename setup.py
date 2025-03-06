#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(name='giants',
      version='0.0.1',
      description="",
      author='Nicholas Saunders, Samuel Grunblatt',
      license='MIT',
      packages=['giants'],
      package_dir={'': 'src'},
      package_data={'giants': ['data/downlinks.txt']},
      install_requires=install_requires,
      )
