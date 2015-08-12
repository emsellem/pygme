#!/usr/bin/env python
"""Pygme: Python Gaussian ModElling - a python implementation of the Multi-Gaussian Expansion Method.
          Fit MGE models, and Generate initial conditions for N body simulations
          See Monnet et al. 1992 and Emsellem et al. 1994 for more details
"""
## Distribution for the PyMGE package

import sys

# simple hack to allow use of "python setup.py develop".  Should not affect
# users, only developers.
if 'develop' in sys.argv:
    # use setuptools for develop, but nothing else
    from setuptools import setup
else:
    from distutils.core import setup

import os
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')
setup(name='pygme',
      version='0.0.2',
      description='PYthon Gaussian ModElling - Python MGE Tool',
      author='Eric Emsellem',
      author_email='eric.emsellem@eso.org',
      maintainer='Eric Emsellem',
#      url='http://',
#      requires=['pymodelfit'],
#      requires=['openopt'],
      license='LICENSE',
      packages=['pygme', 'pygme.binning', 'pygme.astroprofiles', 'pygme.fitting', 'pygme.utils', 'pygme.colormaps'],
      package_dir={'pygme.astroprofiles': 'pygme/astroprofiles'},
      package_data={'pygme.astroprofiles': ['data/*.dat']},
     )
