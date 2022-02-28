""" Copyright (C) 2011 ESO/Centre de Recherche Astronomique de Lyon (CRAL)

    print pygme.__LICENSE__  for the terms of use

    This package allows the use of Multi Gaussian Expansion models (Monnet et al.
    1992, Emsellem et al. 1994). It can read and write MGE input ascii files and
    computes a number of basic parameters for the corresponding models.  It includes
    the derivation of velocity moments via the Jeans Equations, and the generation
    of positions and velocities for N body models.

    WARNING: this module is evolving quickly (and may still contains some obvious bugs).
    You are welcome to send comments to Eric Emsellem (eric.emsellem@eso.org).

    The package provides functions to :

    * Fit MGE models to 1D or 2D models and/or images
    * Create initial conditions (realisations) from MGE N-body models
    * Useful functions and modules to interact with MGE models
       * reconstruct 2D, 3D densities
       * project, deproject models assuming axisymmetry (soon triaxial)
       * Compute epicycle frequencies, circular velocities, potential
       * Derive the dispersion tensor (via Jeans)
    * Read/write GADGET2, RAMSES initial Condition files
    * Display snapshots
    * Includes miscellaneous modules: voronoi binning, simple profiles (nfw, sersic)

    Submodules:
    ===========
    rwcfor:
        input and output for binary files written in C and Fortran

    snapshot :
        Generic snapshot class

    plotsnap :
        Snapshot plot routines

    pygadget :
        Generate and read Gadget 2 Initial Condition files

    pyramses :
        Generate and read RAMSES Initial Condition files

    pyhist :
        Simple histogram functions (1D/2D) and moments


    In sub-module fitting:
        fitn1dgauss.py :
            fitting an MGE model to a set of points (1D)
        fitn2dgauss.py :
            fitting an MGE model to a set of points (2D)

    In sub-module binning:
        voronoibinning:
            provides a way to bin data with Voronoi nodes

    In sub-module colormaps:
        mycolormap:
            provides the SAURON color map

    In sub-module astroprofiles:
        nfw: Navarro-Frenk-White defined by c or Mvir
        sersic: sersic profiles

"""
__LICENSE__ = """
    Copyright (C) 2012 ESO / Centre de Recherche Astronomique de Lyon (CRAL)

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

        1. Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.

        2. Redistributions in binary form must reproduce the above
          copyright notice, this list of conditions and the following
          disclaimer in the documentation and/or other materials provided
          with the distribution.

        3. The name of AURA and its representatives may not be used to
          endorse or promote products derived from this software without
          specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY AURA ``AS IS'' AND ANY EXPRESS OR IMPLIED
    WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL AURA BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
    OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
    TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
    USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.

"""
__version__ = '2.0.1'
__revision__ = '$Revision: 1.00 $'
__date__ = '$Date: 17/09/2012 15:05 $'

"""
    Trying to load the different needed modules
"""

"""
    Import the different submodules
"""
from . import binning
from . import astroprofiles
from . import fitting
from . import colormaps
from . import utils
from .mge import MGE
__all__ = ['mge', 'photMGE', 'dynMGE', 'init_partMGE',
           'mge_miscfunctions', 'paramMGE', 'pygadget',
           'pyramses', 'plotsnap', 'rwcfor', 'pyhist', 'snapshot',
           'binning', 'astroprofiles', 'fitting', 'utils', 'colormaps']
