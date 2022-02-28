"""Copyright (C) 2011 ESO/Centre de Recherche Astronomique de Lyon (CRAL)

   fitting : fitting an MGE model on some coordinates + data
             Include 1d and 2d fitting
"""
__version__ = '0.0.3'
__revision__ = '$Revision: 1.00 $'
__date__ = '$Date: 2012/08/02 10:05 $'

# try :
#     import matplotlib
#     from matplotlib import pylab
# except ImportError:
#     print 'There is a problem with importing matplotlib (pylab) at initialisation'
#
# try :
#     import numpy
# except ImportError:
#     print 'There is a problem with importing numpy at initialisation'

from . import fitn1dgauss, fitn2dgauss, examine_fit
from . import fitGaussHermite

## Use mpfit for the non-linear least-squares
from . import mpfit
## Tring to import lmfit
try :
    # If it works, can use lmfit version
    # for the non-linear least squares
    import lmfit
    from lmfit import minimize, Parameters, Minimizer
    Exist_LMFIT = True
except ImportError :
    Exist_LMFIT = False
    print("WARNING: Only mpfit is available an optimiser")
    print("WARNING: you may want to install lmfit!")

from pygme.mge_miscfunctions import convert_xy_to_polar

# If Openopt is there we can import it and allow the use to use BVLS
# Otherwise we just use nnls which is faster anyway
try :
    # If it works, can use openopt for 
    # bound-constrained linear least squares (wrapper of BVLS)
    import openopt
    from openopt import LLSP
    Exist_OpenOpt = True
except ImportError :
    Exist_OpenOpt = False
    print("WARNING: OpenOpt was not found so only nnls is available as a linmethod")

__all__ = ['fitn1dgauss', 'fitn2dgauss', 'fitGaussHermite', 'mpfit', 'examine_fit']
