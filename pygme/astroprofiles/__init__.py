"""Copyright (C) 2011 ESO/Centre de Recherche Astronomique de Lyon (CRAL)

   astroprofiles: includes a few functions which are useful to describe
                  surface brightness or spatial profiles

"""
__version__ = '0.0.1'
__revision__ = '$Revision: 1.00 $'
__date__ = '$Date: 2012/03/04 15:05 $'

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

from . import nfw
from . import sersic
__all__ = ['nfw', 'sersic']
