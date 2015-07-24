"""Copyright (C) 2011 ESO/Centre de Recherche Astronomique de Lyon (CRAL)

   binning : include Voronoi binning and sectors
"""
__version__ = '0.0.1'
__revision__ = '$Revision: 1.00 $'
__date__ = '$Date: 2012/03/04 15:05 $'

#try :
#    import matplotlib
#    from matplotlib import pylab
#except ImportError:
#    print 'There is a problem with importing matplotlib (pylab) at initialisation'
try:
    import astropy  
    import astropy.io.fits as pyfits
except ImportError:
    print 'There is a problem with importing pyfits at initialisation'
#try :
#    import numpy
#except ImportError:
#    print 'There is a problem with importing numpy at initialisation'

__all__ = ['voronoibinning', 'sector_binning']
