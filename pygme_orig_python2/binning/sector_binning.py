#!/usr/bin/env python
"""
This module is used to bin some data given by cartesian coordinates and values
into sectors.
This follows an algorithm similar (although not identical) to the one developed
by Michele Cappellari in his IDL routine (see his web page and the "sector_photometry.pro" routine).

For questions, please contact Eric Emsellem at eric.emsellem@eso.org
"""

"""
Importing the most import modules
This MGE module requires NUMPY 
The most efficient routine uses SCIPY (> 0.11.0) but it is not required
"""
try:
    import numpy as np
except ImportError:
    raise Exception("numpy is required for pygme")

## OLD SLOW ####
## try:
##     import scipy 
##     if (scipy.__version__ >= "0.11.0") :
##         OPTION_Scipy = 1
##     else :
##         OPTION_Scipy = 0
## except ImportError:
##     raise Exception("scipy is required for pygme")
##     OPTION_Scipy = 0
## from scipy.stats import binned_statistic_2d

from pygme.mge_miscfunctions import convert_xy_to_polar, rotxyC, convert_polar_to_xy
from pygme.utils.biweight import biweight_mean, biweight_sigma
from pygme.utils.biweight import robust_mean, robust_sigma

__version__ = '1.0.0 (26 December, 2012)'

def bin_to_sectors(x, y, data, **kwargs) :
    """
    This routine will bin your data (x, y, data) into sectors defined by
    the number (between 0 and 90 degrees) and their width.

    Ellipticity : ellipticity (scalar) which will be used for defining the rings
    Center : 2 numbers giving the center in X and Y (default is [0.,0.])
    PA : position angle measured from top (counter-clockwise, default is -90.0 meaning the X is
         already along the abscissa)
    
    NSectors : number of sectors. Default is 19 (to cover 0-90 degrees with 5 degrees sectors)
    FactorRadius : factor for sampling the radius. Default is 1.1, which provides about 24 points per decade.
    WidthAngle : total Width in degrees of each sector (default is 5)
    SymQuad: by default to True. Will thus derive the sectors only from 0 to 90. If set to False, it will
             compute the binning in sectors between 0 and 360 degrees.

    MinLevel : minimum level for the data to be taken into account (default is -999.0)
    MinBinLevel : minimum Binned level for the data to be taken into account (default is 0.)
    Gain : in case we need the Poissonian Error to be computed

    verbose : default to 0

    Return: 
            xout, yout, dataout, sigmaout
            Where xout, yout are the cartesian coordinates of the bins
            dataout is the binned value of that bin
            sigmaout is the estimated standard deviation
    """
    Ellipticity = kwargs.get('Ellipticity', 0.0)
    Center = kwargs.get('Center', [0.0,0.0])
    MinLevel = kwargs.get('MinLevel', -999.0)
    MinBinLevel = kwargs.get('MinBinLevel', 0.0)
    PA = kwargs.get('PA', -90.0)

    NSectors = kwargs.get('NSectors', 19)
    WidthAngle = kwargs.get('WidthAngle', 5.0)
    SymQuad = kwargs.get('SymQuad', True)

    Gain = kwargs.get('Gain', 1.0)
    verbose = kwargs.get('verbose', 0)

    ## This provides a factor of 1.1 per range
    FactorRadius = kwargs.get('FactorRadius', 1.1)
    Nradii_per_decade = 1. / np.log10(1.1)

    ## First check that the sizes of the input coordinates + data are the same
    if (np.size(x) != np.size(y)) | (np.size(x) != np.size(data)) :
        print "ERROR : please check the sizes of your input arrays -> they should be the same!"
        return [0.], [0.], [0.]

    ## Then selecting the good points
    seldata = data > MinLevel
    x = x[seldata].ravel()
    y = y[seldata].ravel()
    data = data[seldata].ravel()
    ## Checking if all is ok
    if np.size(data) == 0 :
        print "ERROR : after selecting points above MinLevel (%f), the array is empty!"%(MinLevel)
        return [0.], [0.], [0.]

    ## Then check that the ellipticity is 0 <= eps < 1
    if (Ellipticity < 0) | (Ellipticity >= 1) :
        print "ERROR : please check your input Ellipticity (%f) as it should be in [0,1[ !"%(Ellipticity)
        return [0.], [0.], [0.]

    ## We first recenter the data
    PARadian = np.radians(PA)
    rcx, rcy = rotxyC(x, y, cx=Center[0], cy=Center[1], angle=PARadian + np.pi/2.)
    ## We then convert to polar coordinates with x axis along as abscissa
    ## If the symmetry is forced we use the absolute values
    if SymQuad :
        radii, theta = convert_xy_to_polar(np.abs(rcx), np.abs(rcy))
    else :
        radii, theta = convert_xy_to_polar(rcx, rcy)
    ## We do the same but using the ellipticity now
    qaxis = 1. - Ellipticity
    radii_ell, theta_ell = convert_xy_to_polar(rcx, rcy / qaxis)

##     ## And getting a log spaced sample
##     sample_rell = np.logspace(np.log10(minr_ell), np.log10(maxr_ell), Nradii_per_decade * np.log10(maxr_ell/minr_ell))
##     ## Adding the central points with a negative radii
##     sample_rell = np.concatenate(([-1.0], sample_rell))

    ## Now we sample the radii - First getting the minimum and maximum radii
    minr_ell, maxr_ell = np.min(radii_ell[np.nonzero(radii_ell)]), np.max(radii_ell)
    ## We go from minimum to max * 1.1
    sampleR = np.logspace(np.log10(minr_ell), np.log10(maxr_ell), 
                 Nradii_per_decade * np.log10(maxr_ell/minr_ell) + 1)
    ## And add the centre
    sampleR = np.concatenate(([-1.0], sampleR))
    Nradii = len(sampleR)

    ## Now we sample the Angles - All following angles in Radian
    if SymQuad :
        center_Angles = np.linspace(0., np.pi/2., NSectors)
    else :
        center_Angles = np.linspace(-np.pi / 2., 3. * np.pi / 2., NSectors)
    stepAngle = (center_Angles[1] - center_Angles[0]) / 2.
    low_Angles = center_Angles - stepAngle
    sampleT = np.concatenate((low_Angles, [center_Angles[-1] + stepAngle]))

    ## Creating the output Angle array by duplicating center_Angles Nradii-1 times
    thetaout = np.repeat(center_Angles[np.newaxis,:], Nradii-1, 0)
    ## Now we select for each sector the right points and compute the new binned data and errors
    radout = np.zeros_like(thetaout)
    dataout = np.zeros_like(radout) - 999.0
    sigmaout = np.zeros_like(radout)

##     ################ OLD - SLOW #################################################################################
##     def ChoiceMean(x) :
##         lx = len(x)
##         if lx > 10 : return robust_mean(x)
##         elif lx > 0 : return x.mean()
##         else : return -999.0
## 
##     def ChoiceStd(x) :
##         lx = len(x)
##         if lx > 10 : return robust_mean(x)
##         elif lx > 0 : return np.sqrt(Gain * x.sum()) / lx
##         else : return -999.0
## 
##     def ChoiceRadii(x) :
##         lx = len(x)
##         return np.average(x[:lx/2], weights=x[lx/2:])
## 
##     if OPTION_Scipy :
##         if verbose : print "Using Scipy Version (as scipy 0.11.0 or later is available)"
##         dradii_ell = np.concatenate((radii_ell, radii_ell))
##         dtheta = np.concatenate((theta, theta))
##         ddata = np.concatenate((radii_ell, data))
##         dataout = binned_statistic_2d(radii_ell, theta, data, bins=[sampleR, sampleT], statistic=ChoiceMean)[0]
##         sigmaout = binned_statistic_2d(radii_ell, theta, data, bins=[sampleR, sampleT], statistic=ChoiceStd)[0]
##         radout = binned_statistic_2d(dradii_ell, dtheta, ddata, bins=[sampleR, sampleT], statistic=ChoiceRadii)[0]
##     ################ OLD - SLOW #################################################################################

    ## Counting the number of points per bin
    histBins = np.histogram2d(radii_ell, theta, bins=[sampleR, sampleT])[0]
    ## We use the elliptical radius, but the circular angle
    ## We now get which bins each point gets into
    digitR = np.digitize(radii_ell, sampleR)
    digitT = np.digitize(theta, sampleT)

    ## And we loop over the Sectors
    for j in xrange(NSectors) :
        if verbose :
             print "Section %02d starting"%(j+1)
        ## We select the bins within that sector
        selJ = (digitT == j+1)
        dataJ = data[selJ]
        radiiJ = radii[selJ]
        digitRJ = digitR[selJ]
        ## We select the bins which have >0 and >10 within that sector
        selH0 = np.where((histBins[:,j] > 0) & (histBins[:,j] <= 10))[0]
        selH10 = np.where(histBins[:,j] > 10)[0]
        ## Then we make the calculation for the two species
        ## The ones which have more than 10 points and the ones which have at
        ## least one point.
        for i in selH0 :
            spoints = (digitRJ == i+1) 
            dataS = dataJ[spoints]
            radout[i, j] = np.average(radiiJ[spoints], weights=dataS)
            dataout[i,j] = dataS.mean()
            sigmaout[i,j] = np.sqrt(Gain * dataS.sum()) / histBins[i,j]
        for i in selH10 :
            spoints = (digitRJ == i+1)
            dataS = dataJ[spoints]
            radout[i, j] = np.average(radiiJ[spoints], weights=dataS)
            dataout[i,j] = robust_mean(dataS)
            sigmaout[i,j] = robust_sigma(dataS)

    ## Final selection without the negative points
    selfinal = (dataout > -999) & (dataout > MinBinLevel)
    ## Finally converting it back to x, y
    xout, yout = convert_polar_to_xy(radout[selfinal], thetaout[selfinal] + np.pi/2. + PARadian)
    return xout, yout, dataout[selfinal], sigmaout[selfinal]
