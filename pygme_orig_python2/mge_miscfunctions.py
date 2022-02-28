#!/usr/bin/python
"""
This module includes a few basic functions useful for the pygme
Multi Gaussian Expansion models (Monnet et al. 1992, Emsellem et al. 1994)
python module.
For questions, please contact Eric Emsellem at eric.emsellem@eso.org
"""

"""
Importing the most import modules
This module requires NUMPY and SCIPY and optionally matplotlib (for plots)
"""
try:
    import numpy as np
except ImportError:
    raise Exception("numpy is required for pygme")

from numpy import random, asarray

try:
    from scipy import special, interpolate, optimize
except ImportError:
    raise Exception("scipy is required for pygme")

from rwcfor import floatMGE
from numpy import sin, cos, exp, sqrt, pi

import matplotlib
import matplotlib.pyplot as plt

import os

__version__ = '1.0.2 (14 August 2014)'

# Version 1.0.3 : GaussHermite only returns the derived profile (1 array)
# Version 1.0.2 : Changed some scaling and output
# Version 1.0.1 : Removed imin imax
# Version 1.0.0 : first extraction from pygme
############################################################################
#                   Misc Functions
############################################################################

# ====================================================
# Print a message
# ====================================================
def print_msg(text="", status=0, verbose=0) :
    """ Internal pygme method to print messages

    :param text: Text to be written
    :type text: string
    :param status: Status (default is 0). If 0 just print. If 1, state a WARNING. If 2, state an ERROR.
    :type status: integer (0, 1 or 2)
    :param verbose: 0 or 1 (default is 0).

    """
    if status >= 2 :
        print "ERROR with status %d while %s" %(status, text)
    elif status == 1 :
        print "WARNING with status %d while %s" %(status, text)
    elif status <= 0 :
        if verbose :
            print "Status OK (0) while %s" %(text)
        print text
       
#===================================================

### ##################################################################
### ### Set up the min and max indices depending on the input      ###
### ##################################################################
### def _set_iminmax(imin=None, imax=None, NMAX=0) :
###     """
###     Set the indices as wished for MGE routines.
### 
###     :param imin: id of first Gaussian to consider (between 0 and Ngauss-1)
###     :type imin: int
###     :param imax: id of last Gaussians to consider (between 0 and Ngauss-1)
###     :type imax: int
### 
###     :return: 0, NGauss, NGauss-1 if the input is Null
###              Otherwise imin, imax, and imax+1.
### 
###     DEPRECATED FUNCTION
### 
###     """
###     if imin == None :
###         imin = 0
###     if imax == None :
###         imax = NMAX - 1
###     return imin, imax, imax+1
### ###===============================================================

#=============================================================================================================
# Return abs/weights from the orthogonal scipy quadrature
#=============================================================================================================
def quadrat_ps_roots(Nabs) :
    return return_floatXY(special.orthogonal.ps_roots(Nabs))
#-------------------------------------------------------------------------------------------------------------

#=============================================================================================================
# Extract the right float values from the orthogonal scipy output
#=============================================================================================================
def return_floatXY(temparray) :
    """
    Return a list of 2 arrays, converted from an input array which has a real and
    imaginary part, as in the output of the scipy.orthogonal.ps_roots scipy routine
    """

    X = asarray(temparray[0].real, dtype=floatMGE)
    Y = asarray(temparray[1], dtype=floatMGE)
    return [X,Y]
#-------------------------------------------------------------------------------------------------------------

#=============================================================================================================
# Return a realisation for a truncated Gaussian with sigma
#=============================================================================================================
def sample_trunc_gauss(sigma=1., cutX=1., npoints=1, even=0):
    """
    Function which returns a sample of points (npoints) which follow
    a Gaussian distribution truncated at X=cutX
    This uses the special erf function from scipy and the random uniform function
    As well as the erfinv (inverse of erf) function
    Input:
       sigma    :   sigma of the Gaussian in arbitrary units (default is 1.0)
       cutX     :   truncature (positive) in same units than sigma (default is 1.0)
       npoints  :   Number of points for the output sample (default is 1)
       even     :   if even=1, cut with -cutX, cutX
                       otherwise, cut between 0 and cutX (default is 0 => not even)
    """
    sqrt2sig = np.sqrt(2.)*sigma
    cutsamp = special.erf(cutX/sqrt2sig)
    if even :  ## If distribution needs to be symmetric
        return sqrt2sig * special.erfinv(random.uniform(-cutsamp, cutsamp, npoints))
    else :
        return sqrt2sig * special.erfinv(random.uniform(0., cutsamp, npoints))

#-------------------------------------------------------------------------------------------------------------

#=============================================================================================================
# Return a realisation for a truncated r^2*Gaussian with sigma
#=============================================================================================================
def sample_trunc_r2gauss(sigma=1., cutr=1., npoints=1, nSamp=10000):
    """
    Function which returns a sample of points (npoints) which follow
    a r^2 * Gaussian distribution truncated at r=cutr
    This uses the special erf function from scipy, an interpolation for the
    cumulative integrated function and then a random uniform function
    Input:
       sigma    :   sigma of the Gaussian in arbitrary units
       cutr     :   truncature (positive) in same units than sigma
       npoints  :   Number of points for the output sample
    """
    sqrt2sig = np.sqrt(2.)*sigma
    ## Sampling in r with nSamp points - default to 10000 points to sample well the profile
    sampx = np.linspace(0.,cutr,nSamp)
    sampxsig = sampx / sqrt2sig     # normalised to the sigma

    ## Cumulative function of 4*pi*r2*exp(-r2/2*sigma**2)
    fSG = 2.*np.pi * sqrt2sig**3. * (-sampxsig * np.exp(-sampxsig**2)+special.erf(sampxsig) * np.sqrt(np.pi)/2.)
    ## Interpolation to get the inverse function
    invF = interpolate.interp1d(fSG, sampx)

    return invF(random.uniform(0, fSG[-1], npoints))

#-------------------------------------------------------------------------------------------------------------
#=============================================================================================================
# Return a realisation for a truncated r^2*Gaussian with sigma
#=============================================================================================================
def gridima_XY(npix=(1,1), center=(0.,0.), step=(1.,1.)) :
    """
    Return 2D X,Y grids assuming npix pixels, given the centre and step

    npix : tuple of integers providing the number of pixels in X and Y
    center : tuple of float providing the centre of the array
    step : tuple of float (or single float) providing the size of the pixel
    """

    if len(step) == 1 : step = (step, step)
    X,Y = np.meshgrid(np.linspace(0,npix[1]-1, npix[1]), np.linspace(0,npix[0]-1,npix[0]))
    X = (X - center[0]) * step[0]
    Y = (Y - center[1]) * step[1]
    return X, Y

def convert_xy_to_polar(x, y, cx=0.0, cy=0.0, PA=None) :
    """
    Convert x and y coordinates into polar coordinates

    cx and cy: Center in X, and Y. 0 by default.
    PA : position angle in radians
         (Counter-clockwise from vertical)
         This allows to take into account some rotation
         and place X along the abscissa
         Default is None and would be then set for no rotation

    Return : R, theta (in radians)
    """
    if PA is None : PA = -np.pi / 2.
    ## If the PA does not have X along the abscissa, rotate
    if np.mod(PA+np.pi/2., np.pi) != 0.0 : x, y = rotxyC(x, y, cx=cx, cy=cy, angle=PA+np.pi/2.)
    else : x, y = x - cx, y - cy

    ## Polar coordinates
    r = np.sqrt(x**2 + y**2)
    ## Now computing the true theta
    theta = np.zeros_like(r)
    theta[(x == 0.) & (y >= 0.)] = pi / 2.
    theta[(x == 0.) & (y < 0.)] = -pi / 2.
    theta[(x < 0.)] = np.arctan(y[(x < 0.)] / x[(x < 0.)]) + pi
    theta[(x > 0.)] = np.arctan(y[(x > 0.)] / x[(x > 0.)])
    return r, theta
#-------------------------------------------------------------------------------------------------------------
def convert_polar_to_xy(r, theta) :
    """
    Convert x and y coordinates into polar coordinates Theta in Radians
    Return :x, y
    """

    ## cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y
#-------------------------------------------------------------------------------------------------------------
def rotxC(x, y, cx=0.0, cy=0.0, angle=0.0) :
    """ Rotate by an angle (in radians) 
        the x axis with a center cx, cy

        Return rotated(x)
    """
    return (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)

def rotyC(x, y, cx=0.0, cy=0.0, angle=0.0) :
    """ Rotate by an angle (in radians) 
        the y axis with a center cx, cy

        Return rotated(y)
    """
    return (cx - x) * np.sin(angle) + (y - cy) * np.cos(angle)

def rotxyC(x, y, cx=0.0, cy=0.0, angle=0.0) :
    """ Rotate both x, y by an angle (in radians) 
        the x axis with a center cx, cy

        Return rotated(x), rotated(y)
    """
    ## First centring
    xt = x - cx
    yt = y - cy
    ## Then only rotation
    return rotxC(xt, yt, angle=angle), rotyC(xt, yt, angle=angle)
#-------------------------------------------------------------------------------------------------------------
def _gaussianROTC(height, center_x, center_y, width_x, width_y, angle):
    """Returns a gaussian function with the given parameters
       First is the shift, then rotation
    """
    width_x = np.float(width_x)
    width_y = np.float(width_y)
    return lambda x,y: height*np.exp( -(((rotxC(x, y, center_x, center_y, angle)) / width_x)**2+((rotyC(x, y, center_x, center_y, angle)) / width_y)**2)/2.)
#-------------------------------------------------------------------------------------------------------------
def twod_moments(data):
    """Returns (height, x, y, width_x, width_y, 0.)
       the gaussian parameters of a 2D distribution by calculating its moments 
       The last value (0.) stands for the default position angle
    """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y, 0.
#-------------------------------------------------------------------------------------------------------------
def Inertia_2DMoments(x, y, data) :
    """
    Derive the moment of inertia from a flux map
    It returns the major and minor axes, ellipticity and PA
    """
    momI = np.sum(data, axis=0)
    if momI == 0. :
       return 0., 0., 1., 0.
    momIX = np.sum(data * x, axis=0) / momI
    momIY = np.sum(data * y, axis=0) / momI
    a = np.sum(data * x * x, axis=0) / momI - momIX**2
    b = np.sum(data * y * y, axis=0) / momI - momIY**2
    c = np.sum(data * x * y, axis=0) / momI - momIX*momIY
    if c == 0 :
       if a == 0. :
          return b,a,0.,0.
       if b > a :
          return b,a,1.-np.sqrt(a/b),0.
       else :
          if b == 0.:
             return a,b,0.,0.
          else :
             return a,b,1.-np.sqrt(b/a),90.

    delta = (a-b)**2. + 4 * c*c
    Lp2 = ((a+b) + np.sqrt(delta)) / 2.
    Lm2 = ((a+b) - np.sqrt(delta)) / 2.
    Lp = np.sqrt(np.maximum(Lp2,0.))
    Lm = np.sqrt(np.maximum(Lm2,0.))
    eps = (Lp - Lm) / Lp
    theta = np.degrees(np.arctan((b - Lp2) / c))
    return Lp, Lm, eps, theta
#-------------------------------------------------------------------------------------------------------------
def fit_1gaussian(data):
    """  Returns (height, x, y, width_x, width_y)
         the gaussian parameters of a 2D distribution found by a fit
    """
    params = twod_moments(data)
    errorfunction = lambda p: np.ravel(_gaussianROTC(*p)(*np.indices(data.shape)) - data)
    bestfitparams, success = optimize.leastsq(errorfunction, params)
    return bestfitparams
#-------------------------------------------------------------------------------------------------------------
def _find_Image_labels(image=None, threshold=None) :
    """
    Select pixels within an image which are contiguous
    Used for find_ImageCenter, but here can be also used with a non-default threshold
    for the image level.

    image: input data array
    threshold : threshold above which the selection should be done

    Return: a selection from the data
    """
    import scipy.ndimage as ndima

    if threshold is None : threshold = image.mean() 
    labels, num = ndima.label(image > threshold, np.ones((3,3)))
    maxLabel = np.argmax(np.bincount(labels[labels>0].ravel()))

    select_label = (labels == maxLabel)
    return select_label, labels

#-------------------------------------------------------------------------------------------------------------
def find_ImageCenter(image=None, showfit=False, verbose=True, threshold=None) :
    """
       Find the centre of a galaxy using the centre of mass after filtering

       image : An input data array

       showfit: show the residual fit - default to False
       verbose : print results - default to True
       threshol : where to cut the image level - default to None (mean of Image)

       Return: xcen, ycen, major, minor, eps, theta
    """
    import scipy.ndimage as ndima
    from matplotlib.patches import Ellipse

    Mimage = ndima.median_filter(image, 3)
    ## Extracting headers and data
    startx, starty = 0., 0.
    stepx, stepy = 1., 1.
    npy, npx = image.shape
    endx = startx + (npx -1) * stepx
    endy = starty + (npy -1) * stepy
    Xin,Yin = np.meshgrid(np.linspace(startx,endx,npx), np.linspace(starty,endy,npy))

    ## We select the labels after aggregation with a default threshold (mean of the image)
    select_label, labels = _find_Image_labels(Mimage, threshold=threshold)
    maxLabel = np.argmax(np.bincount(labels[labels>0].ravel()))

    ## With the selection we can find the centre of mass using ndima
    centers = ndima.center_of_mass(image, labels, maxLabel)
    ## Being careful here as X and Y are in fact Y, X
    xcen = np.array(centers)[1]
    ycen = np.array(centers)[0]

    major, minor, eps, theta = Inertia_2DMoments(Xin[select_label]-xcen, Yin[select_label]-ycen, Mimage[select_label])
    maxRadius = np.max(np.sqrt((Xin[select_label]-xcen)**2+ (Yin[select_label]-ycen)**2))

    if showfit :
        fig = plt.figure(1, figsize=(8,6))
        ax = fig.add_subplot(111, aspect='equal')
        ax.plot(Xin[select_label], Yin[select_label], ',')
        ellipse = Ellipse(xy=(xcen,ycen), width=maxRadius*1.1*2.0, height=maxRadius*1.1*(1.-eps)*2.0,
                          angle=90.+theta, edgecolor='r', fc='None', lw=2)
        ax.add_patch(ellipse)
        ax.set_xlim(startx, endx)
        ax.set_ylim(starty, endy)

    if verbose :
        print "Center of the image found at: ", xcen, "  ", ycen
        print "Ellipticity: ", eps
        print "Position Angle: ", theta
        print "Maximum Radius of point accounted for: ", maxRadius
    return xcen, ycen, major, minor, eps, theta
#-------------------------------------------------------------------------------------------------------------
def fit_ImageCenter(image=None, showfit=False) :
    """
       Fit a gaussian on an image and show the best fit and returns the centre

       fitsfile : An input fits file
       showfit : default is False (not ploting the fit), if True -> will show the image and fitted gaussian

       Return: X, Y, fitdata which are the found central position and the fitted data
    """
    ## Extracting headers and data
    startx, stary = 0., 0.
    stepx, stepy = 1., 1.
    npx, npy = image.shape
    endx = startx + (npx -1) * stepx
    endy = starty + (npy -1) * stepy
    Xin,Yin = np.meshgrid(np.linspace(startx,endx,npx), np.linspace(starty,endy,npy))

    ## Starting the fit
    fitparams = fit_1gaussian(datatofit)

    ## Recompute the fitted gaussian
    fit = _gaussianROTC(*fitparams)
    (height, cenx, ceny, width_x, width_y, angle) = fitparams

    ## Rescaling the values
    cenxarc = ceny * stepx + startx
    cenyarc = cenx * stepy + starty
    width_xarc = width_x * stepx
    width_yarc = width_y * stepy 
    
    ## Some printing
    print "Center is X, Y = %8.4f %8.4f" %(cenxarc, cenyarc)
    print "Width is X, Y = %8.4f %8.4f" %(width_xarc, width_yarc)
    print "Start/End %8.4f %8.4f %8.4f %8.4f" %(startx, starty, endx, endy)
    print "Angle %8.4f" %(np.degrees(angle))

    fitdata = fit(*np.indices(datatofit.shape))
    if showfit:
        ## Doing the plot
        plt.clf()
        ## image
        plt.imshow(np.log10(datatofit+1.), extent=(startx, endx, starty, endy))

        ## Contours
        plt.contour(Xin, Yin, fitdata, cmap=plt.cm.copper)

    return Xin, Yin, fitdata
#-------------------------------------------------------------------------------------------------------------
def GaussHermite(Vbin=None, GH=None) :
    """ Returns the Gauss-Hermite function given 
    a set of parameters given as an array GH (first three moments are flux, velocity and dispersion)
    and the input sampling (velocities) Vbin
    """
    if Vbin is None :
        Vbin = np.linspace(-GH[2]*5. + GH[1], GH[2]*5. + GH[1],101)

    degree = len(GH) - 1
    if degree < 2 :
        print "Error: no enough parameters here"
        return Vbin * 0.
    if GH[2] == 0. :
        print "Error: Sigma is 0!"
        return Vbin * 0.
    VbinN = (Vbin - GH[1]) / GH[2]
    VbinN2 = VbinN * VbinN
    
    GH0 = (2. * VbinN2 - 1.0) / sqrt(2.)
    GH1 = (2. * VbinN2 - 3.) * VbinN / sqrt(3.)
    GH2 = GH1

    var = 1.0
    for i in xrange(3, degree+1) :
        var += GH[i] * GH2
        GH2 = (sqrt(2.) * GH1 * VbinN - GH0) / sqrt(i+1.0);
        GH0 = GH1;
        GH1 = GH2;
    return GH[0] * var * exp(- VbinN2 / 2.) 

def oldGaussHermite(Vbin, V, S, GH) :
    """
    Return the Gauss-Hermite function up to a certain degree
    using V, Sigma, and then an array describing h3, h4, ...
    """
    #------------------------------------------------------------
    # The Gauss-Hermite function is a superposition of functions of the form
    # F = (x-xc)/s                                            
    # E =  A.Exp[-1/2.F^2] * {1 + h3[c1.F+c3.F^3] + h4[c5+c2.F^2+c4.F^4]} 
    #------------------------------------------------------------
    c0 =     sqrt(6.0)/4.0
    c1 =    -sqrt(3.0)
    c2 =    -sqrt(6.0)
    c3 = 2.0*sqrt(3.0)/3.0
    c4 =     sqrt(6.0)/3.0

    F = (x-x0)/s
    E = A*numpy.exp(-0.5*F*F)*( 1.0 + h3*F*(c3*F*F+c1) + h4*(c0+F*F*(c2+c4*F*F)) )
    return E
