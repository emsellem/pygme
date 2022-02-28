######################################################################################
# fitn2dgauss.py
# V.1.0.2   19 May 2015 - Typo fixed on residuals
# V.1.0.1 - 13 March 2015 - Emsellem 
#           Changed a few lines to simplify the functions and variables
# V.1.0.0 - 12 Mai 2014 - Emsellem 
#           Adding some options including the PSF convolution and PA following the original Lyon lib C pgm.
# V.0.9.8 - 12 August 2013 - Emsellem / Ginsburg
#           Adding a printing function suggested by Adam Ginsburg
#           Removed epsfcn in lmfit
# V.0.9.7 - 23 July 2013 - Emsellem
#           Changing the residual function using the standard normalisation
# V.0.9.6 - 8 August 2012 - Emsellem
#           Adding a function to filter Gaussians which are not relevant
# V.0.9.5 - 3 August 2012 - Emsellem
#           Integrated all methods (mpfit, lmfit + NNLS or BVLS) in a single file
# V.0.9.2 - 27 June 2012 - Bois + Emsellem
#           A few fixes from Maxime Bois, and small changes from Emsellem
# V.0.9.1 - 22 January 2012 - Emsellem
#           Included np.log10 instead of log10
# V.0.9.0 - 22 January 2012 - Emsellem
# V.0.1.0 - 2010 - Emsellem
#           Reshuffling and packaging from old python MGE routines
#
# Largely Inspired from gaussfiter.py 
#    written by Adam Ginsburg (adam.ginsburg@colorado.edu or keflavich@gmail.com) 3/17/08)
#    latest version available at http://code.google.com/p/agpy/source/browse/trunk/agpy/gaussfitter.py
#
# Specific development of a N 2D Gaussian fitter # Eric Emsellem (eric.emsellem@eso.org)
#   Basis for the Multi-Gaussian Expansion method as described in Emsellem et al. 1994
#     and subsequent papers on the MGE technique
#   Decoupling the linear and non-linear parts, following the
#   algo fitting approach proposed by Michele Cappellari (see 2002 paper and available idl/python routines)
######################################################################################
import numpy as np
from numpy import exp, cos, sin, arctan, sqrt, pi, newaxis
from numpy import float64 as floatFit

## Use mpfit for the non-linear least-squares
from pygme.fitting.mpfit import mpfit

## Tring to import lmfit
try :
    # If it works, can use lmfit version
    # for the non-linear least squares
    import lmfit
    from lmfit import minimize, Parameters, Minimizer
    Exist_LMFIT = True
except ImportError :
    Exist_LMFIT = False
    print("WARNING: Only mpfit is available as an optimiser")
    print("WARNING: you may want to install lmfit")

from pygme.mge_miscfunctions import convert_xy_to_polar

# Use scipy nnls to solve the linear part
from scipy.linalg import lstsq
from scipy.optimize import nnls

""" 
Note (Ginsburg) about mpfit/leastsq: 
I switched everything over to the Markwardt mpfit routine for a few reasons,
but foremost being the ability to set limits on parameters, not just force them
to be fixed.  As far as I can tell, leastsq does not have that capability.  

The version of mpfit I use can be found here:
    http://code.google.com/p/agpy/source/browse/trunk/mpfit
    This is included in the present distribution

Note (Emsellem): this is now also solved using lmfit which is based on leastsq from scipy
  So two methods are implemented
"""
## Useful constants
twopi = 2.0 * np.pi

_default_parPSF = np.array([1.,0.])

## # ==========================================================================
## ## Extract the Gaussian parameters from the input pars
## # ==========================================================================
## def _extract_PSFParam(pars) :
##     """
##     Extract the PSF parameters from the original formatting
##     Internal routine for fitn2gauss.py
##     """
##     pars = pars.ravel()
##     if len(pars) % 2 == 0:
##         ## Normalising the energy of the total PSF to 1
##         ## Warning: Each Intensity is Imax for each PSF Gaussian
##         I = pars[::2] / np.sum(pars[::2])
##         sigma = pars[1::2]
##     else:
##         raise ValueError("Wrong array lengths!") 
##         return [1.], [0.]
##     return I, sigma
## 
# ==========================================================================
## Normalise the PSF given by Imax and sigmas
# ==========================================================================
def norm_PSFParam(pars) :
    """
    Normalise the PSF parameters from the original formatting
    Calculate the fraction (energy) of each Gaussian and normalise
    by the sum of all PSF Gaussians
    Internal routine for fitn2gauss.py
    """
    pars = pars.ravel()
    if len(pars) % 2 == 0:
        ## Normalising the energy of the total PSF to 1
        ## Warning: Each Intensity initially given is Imax for each PSF Gaussian
        ## So now we calculate the fraction of the energy taken by each Gaussian
        ## Hence the sum of intensities should be 1 after the normalisation
        if np.sum(pars[::2] * pars[1::2]**2) == 0 :
            pars[::2] = 1.
            pars[1::2] = 0.
        else :
            pars[::2] = pars[::2] * pars[1::2]**2 / np.sum(pars[::2] * pars[1::2]**2)
    else:
        raise ValueError("Wrong array lengths!") 
        return np.array([1., 0.])
    return pars

# ==========================================================================
## Returns a function which does the calculation, given a set of parameters,
## of the sum of Gaussians, the I being the maximum intensities (not normed)
# ==========================================================================
def n_centred_twodgaussian_Imax(pars, parPSF=_default_parPSF, I=None, q=None, sigma=None, pa=None):
    """
    Returns a function that provides the sum 
    over N 2D gaussians, where N is the length of
    I,q,sigma,pa *OR* N = len(pars) / 4

    The background level is assumed to be zero (you must "baseline" your
    spectrum before fitting)

    pars  - an array with len(pars) = 4n, assuming I, sigma, q, pa repeated n times
    parPSF - an array with the parameters from the PSF (formatted as pars)
    I, sigma, q, pa  - alternative amplitudes, line widths, axis ratios, PA
                       if pars is not provided (PA in degrees).
    """
    if pars is None :
        if not(len(I) == len(q) == len(sigma) == len(pa)):
            raise ValueError("Wrong array lengths! q: %i  sigma: %i  pa: %i" \
                    % (len(q), len(sigma), len(pa)))
    else :
        pars = pars.ravel()
        if len(pars) % 4 == 0:
            I = pars[::4]
            sigma = pars[1::4]
            q = pars[2::4]
            pa = pars[3::4]

    ## Convolution of the sigmas with the PSF
    ImaxPSF, sigmaPSF = parPSF[::2], parPSF[1::2]
    nPSF = len(ImaxPSF)
    sigmaX = np.sqrt((sigma**2)[:,newaxis] + sigmaPSF**2)
    sigmaY = np.sqrt(((q * sigma)**2)[:,newaxis] + sigmaPSF**2)
    Itot = (I * sigma**2 * q)[:,newaxis] * ImaxPSF[newaxis,:] / (sigmaX * sigmaY)

    parad = np.radians(pa)

    def g2d(r, theta):
        v = np.zeros_like(r)
        for i in range(len(q)):
            angle = - parad[i] + theta + pi / 2.
            v += np.sum([Itot[i,j] * exp( - 0.5 * r**2 * ((cos(angle)/sigmaX[i,j])**2 +  (sin(angle) / sigmaY[i,j])**2)) for j in range(nPSF)], axis=0)
        return v
    return g2d

# ==========================================================================
## Returns a function which does the calculation, given a set of parameters
## of the sum of Gaussians, the I being the NORMED intensities
# ==========================================================================
def _n_centred_twodgaussian_Inorm(pars, parPSF=_default_parPSF, I=None, logsigma=None, q=None, pa=None):
    """
    Returns a function that provides the sum 
    over N 2D gaussians, where N is the length of
    I,q,sigma,pa *OR* N = len(pars) / 4

    The background "height" is assumed to be zero (you must "baseline" your
    spectrum before fitting)

    pars  - an array with len(pars) = 4n, assuming I, logsigma, q, pa repeated n times
    I, logsigma, q, pa  - alternative amplitudes, line widths (in log10), axis ratios, PA
                       if pars is not provided (PA in degrees).
    """
    pars = pars.ravel()
    I = pars[::4]
    sigma = 10**(pars[1::4])
    q = pars[2::4]
    pa = pars[3::4]

    ## Convolution of the sigmas with the PSF
    ImaxPSF, sigmaPSF = parPSF[::2], parPSF[1::2]
    nPSF = len(ImaxPSF)
    sigmaX = np.sqrt(sigma[:,newaxis]**2 + sigmaPSF**2)
    sigmaY = np.sqrt(((q * sigma)**2)[:,newaxis] + sigmaPSF**2)

#    Fgauss = twopi * sigmaX * sigmaY
#    Itot = (I * sigma**2 * q)[:,newaxis] * ImaxPSF[newaxis,:] / (sigmaX * sigmaY)
    Itot = ImaxPSF[newaxis,:] / (twopi * sigmaX * sigmaY)

    parad = np.radians(pa)

    def g2d(r, theta):
        v = np.zeros_like(r)
        for i in range(len(q)):
            angle = - parad[i] + theta + pi / 2.
#            v += np.sum([Itot[i,j] * exp( - 0.5 * r**2 * ((cos(angle)/sigmaX[i,j])**2 +  (sin(angle) / sigmaY[i,j])**2)) / Fgauss[i,j] for j in range(nPSF)], axis=0)
            v += np.sum([Itot[i,j] * exp( - 0.5 * r**2 * ((cos(angle)/sigmaX[i,j])**2 +  (sin(angle) / sigmaY[i,j])**2)) for j in range(nPSF)], axis=0)
        return v.ravel()
    return g2d

# ==========================================================================
#---------- Centred (no offset) 2D Gaussian but without flux ---------------##
#---------- Returns a function which provide a set of N normalised Gaussians 
#---------- But NORMALISED BY the data
# ==========================================================================
def _n_centred_twodgaussian_Datanorm(pars, parPSF=_default_parPSF, logsigma=None, q=None, pa=None):
    """
    Returns a function that provides an array of N NORMALISED 2D gaussians, 
    where N is the length of q,sigma,pa *OR* N = len(pars) / 3. These 
    Gaussians are also normalised by an input data array
    (the returned function takes 3 arguments: r, theta and data)

    pars  - an array with len(pars) = 3n, assuming q, logsigma, pa repeated n times
    logsigma, q, pa  - alternative line widths (in log10), axis ratios, PA
                       if pars is not provided (PA in degrees).
    """
    pars = pars.ravel()
    sigma = 10**(pars[::3])
    q = pars[1::3]
    pa = pars[2::3]

    ## Convolution of the sigmas with the PSF
    ImaxPSF, sigmaPSF = parPSF[::2], parPSF[1::2]
    nPSF = len(ImaxPSF)
    sigmaX = np.sqrt(sigma[:,newaxis]**2 + sigmaPSF**2)
    sigmaY = np.sqrt(((q * sigma)**2)[:,newaxis] + sigmaPSF**2)
#    Fgauss = twopi * sigmaX * sigmaY
#    Itot = (sigma**2 * q)[:,newaxis] * ImaxPSF[newaxis,:] / (sigmaX * sigmaY)
    Itot = ImaxPSF[newaxis,:] / (twopi * sigmaX * sigmaY)

    parad = np.radians(pa)

    def g2d_datanorm(r, theta, data):
        v = np.zeros((np.size(r),len(q)))
        for i in range(len(q)):
            angle = - parad[i] + theta + pi / 2. # in radians
#            v[:,i] = np.sum([Itot[i,j] * exp( - 0.5 * r**2 * ((cos(angle)/sigmaX[i,j])**2 +  (sin(angle) / sigmaY[i,j])**2) ) \
#                    / (Fgauss[i,j]) for j in range(nPSF)], axis=0) / data
            v[:,i] = np.sum([Itot[i,j] * exp( - 0.5 * r**2 * ((cos(angle)/sigmaX[i,j])**2 +  (sin(angle) / sigmaY[i,j])**2)) for j in range(nPSF)], axis=0) / data
        return v
    return g2d_datanorm

## # ==========================================================================
## ##------ Find the best set of amplitudes for fixed q,sigma,pa ----- ##
## ##------ This is a linear bounded solution using BVLS
## # ==========================================================================
## def optimise_twodgaussian_amp_bvls(pars, parPSF=_default_parPSF, r=None, theta=None, data=None) :
##     """
##     Returns the best set of amplitude for a given set of q,sigma,pa
##     for a set of N 2D Gaussian functions
##     The function returns the result of the BVLS solving by LLSP (openopt)
##     pars  : input parameters including q, sigma, pa (in degrees)
##     r     : radii
##     theta : angle for each point in radians
##     data  : data to fit
##        data, theta and r should have the same size
##     """
## 
##     pars = pars.ravel()
##     ngauss = len(pars) / 3 
## 
##     ## First get the normalised values from the gaussians
##     ## We normalised this also to 1/data to have a sum = 1
##     nGnorm = _n_centred_twodgaussian_Datanorm(pars=pars, parPSF=parPSF)(r,theta,data)
## 
##     ## This is the vector we wish to get as close as possible
##     ## The equation being : Sum_n In * (G2D_n) = 1.0
##     ##                   or       I  x    G    = d
##     d = np.ones(np.size(r), dtype=floatFit)
## 
##     ## Lower and upper bounds (only positiveness)
##     lb = np.zeros(ngauss)
##     ub = lb + np.inf
## 
##     ## Set up LLSP with the parameters and data (no print)
##     parBVLS = LLSP(nGnorm, d, lb=lb, ub=ub, iprint=-1)
##     ## Return the solution
##     sol_bvls = parBVLS.solve('bvls')
##     del parBVLS
##     return nGnorm, sol_bvls.xf

# ==========================================================================
##------ Find the best set of amplitudes for fixed q,sigma,pa ----- ##
##------ This is a linear bounded solution using NNLS
# ==========================================================================
def optimise_twodgaussian_amp_nnls(pars, parPSF=_default_parPSF, r=None, theta=None, data=None) :
    """
    Returns the best set of amplitude for a given set of q,sigma,pa
    for a set of N 2D Gaussian functions
    The function returns the result of the NNLS solving (scipy)
    pars  : input parameters including q, sigma, pa (in degrees)
    r     : radii
    theta : angle for each point in radians
    data  : data to fit
       data, theta and r should have the same size
    """

    pars = pars.ravel()

    ## First get the normalised values from the gaussians
    ## We normalised this also to 1/data to have a sum = 1
    nGnorm = _n_centred_twodgaussian_Datanorm(pars=pars, parPSF=parPSF)(r,theta,data)

    ## This is the vector we wish to get as close as possible
    ## The equation being : Sum_n In * (G2D_n) = 1.0
    ##                   or       I  x    G    = d
    d = np.ones(np.size(r), dtype=floatFit)

    ## Use NNLS to solve the linear bounded (0) equations
    sol_nnls, norm_nnls = nnls(nGnorm, d, maxiter=faciter_nnls*nGnorm.shape[1])
    return nGnorm, sol_nnls
# ==========================================================================

def optimise_twodgaussian_amp_lstsq(pars, parPSF=_default_parPSF, r=None, theta=None, data=None) :
    """
    Returns the best set of amplitude for a given set of q,sigma,pa
    for a set of N 2D Gaussian functions
    The function returns the result of the linear algorithm solving lstsq (scipy)
    pars  : input parameters including q, sigma, pa (in degrees)
    r     : radii
    theta : angle for each point in radians
    data  : data to fit
       data, theta and r should have the same size
    """

    pars = pars.ravel()

    ## First get the normalised values from the gaussians
    ## We normalised this also to 1/data to have a sum = 1
    nGnorm = _n_centred_twodgaussian_Datanorm(pars=pars, parPSF=parPSF)(r,theta,data)

    ## This is the vector we wish to get as close as possible
    ## The equation being : Sum_n In * (G2D_n) = 1.0
    ##                   or       I  x    G    = d
    d = np.ones(np.size(r), dtype=floatFit)

    ## Use NNLS to solve the linear bounded (0) equations
    sol_lstsq, res_lstsq, rank, s = lstsq(nGnorm, d, cond=cond_lstsq)
    return nGnorm, sol_lstsq

################################################################################
## Find the best set of N 2D Gaussians whose sums to the input data
## 
## This is a non-linear least squares problem, 
##   split into a non-linear one (q,sigma,pa) - solved with lmfit or mpfit
##   and a linear one on the amplitude  - solved with NNLS or BVLS
## 
################################################################################

## Default values
default_minpars=[-3, 0.05, -180.]
default_maxpars=[3.0, 1.0, 180.]
default_fixed=[False,False,True]
default_limitedmin=[True,True,True]
default_limitedmax=[True,True,True]
dic_linear_methods = {"nnls": optimise_twodgaussian_amp_nnls, 
                      "lstsq": optimise_twodgaussian_amp_lstsq}
faciter_nnls = 10
cond_lstsq = 0.01

def set_parameters_and_default(parlist, default, ngauss) :
    """
    Set up the parameters given a default and a number of gaussians
    Input is 

    parlist : the input parameters you wish to set
    default : the default when needed for one gaussian
    ngauss : the number of Gaussians
    """
    ## If the len of the parlist is the good one, then just keep it
    if parlist is None : parlist = []
    if len(parlist) != 3*ngauss:

        ## If the length is 3, then it is just to be replicated
        ## ngauss times
        if len(parlist) == 3: 
            parlist = parlist * ngauss 
        ## Otherwise you need to use the default times the number of Gaussians
        elif len(default) == 3*ngauss :
            parlist[:] = default
        elif len(default) == 3 :
            parlist[:] = default * ngauss
        else :
            print("ERROR: wrong passing of arguments in set_parameters_and_default")
            print("ERROR: the default has not the right size ", len(default))

    return parlist

##----------- Regrow the parameters when samePA is true ---------##
def regrow_PA(p) :
    pall = np.zeros(((len(p)-1)//2, 3), floatFit) 
    pall[:,0] = p[:-1:2]
    pall[:,1] = p[1::2]
    pall[:,2].fill(p[-1])
    return pall.ravel()

##----------- Shrink the parameters when samePA is true ---------##
def shrink_PA(parinfo) :
    indpop = 2
    ngauss = len(parinfo) // 3
    for ii in range(ngauss-1) :
        parinfo.pop(indpop)
        indpop += 2
    return parinfo

## -----------------------------------------------------------------------------------------
## Return the difference between a model and the data WITH ERRORS
## -----------------------------------------------------------------------------------------
def fitn2dgauss_residuals_err1(err, nGnorm, Iamp) :
    return ((1.0 - nGnorm.dot(Iamp)) / err.ravel())
## -----------------------------------------------------------------------------------------
## Return the difference between a model and the data
## -----------------------------------------------------------------------------------------
def fitn2dgauss_residuals1(nGnorm, Iamp) :
    return (1.0 - nGnorm.dot(Iamp))
## -----------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------
## Return the difference between a model and the data WITH ERRORS
## -----------------------------------------------------------------------------------------
# def fitn2dgauss_residuals_err(par, parPSF, r, theta, data, err) :
#     return ((data.ravel() - _n_centred_twodgaussian_Inorm(pars=par, parPSF=parPSF)(r, theta))/(data.ravel() * err.ravel()))
## -----------------------------------------------------------------------------------------
## Return the difference between a model and the data WITH ERRORS
## -----------------------------------------------------------------------------------------
def fitn2dgauss_chi2_err(par, parPSF, r, theta, data, err) :
    return np.sum(((data.ravel() - _n_centred_twodgaussian_Inorm(pars=par, parPSF=parPSF)(r, theta))/(data.ravel() * err.ravel()))**2)
## -----------------------------------------------------------------------------------------
## Return the difference between a model and the data
## -----------------------------------------------------------------------------------------
# def fitn2dgauss_residuals(par, parPSF, r, theta, data) :
#     return (1.0 - _n_centred_twodgaussian_Inorm(pars=par, parPSF=parPSF)(r, theta) / data.ravel())
## -----------------------------------------------------------------------------------------
## Return the difference between a model and the data
## -----------------------------------------------------------------------------------------
# def fitn2dgauss_chi2(par, parPSF, r, theta, data) :
#     return np.sum((1.0 - _n_centred_twodgaussian_Inorm(pars=par, parPSF=parPSF)(r, theta)/data.ravel())**2)

## -------------------------------------------------------------------------------------
## Printing routine for mpfit
## -------------------------------------------------------------------------------------
class _mpfitprint(object):
    def __init__(self):
        self.pars = []
        self.chi2 = []
        self.parinfo = []

    def __call__(self, mpfitfun, p, iter, fnorm, functkw=None, parinfo=None, quiet=0, dof=None):
        self.chi2.append(fnorm)
        self.pars.append(p)
        self.parinfo.append(parinfo)
        print("Chi2 = ", fnorm)
## -------------------------------------------------------------------------------------

SLOPE_outer = 2.0
################################################################################
# MPFIT version of the multigauss 2D fitting routine
################################################################################
def multi_2dgauss_mpfit(xax, yax, data, ngauss=1, err=None, params=None, paramsPSF=None,
        fixed=None, limitedmin=None, limitedmax=None, minpars=None, 
        maxpars=None, force_Sigma_bounds=True, factor_Chi2=1.01, verbose=True, veryverbose=False, 
        linear_method="nnls", default_q=0.3, default_PA=0.0, samePA=True, minSigma=None, maxSigma=None, 
        mpfitprint=_mpfitprint(), **fcnargs):
    """
    An improvement on gaussfit.  Lets you fit multiple 2D gaussians.

    Inputs:
       xax - x axis
       yax - y axis
       data - count axis
       ngauss - How many gaussians to fit?  Default 1 
       err - error corresponding to data

     These parameters need to have length = 3*ngauss.  If ngauss > 1 and length = 3, they will
     be replicated ngauss times, otherwise they will be reset to defaults:
       params - Fit parameters: [width, axis ratio, pa] * ngauss
              If len(params) % 3 == 0, ngauss will be set to len(params) / 3
       fixed - Is parameter fixed?
       limitedmin/minpars - set lower limits on each parameter (default: width>0)
       limitedmax/maxpars - set upper limits on each parameter

       force_Sigma_bounds: force the Sigmas to be within the radii range with some margin
                           default to True
       factor_Chi2 : if one Gaussian contribute less than (factor-1) to the Chi2, we remove it
                     If set to 1, it means only zero Gaussians will be removed
                     If set to default=1.01, it means any Gaussian contributing to less than 1% will be
                     removed

       minSigma, maxSigma: default to None but can be set for bounds for Sigma

       linearmethod: Method used to solve the linear part of the problem
                     One method is implemented (bvls has been discarded)
                         "nnls" -> NNLS (default, included in scipy)

       **fcnargs - Will be passed to MPFIT, you can for example use: 
                   xtol, gtol, ftol, quiet

       verbose - self-explanatory
       veryverbose - self-explanatory

    Returns:
       Fit parameters
       Model
       Fit errors
       chi2
    """
    import copy

    ## Set up some default parameters for mpfit
    if "xtol" not in list(fcnargs.keys()) : fcnargs["xtol"] = 1.e-7
    if "gtol" not in list(fcnargs.keys()) : fcnargs["gtol"] = 1.e-7
    if "ftol" not in list(fcnargs.keys()) : fcnargs["ftol"] = 1.e-7
    if "quiet" not in list(fcnargs.keys()) : fcnargs["quiet"] = True

    ## Checking the method used for the linear part
    linear_method == "nnls"

    ## Checking if the linear_method is implemented
    if linear_method.lower() not in list(dic_linear_methods.keys()):
        print("ERROR: you should use one of the following linear_method: ", list(dic_linear_methods.keys()))
        return 0, 0, 0, 0

    f_Get_Iamp = dic_linear_methods[linear_method.lower()]
    ## If no coordinates is given, create them
    if xax is None:
        xax = np.arange(len(data))
    if yax is None:
        yax = np.arange(len(data))

    if not isinstance(xax,np.ndarray): 
        xax = np.asarray(xax)
    if not isinstance(yax,np.ndarray): 
        yax = np.asarray(yax)
    if not isinstance(data,np.ndarray): 
        data = np.asarray(data)
    xax = xax.ravel()
    yax = yax.ravel()
    datashape = data.shape
    data = data.ravel()

    ## Polar coordinates
    r, theta = convert_xy_to_polar(xax, yax)

    selxy = (xax != 0) & (yax != 0)
    rin = sqrt(xax[selxy]**2+yax[selxy]**2)
    if minSigma is None : minSigma = np.min(rin)
    if maxSigma is None : maxSigma = np.max(rin) / sqrt(SLOPE_outer)
    lminSigma = np.log10(minSigma)
    lmaxSigma = np.log10(maxSigma)
    DlSigma = 0.5 * (lmaxSigma - lminSigma) / ngauss

    if isinstance(params,np.ndarray): params=params.tolist()
    if params is not None : 
        if len(params) != ngauss and (len(params) / 3) > ngauss: 
            ngauss = len(params) / 3 
            if verbose :
                print("WARNING: Your input parameters do not fit the Number of input Gaussians")
                print("WARNING: the new number of input Gaussians is: ", ngauss)

    ## Extracting the parameters for the PSF and normalising the Imax for integral = 1
    if paramsPSF is None : 
        paramsPSF = _default_parPSF
    paramsPSF = norm_PSFParam(paramsPSF)

    ## if no input parameters are given, we set up the guess as a log spaced sigma between min and max
    default_params = np.concatenate((np.log10(np.logspace(lminSigma + DlSigma, lmaxSigma - DlSigma, ngauss)), \
            np.array([default_q]*ngauss), np.array([default_PA]*ngauss))).reshape(3,ngauss).transpose().ravel()

    newdefault_minpars = copy.copy(default_minpars)
    newdefault_maxpars = copy.copy(default_maxpars)
    if force_Sigma_bounds :
        newdefault_minpars[0] = lminSigma
        newdefault_maxpars[0] = lmaxSigma
    else :
        newdefault_minpars[0] = lminSigma - np.log10(2.)
        newdefault_maxpars[0] = lmaxSigma + np.log10(2.)

    ## Set up the default parameters if needed
    params = set_parameters_and_default(params, default_params, ngauss)
    fixed = set_parameters_and_default(fixed, default_fixed, ngauss)
    limitedmin = set_parameters_and_default(limitedmin, default_limitedmin, ngauss)
    limitedmax = set_parameters_and_default(limitedmax, default_limitedmax, ngauss)
    minpars = set_parameters_and_default(minpars, newdefault_minpars, ngauss)
    maxpars = set_parameters_and_default(maxpars, newdefault_maxpars, ngauss)

    ## -------------------------------------------------------------------------------------
    ## mpfit function which returns the residual from the best fit N 2D Gaussians
    ## Parameters are just sigma,q,pa - the amplitudes are optimised at each step
    ## Two versions are available depending on whether BVLS or NNLS is used (and available)
    ## -------------------------------------------------------------------------------------
    def mpfitfun(p, parPSF, fjac=None, r=None, theta=None, err=None, data=None, f_Get_Iamp=None, samePA=False):
        if samePA : p = regrow_PA(p)
        nGnorm, Iamp = f_Get_Iamp(p, parPSF, r, theta, data)
        if err is None : 
            return [0,fitn2dgauss_residuals1(nGnorm, Iamp)]
        else :
            return [0,fitn2dgauss_residuals_err1(err, nGnorm, Iamp)]
#        newp = (np.vstack((Iamp, p.reshape(ngauss,3).transpose()))).transpose()
#        if err is None : 
#            return [0,fitn2dgauss_residuals(newp, parPSF, r, theta, data)]
#        else :
#            return [0,fitn2dgauss_residuals_err(newp, parPSF, r, theta, data, err)]
    ## -------------------------------------------------------------------------------------

    ## Information about the parameters
    if verbose :
        print("--------------------------------------")
        print("GUESS:      Sig         Q         PA")
        print("--------------------------------------")
        for i in range(ngauss) :
            print("GAUSS %02d: %8.3e  %8.3f  %8.3f"%(i+1, 10**(params[3*i]), params[3*i+1], params[3*i+2]))
    print("--------------------------------------")

    ## Information about the parameters
    parnames = {0:"LOGSIGMA",1:"AXIS RATIO",2:"POSITION ANGLE"}
    parinfo = [ {'n':ii, 'value':params[ii], 'limits':[minpars[ii],maxpars[ii]], 
        'limited':[limitedmin[ii],limitedmax[ii]], 'fixed':fixed[ii], 
        'parname':parnames[ii%3]+str(ii/3+1)} for ii in range(len(params)) ]

    ## If samePA we remove all PA parameters except the last one
    ## We could use the 'tied' approach but we prefer setting up just one parameter
    if samePA : parinfo = shrink_PA(parinfo)

    ## Fit with mpfit of q, sigma, pa on xax, yax, and data (+err)
    fa = {'parPSF':paramsPSF, 'r': r, 'theta': theta, 'data': data, 'err':err, 'f_Get_Iamp':f_Get_Iamp, 'samePA':samePA}

    result = mpfit(mpfitfun, functkw=fa, iterfunct=mpfitprint, nprint=10, parinfo=parinfo, **fcnargs) 

    ## Getting these best fit values into the dictionnary
    if samePA : result.params = regrow_PA(result.params)
    bestparinfo = [ {'n':ii, 'value':result.params[ii], 'limits':[minpars[ii],maxpars[ii]], 
        'limited':[limitedmin[ii],limitedmax[ii]], 'fixed':fixed[ii], 
        'parname':parnames[ii%3]+str(ii/3)} for ii in range(len(result.params)) ]

    ## Recompute the best amplitudes to output the right parameters
    ## And renormalising them
    nGnorm, Iamp = f_Get_Iamp(result.params, paramsPSF, r, theta, data)
#    Ibestpar_array = (np.vstack((Iamp, result.params.reshape(ngauss,3).transpose()))).transpose()
    bestpar_array = result.params.reshape(ngauss,3)

    ## Getting rid of the non-relevant Gaussians
    ## If parameters factor_Chi2 is set we use it as a threshold to remove gaussians
    ## Otherwise we just remove the zeros
    if err is None : nerr = np.ones_like(data)
    else : nerr =  err
    ## First get the Chi2 from this round
#    bestChi2 = fitn2dgauss_chi2_err(Ibestpar_array, paramsPSF, r, theta, data, nerr)
    bestChi2 = np.sum(fitn2dgauss_residuals1(nGnorm, Iamp)**2)
    result.ind = list(range(ngauss))

    if factor_Chi2 > 1.:
        k = 0
        Removed_Gaussians = []
        for i in range(ngauss) :
            ## Derive the Chi2 WITHOUT the ith Gaussian
            new_nGnorm, new_Iamp = f_Get_Iamp(np.delete(bestpar_array, i, 0), paramsPSF, r, theta, data)
            newChi2 = np.sum(fitn2dgauss_residuals1(new_nGnorm, new_Iamp)**2)
#            newChi2 = fitn2dgauss_chi2_err(np.delete(Ibestpar_array, i, 0), paramsPSF, r, theta, data, nerr)
            ## If this Chi2 is smaller than factor_Chi2 times the best value, then remove
            ## It just means that Gaussian is not an important contributor
            if newChi2 <= factor_Chi2 * bestChi2 :
                val = bestparinfo.pop(3*k)
                val = bestparinfo.pop(3*k)
                val = bestparinfo.pop(3*k)
                result.ind.pop(k)
                Removed_Gaussians.append(i+1)
            else : k += 1

        if veryverbose :
            if len(Removed_Gaussians) != 0 :
                print("WARNING Removed Gaussians ", Removed_Gaussians)
                print("WARNING: (not contributing enough to the fit)")
        ngauss = len(result.ind)

        if samePA : bestparinfo = shrink_PA(bestparinfo)
        ## New minimisation after removing all the non relevant Gaussians
        newresult = mpfit(mpfitfun, functkw=fa, iterfunct=mpfitprint, nprint=10, parinfo=bestparinfo, **fcnargs) 
        if samePA : newresult.params = regrow_PA(newresult.params)
    else:
    # If factor_Chi2 is set to 0 do not remove Gaussians
        newresult = result

    newresult.ind = list(range(ngauss))
    bestfit_params = newresult.params.reshape(ngauss, 3)

    ## We add the Amplitudes to the array and renormalise them
    nGnorm, Iamp = f_Get_Iamp(bestfit_params, paramsPSF, r, theta, data)
    Ibestfit_params = (np.vstack((Iamp, bestfit_params.transpose()))).transpose()
    ## Going back to sigma from logsigma
    Ibestfit_params[:,1] = 10**(Ibestfit_params[:,1])
    Ibestfit_params[:,0] /= (2.0 * Ibestfit_params[:,1]**2 * Ibestfit_params[:,2] * pi)
    ## And we sort them with Sigma
    Ibestfit_params = Ibestfit_params[Ibestfit_params[:,1].argsort()]

    if newresult.status == 0:
        raise Exception(newresult.errmsg)

    if verbose :
        print("==================================================")
        print("FIT:      Imax       Sig         Q           PA")
        print("==================================================")
        for i in range(ngauss) :
            print("GAUSS %02d: %8.3e  %8.3f  %8.3f  %8.3f"%(i+1, Ibestfit_params[i,0], Ibestfit_params[i,1], Ibestfit_params[i,2], Ibestfit_params[i,3]))

        print("Chi2: ",newresult.fnorm," Reduced Chi2: ",newresult.fnorm/len(data))

    return Ibestfit_params, newresult, n_centred_twodgaussian_Imax(pars=Ibestfit_params, parPSF=paramsPSF)(r, theta).reshape(datashape)
   
################################################################################
# LMFIT version of the multigauss 2D fitting routine
################################################################################
##------ Transform the Parameters class into a single array ---- ##
def extract_mult2dG_params(Params) :
    ind = Params.ind
    ngauss = len(ind)
    p = np.zeros((ngauss, 3), floatFit)
    for i in range(ngauss) :
        p[i,0] = Params['logSigma%02d'%(ind[i]+1)].value
        p[i,1] =  Params['Q%02d'%(ind[i]+1)].value
        if Params['PA%02d'%(ind[i]+1)].value is None :
            p[i,2] = Params[Params['PA%02d'%(ind[i]+1)].expr].value
        else :
            p[i,2] =  Params['PA%02d'%(ind[i]+1)].value
    return p

##------ Reimpose the same PA on all the parameters -----##
def Set_SamePA_params(Params, ngauss, valuePA, minPA, maxPA, varyPA) :
    ## Now we reset the Parameters according to the PA
    ## So we look for the first Gaussian PA
    FirstParamName = 'PA%02d'%(Params.ind[0]+1)
    FirstParam = Params[FirstParamName]
    # And we save its value
    FirstParam.value, FirstParam.min, FirstParam.max, FirstParam.vary = valuePA, minPA, maxPA, varyPA
    FirstParam.expr = None

    # Now we set the others to be the same than the first one (using the "expr" option)
    for i in range(1, ngauss) :
        currentParamName = 'PA%02d'%(Params.ind[i]+1)
        currentParam = Params[currentParamName]
        currentParam.value, currentParam.min, currentParam.max, currentParam.vary = None, None, None, False
        currentParam.expr = FirstParamName

    # Returning the updated Parameters
    return Params

##------------- Printing option for lmfit -------------- ##
class lmfit_iprint(object):
    def __init__(self):
        self.chi2 = []
        self.pars = []

    def __call__(self, res, myinput, pars):
        """ Printing function for the iteration in lmfit
        """
        if (myinput.iprint > 0) & (myinput.verbose):
            if myinput.aprint == myinput.iprint:
                chi2 = ((res*res).sum())
                self.chi2.append(chi2)
                self.pars.append(pars)
                print("Chi2 = %g" % chi2)
                myinput.aprint = 0
            myinput.aprint += 1

def multi_2dgauss_lmfit(xax, yax, data, ngauss=1, err=None, params=None, paramsPSF=None,
        fixed=None, limitedmin=None, limitedmax=None, minpars=None, 
        maxpars=None, force_Sigma_bounds=True, factor_Chi2=1.01, iprint=50, lmfit_method='leastsq',
        verbose=True, veryverbose=False, linear_method="nnls", default_q=0.3, default_PA=0.0, 
        samePA=True, sameQ=False, minSigma=None, maxSigma=None, lmfit_iprint=lmfit_iprint(), **fcnargs):
    """
    An improvement on gaussfit.  Lets you fit multiple 2D gaussians.

    Inputs:
       xax - x axis
       yax - y axis
       data - count axis
       ngauss - How many gaussians to fit?  Default 1 
       err - error corresponding to data

     These parameters need to have the same length. 
        It should by default be 3*ngauss.  
        If ngauss > 1 and length = 3, they will be replicated ngauss times, 
        otherwise they will be reset to defaults:

       params - Fit parameters: [width, axis ratio, pa] * ngauss
              If len(params) % 3 == 0, ngauss will be set to len(params) / 3
       fixed - Is parameter fixed?
       limitedmin/minpars - set lower limits on each parameter (default: width>0)
       limitedmax/maxpars - set upper limits on each parameter

       force_Sigma_bounds: force the Sigmas to be within the radii range with some margin
                           default to True
       factor_Chi2 : if one Gaussian contribute less than (factor-1) to the Chi2, we remove it
                     If set to 1, it means only zero Gaussians will be removed
                     If set to default=1.01, it means any Gaussian contributing to less than 1% will be
                     removed
       minSigma, maxSigma: default to None but can be set for bounds for Sigma
       samePA : by default set to True. In that case, only one PA value is used as a free parameter
                (all Gaussians will share the same PA)
       sameQ: by default set to False. In that case, only one Axis ratio value is used as a free parameter
                (all Gaussians will share the same axis ratio)


       lmfit_method : method to pass on to lmfit ('leastsq', 'lbfgsb', 'anneal')
                      Default is leastsq (most efficient for the problem)

       linearmethod: Method used to solve the linear part of the problem
                     One method is implemented (bvls is discarded): 
                         "nnls" -> NNLS (default, included in scipy)

       **fcnargs - dictionary which will be passed to LMFIT, you can for example use: 
                   xtol , gtol, ftol,  etc

       iprint - if > 0, print every iprint iterations of lmfit. default is 50
       verbose - self-explanatory
       veryverbose - self-explanatory

    Returns:
       Fit parameters
       Model
       Fit errors
       chi2
    """
    import copy

    ## Default values
    lmfit_methods = ['leastsq', 'lbfgsb', 'anneal']

    ## Method check
    if lmfit_method not in lmfit_methods :
        print("ERROR: method must be one of the three following methods : ", lmfit_methods)

    ## Setting up epsfcn if not forced by the user
    ## Removing epsfcn to get the default machine precision
    ## if "epsfcn" not in fcnargs.keys() : fcnargs["epsfcn"] = 0.01
    if "xtol" not in list(fcnargs.keys()) : fcnargs["xtol"] = 1.e-7
    if "gtol" not in list(fcnargs.keys()) : fcnargs["gtol"] = 1.e-7
    if "ftol" not in list(fcnargs.keys()) : fcnargs["ftol"] = 1.e-7

    ## Checking the method used for the linear part
    linear_method == "nnls"

    ## Checking if the linear_method is implemented
    if linear_method.lower() not in list(dic_linear_methods.keys()):
        print("ERROR: you should use one of the following linear_method: ", list(dic_linear_methods.keys()))
        return 0, 0, 0

    f_Get_Iamp = dic_linear_methods[linear_method.lower()]
    ## If no coordinates is given, create them
    if xax is None:
        xax = np.arange(len(data))
    if yax is None:
        yax = np.arange(len(data))

    if not isinstance(xax,np.ndarray): 
        xax = np.asarray(xax)
    if not isinstance(yax,np.ndarray): 
        yax = np.asarray(yax)
    if not isinstance(data,np.ndarray): 
        data = np.asarray(data)
    xax = xax.ravel()
    yax = yax.ravel()
    datashape = data.shape
    data = data.ravel()

    ## Polar coordinates
    r, theta = convert_xy_to_polar(xax, yax)

    selxy = (xax != 0) & (yax != 0)
    rin = sqrt(xax[selxy]**2+yax[selxy]**2)

    if minSigma is None : minSigma = np.min(rin)
    if maxSigma is None : maxSigma = np.max(rin) / sqrt(SLOPE_outer)
    lminSigma = np.log10(minSigma)
    lmaxSigma = np.log10(maxSigma)
    DlSigma = 0.5 * (lmaxSigma - lminSigma) / ngauss

    if isinstance(params,np.ndarray): params=params.tolist()
    if params is not None : 
        if len(params) != ngauss and (len(params) / 3) > ngauss: 
            ngauss = len(params) / 3 
            if verbose :
                print("WARNING: Your input parameters do not fit the Number of input Gaussians")
                print("WARNING: the new number of input Gaussians is: ", ngauss)

    ## Extracting the parameters for the PSF and normalising the Imax for integral = 1
    if paramsPSF is None : 
        paramsPSF = _default_parPSF
    paramsPSF = norm_PSFParam(paramsPSF)

    ## if no input parameters are given, we set up the guess as a log spaced sigma between min and max
    default_params = np.concatenate((np.log10(np.logspace(lminSigma + DlSigma, lmaxSigma - DlSigma, ngauss)), \
            np.array([default_q]*ngauss), np.array([default_PA]*ngauss))).reshape(3,ngauss).transpose().ravel()

    newdefault_minpars = copy.copy(default_minpars)
    newdefault_maxpars = copy.copy(default_maxpars)
    if force_Sigma_bounds :
        newdefault_minpars[0] = lminSigma
        newdefault_maxpars[0] = lmaxSigma
    else :
        newdefault_minpars[0] = lminSigma - np.log10(2.)
        newdefault_maxpars[0] = lmaxSigma + np.log10(2.)

    ## Set up the default parameters if needed
    params = set_parameters_and_default(params, default_params, ngauss)
    fixed = set_parameters_and_default(fixed, default_fixed, ngauss)
    limitedmin = set_parameters_and_default(limitedmin, default_limitedmin, ngauss)
    limitedmax = set_parameters_and_default(limitedmax, default_limitedmax, ngauss)
    minpars = set_parameters_and_default(minpars, newdefault_minpars, ngauss)
    maxpars = set_parameters_and_default(maxpars, newdefault_maxpars, ngauss)

    class input_residuals() :
        def __init__(self, iprint, verbose) :
            self.iprint = iprint
            self.verbose = verbose
            self.aprint = 0

    ## -----------------------------------------------------------------------------------------
    ## lmfit function which returns the residual from the best fit N 2D Gaussians
    ## Parameters are just sigma,q,pa - the amplitudes are optimised at each step
    ## -----------------------------------------------------------------------------------------
    def opt_lmfit(pars, parPSF, myinput=None, r=None, theta=None, err=None, data=None, f_Get_Iamp=None):
        """ Provide the residuals for the lmfit minimiser
            for a Multi 1D gaussian
        """

        # We retrieve the parameters
        pars_array = extract_mult2dG_params(pars)

        ## Derive the Normalised Gaussians for this set of parameters
        nGnorm, Iamp = f_Get_Iamp(pars_array, parPSF, r, theta, data)
        if err is None :
            res = fitn2dgauss_residuals1(nGnorm, Iamp)
        else :
            res = fitn2dgauss_residuals_err1(err, nGnorm, Iamp)
#        newp = (np.vstack((Iamp, pars_array.transpose()))).transpose()
#        if err is None :
#            res = fitn2dgauss_residuals(newp, parPSF, r, theta, data)
#        else :
#            res = fitn2dgauss_residuals_err(newp, parPSF, r, theta, data, err)
        lmfit_iprint(res, myinput, pars)
        return res
    ## -----------------------------------------------------------------------------------------

    ## Information about the parameters
    nameParam = ['logSigma', 'Q', 'PA']
    Lparams = Parameters()
    if verbose :
        print("--------------------------------------")
        print("GUESS:      Sig         Q         PA")
        print("--------------------------------------")
        for i in range(ngauss) :
            print("GAUSS %02d: %8.3e  %8.3f  %8.3f"%(i+1, 10**(params[3*i]), params[3*i+1], params[3*i+2]))
    print("--------------------------------------")

    for i in range(ngauss) :
        Lparams.add(nameParam[0]+"%02d"%(i+1), value=params[3*i], min=minpars[3*i], max=maxpars[3*i], vary= not fixed[3*i])
        Lparams.add(nameParam[1]+"%02d"%(i+1), value=params[3*i+1], min=minpars[3*i+1], max=maxpars[3*i+1], vary= not fixed[3*i+1])
        Lparams.add(nameParam[2]+"%02d"%(i+1), value=params[3*i+2], min=minpars[3*i+2], max=maxpars[3*i+2], vary= not fixed[3*i+2])
    ## Adding indices to follow up the Gaussians we may remove
    Lparams.ind = list(range(ngauss))

    ## Setting the samePA option if True
    ## For this we set up the first PA to the default and 
    ## then use "expr" to say that all other PA are equal to the first one
    if samePA:
        Lparams = Set_SamePA_params(Lparams, ngauss, params[2], minpars[2], maxpars[2], not fixed[2])

    if veryverbose :
        for i in range(ngauss) :
            print("GAUSS %02d: %8.3e  %8.3f  %8.3f"%(i+1, 10**(params[3*i]), params[3*i+1], params[3*i+2]))
        if samePA:
            print("WARNING: All PAs will be forced to one single value")
    print("--------------------------------------")

    ## Setting up the printing option
    myinput = input_residuals(iprint, verbose)

    ####################################
    ## Doing the minimisation with lmfit
    ####################################
    if verbose: 
        print("------ Starting the minimisation -------")
    result = minimize(opt_lmfit, Lparams, method=lmfit_method, args=(paramsPSF, myinput, r, theta, err, data, f_Get_Iamp), **fcnargs)
    ## Remove the Null Gaussians
    result.params.ind = list(range(ngauss))
    ngauss, Ind_ZGauss = Remove_Zero_2DGaussians(ngauss, nameParam, result, paramsPSF, r, theta, data, err, factor_Chi2,
            f_Get_Iamp, niter=1, verbose=veryverbose, samePA=samePA) 

    ## Recall the Minimizer function for a second iteration to get the new chi2 etc
    newresult = minimize(opt_lmfit, result.params, method=lmfit_method, args=(paramsPSF, myinput, r, theta, err, data, f_Get_Iamp), **fcnargs)
    ## Remove the Null Gaussians
    newresult.params.ind = result.params.ind 
    ngauss, Ind_ZGauss = Remove_Zero_2DGaussians(ngauss, nameParam, newresult, paramsPSF, r, theta, data, err, factor_Chi2,
            f_Get_Iamp, niter=2, verbose=veryverbose, samePA=samePA) 

    ## We add the Amplitudes to the array and renormalise them
    bestfit_params = extract_mult2dG_params(newresult.params)
    nGnorm, Iamp = f_Get_Iamp(bestfit_params, paramsPSF, r, theta, data)
    Ibestfit_params = (np.vstack((Iamp, bestfit_params.transpose()))).transpose()
    ## Changing the parameters back to Sigma
    Ibestfit_params[:,1] = 10**(Ibestfit_params[:,1])
    Ibestfit_params[:,0] /= (2.0 * Ibestfit_params[:,1]**2 * Ibestfit_params[:,2] * pi)
    ## And we sort them with Sigma
    Ibestfit_params = Ibestfit_params[Ibestfit_params[:,1].argsort()]

    if verbose :
        print("==================================================")
        print("FIT:      Imax       Sig         Q           PA")
        print("==================================================")
        for i in range(ngauss) :
            print("GAUSS %02d: %8.3e  %8.3f  %8.3f  %8.3f"%(i+1, Ibestfit_params[i,0], Ibestfit_params[i,1], Ibestfit_params[i,2], Ibestfit_params[i,3]))

        print("Chi2: ",newresult.chisqr," Reduced Chi2: ",newresult.redchi)

    return Ibestfit_params, newresult, n_centred_twodgaussian_Imax(pars=Ibestfit_params, parPSF=paramsPSF)(r, theta).reshape(datashape)

def Remove_Zero_2DGaussians(ngauss, nameParam, result, parPSF, r, theta, data, err, factor_Chi2, f_Get_Iamp, niter=1, verbose=False, samePA=True) :
    
    ## Recompute the best amplitudes and remove the ones that are zeros
    bestpar_array = extract_mult2dG_params(result.params)
    nGnorm, Iamp = f_Get_Iamp(bestpar_array, parPSF, r, theta, data)
#    ## New array including the Imax values
#    Ibestpar_array = (np.vstack((Iamp, bestpar_array.transpose()))).transpose()

    ## Getting rid of the non-relevant Gaussians
    ## If parameters factor_Chi2 is set we use it as a threshold to remove gaussians
    ## Otherwise we just remove the zeros
    if err is None : nerr = np.ones_like(data)
    else : nerr =  err
    ## First get the Chi2 from this round
    bestChi2 = np.sum(fitn2dgauss_residuals1(nGnorm, Iamp)**2)
#    bestChi2 = fitn2dgauss_chi2_err(Ibestpar_array, parPSF, r, theta, data, nerr)

    k = 0
    Removed_Gaussians = []
    ## If SamePA we need to save the first set of parameters for the PA
    if samePA :
        FirstParamName = nameParam[2]+'%02d'%(result.params.ind[0]+1)
        FirstParam = result.params[FirstParamName]
        valuePA, minPA, maxPA, varyPA = FirstParam.value, FirstParam.min, FirstParam.max, FirstParam.vary

    for i in range(ngauss) :
        ## Derive the Chi2 WITHOUT the ith Gaussian
        new_nGnorm, new_Iamp = f_Get_Iamp(np.delete(bestpar_array, i, 0), parPSF, r, theta, data)
        newChi2 = np.sum(fitn2dgauss_residuals1(new_nGnorm, new_Iamp)**2)
#        newChi2 = fitn2dgauss_chi2_err(np.delete(Ibestpar_array, i, 0), parPSF, r, theta, data, nerr)
        ## If this Chi2 is smaller than factor_Chi2 times the best value, then remove
        ## It just means that Gaussian is not an important contributor
        if newChi2 <= factor_Chi2 * bestChi2 :
            if verbose : 
                print("Removing Gaussian ", i)
            result.params.pop(nameParam[0]+'%02d'%(result.params.ind[k]+1))
            result.params.pop(nameParam[1]+'%02d'%(result.params.ind[k]+1))
            result.params.pop(nameParam[2]+'%02d'%(result.params.ind[k]+1))
            result.params.ind.pop(k)
            Removed_Gaussians.append(i+1)
        else : k += 1

    New_ngauss = len(result.params.ind)

    ## Setting the samePA option if True
    ## For this we set up the first PA to the default and 
    ## then use "expr" to say that all other PA are equal to the first one
    if samePA:
        result.params = Set_SamePA_params(result.params, New_ngauss, valuePA, minPA, maxPA, varyPA)

    if verbose :
        if len(Removed_Gaussians) != 0 :
            print("WARNING Removed Gaussians (iteration %d) : "%(niter), Removed_Gaussians)
            print("WARNING: (not contributing enough to the fit)")

    return New_ngauss, Removed_Gaussians
