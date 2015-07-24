######################################################################################
# fitn1dgauss.py
# This version is directly inspired from the fitn2dgauss.py from the same package
# Just removing the axis ratio and PA.
#
# VERSION Of fit1dngauss is 
#
# V.0.9.5   19 May 2015 - Typo fixed on residuals
# V.0.9.4   15 Dec 2014 - Typo fixed on sq2pi
# V.0.9.3   17 Sep 2012 - Adaptation from v.0.9.6 of fit2dngauss
# 
###########################################################
# VERSION OF fit2dngauss was:
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
###########################################################
# This version is largely Inspired from gaussfiter.py 
#    written by Adam Ginsburg (adam.ginsburg@colorado.edu or keflavich@gmail.com) 3/17/08)
#    latest version available at http://code.google.com/p/agpy/source/browse/trunk/agpy/gaussfitter.py
#
# Specific development of a N 2D Gaussian fitter # Eric Emsellem (eric.emsellem@eso.org)
#   Basis for the Multi-Gaussian Expansion method as described in Emsellem et al. 1994
#     and subsequent papers on the MGE technique
#   Decoupling the linear and non-linear parts, following the
#   fitting approach proposed by Michele Cappellari (see 2002 paper and available idl routine)
######################################################################################
import numpy as np
from numpy import exp, cos, sin, arctan, sqrt, pi

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
    print "WARNING: Only mpfit is available an optimiser"
    print "WARNING: you may want to install lmfit!"

# # If Openopt is there we can import it and allow the use to use BVLS
# # Otherwise we just use nnls which is faster anyway
# try :
#     # If it works, can use openopt for 
#     # bound-constrained linear least squares (wrapper of BVLS)
#     import openopt
#     from openopt import LLSP
#     Exist_OpenOpt = True
# except ImportError :
#     Exist_OpenOpt = False
#     print "WARNING: Only nnls is available as a linmethod"
# 
# Use scipy nnls to solve the linear part
from scipy.optimize import nnls

## Useful constants
sq2pi = sqrt(2.0 * np.pi)

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

# ==========================================================================
## Returns a function which does the calculation, given a set of parameters,
## of the sum of Gaussians, the I being the maximum intensities (not normed)
# ==========================================================================
def n_centred_onedgaussian_Imax(pars, I=None, sigma=None):
    """
    Returns a function that provides the sum 
    over N 1D gaussians, where N is the length of
    I,sigma *OR* N = len(pars) / 2

    The background "height" is assumed to be zero (you must "baseline" your
    spectrum before fitting)

    pars  - an array with len(pars) = 4n, assuming I, sigma repeated
    I     - amplitude
    sigma - line widths
    """
    pars = pars.ravel()
    if len(pars) % 2 == 0:
        I = pars[::2]
        sigma = pars[1::2]
    elif not(len(I) == len(sigma)):
        raise ValueError("Wrong array lengths! I: %i  sigma: %i " \
                % (len(I), len(sigma)))

    def g1d(x):
        v = np.zeros_like(x)
        for i in xrange(len(sigma)):
            v += I[i] * exp( - 0.5 * (x / sigma[i])**2)
        return v
    return g1d

# ==========================================================================
## Returns a function which does the calculation, given a set of parameters
## of the sum of Gaussians, the I being the NORMED intensities
# ==========================================================================
def _n_centred_onedgaussian_Inorm(pars, I=None, sigma=None):
    """
    Returns a function that provides the sum 
    over N 1D gaussians, where N is the length of
    I,sigma *OR* N = len(pars) / 2

    The background "height" is assumed to be zero (you must "baseline" your
    spectrum before fitting)

    pars  - an array with len(pars) = 4n, assuming I, sigma, q, pa repeated
    I     - amplitude
    sigma - line widths
    """
    pars = pars.ravel()
    if len(pars) % 2 == 0:
        I = pars[::2]
        sigma = pars[1::2]
    elif not(len(I) == len(sigma)):
        raise ValueError("Wrong array lengths! I: %i  sigma: %i" % (len(I), len(sigma)))
    Itot = I / (sq2pi * sigma)

    def g1d(x):
        v = np.zeros_like(x)
        for i in xrange(len(I)):
            v += Itot[i] * exp( - 0.5 * (x / sigma[i])**2)
        return v.ravel()
    return g1d

# ==========================================================================
#---------- Centred (no offset) 1D Gaussian but without flux ---------------##
#---------- Returns a function which provide a set of N normalised Gaussians 
#---------- But NORMALISED BY the data
# ==========================================================================
def _n_centred_onedgaussian_Datanorm(pars, sigma=None):
    """
    Returns a function that provides an array of N NORMALISED 1D gaussians, 
    where N is the length of sigma *OR* N = len(pars). These 
    Gaussians are also normalised by an input data array
    (the returned function takes 2 arguments: r and data)

    pars  - an array with len(pars) = n, assuming sigma repeated
    sigma - line widths
    """
    if pars is not None : 
        sigma = pars

    Ng = len(sigma)
    Itot = 1.0 / (sq2pi * sigma)

    ## Function itself - this will be returned
    ## Using polar cooordinates for simpler handling of PA
    def g1d_datanorm(x, data):
        v = np.zeros((np.size(x),Ng))
        for i in xrange(Ng):
            v[:,i] = (Itot[i] * exp( - 0.5 * (x / sigma[i])**2) / data).ravel()
        return v
    return g1d_datanorm

# ==========================================================================
##------ Find the best set of amplitudes for fixed q,sigma,pa ----- ##
##------ This is a linear bounded solution using BVLS
# ==========================================================================
def _optimise_onedgaussian_amp_bvls(pars, x=None, data=None) :
    """
    Returns the best set of amplitude for a given set of sigma
    for a set of N 1D Gaussian functions
    The function returns the result of the BVLS solving by LLSP (openopt)
    pars  : input parameters including sigma
    x     : radii
    data  : data to fit
       data and r should have the same size
    """

    ngauss = len(pars)

    ## First get the normalised values from the gaussians
    ## We normalised this also to 1/data to have a sum = 1
    nGnorm = _n_centred_onedgaussian_Datanorm(pars=pars)(x,data)

    ## This is the vector we wish to get as close as possible
    ## The equation being : Sum_n In * (G1D_n) = 1.0
    ##                   or       I  x    G    = d
    d = np.ones(np.size(x), dtype=np.float64)

    ## Lower and upper bounds (only positiveness)
    lb = np.zeros(ngauss)
    ub = lb + np.inf

    ## Set up LLSP with the parameters and data (no print)
    parBVLS = LLSP(nGnorm, d, lb=lb, ub=ub, iprint=-1)
    ## Return the solution
    sol_bvls = parBVLS.solve('bvls')
    del parBVLS
    return sol_bvls.xf

# ==========================================================================
##------ Find the best set of amplitudes for fixed q,sigma,pa ----- ##
##------ This is a linear bounded solution using NNLS
# ==========================================================================
def _optimise_onedgaussian_amp_nnls(pars, x=None, data=None) :
    """ Returns the best set of amplitude for a given set of sigma
    for a set of N 1D Gaussian functions

    The function returns the result of the NNLS solving (scipy)

    :param pars: input parameters including sigma
    :param x: radii
    :param data: data to fit
        data and r should have the same size
    """

    ngauss = len(pars) 

    ## First get the normalised values from the gaussians
    ## We normalised this also to 1/data to have a sum = 1
    nGnorm = _n_centred_onedgaussian_Datanorm(pars=pars)(x,data)

    ## This is the vector we wish to get as close as possible
    ## The equation being : Sum_n In * (G1D_n) = 1.0
    ##                   or       I  x    G    = d
    d = np.ones(np.size(x), dtype=np.float64)

    ## Use NNLS to solve the linear bounded (0) equations
    try :
        sol_nnls, norm_nnls = nnls(nGnorm, d)
    except RuntimeError :
        print "Warning: Too many iterations in NNLS"
        return np.zeros(ngauss, dtype=np.float64)
    return sol_nnls

################################################################################
## Find the best set of N 1D Gaussians whose sums to the input data
## 
## This is a non-linear least squares problem, 
##   split into a non-linear one (sigma) - solved with lmfit or mpfit
##   and a linear one on the amplitude  - solved with NNLS or BVLS
## 
################################################################################

## Default values
default_minpars=[0.01]
default_maxpars=[np.inf]
default_fixed=[False]
default_limitedmin=[True]
default_limitedmax=[False]
dic_linear_methods = {"nnls": _optimise_onedgaussian_amp_nnls, "bvls": _optimise_onedgaussian_amp_bvls}

def set_parameters_and_default_1D(parlist, default, ngauss) :
    """
    Set up the parameters given a default and a number of gaussians
    Input is 

    parlist : the input parameters you wish to set
    default : the default when needed for one gaussian
    ngauss : the number of Gaussians
    """
    ## If the len of the parlist is the good one, then just keep it
    if parlist is None : parlist = []
    if len(parlist) != ngauss:

        ## If the length is 3, then it is just to be replicated
        ## ngauss times
        if len(parlist) == 1: 
            parlist = parlist * ngauss 
        ## Otherwise you need to use the default times the number of Gaussians
        elif len(default) == ngauss :
            parlist[:] = default
        elif len(default) == 1 :
            parlist[:] = default * ngauss
        else :
            print "ERROR: wrong passing of arguments in set_parameters_and_default_1D"
            print "ERROR: the default has not the right size ", len(default)

    return parlist

## -----------------------------------------------------------------------------------------
## Return the difference between a model and the data WITH ERRORS
## -----------------------------------------------------------------------------------------
def fitn1dgauss_residuals_err(par, x, data, err) :
    return ((1.0 - _n_centred_onedgaussian_Inorm(pars=par)(x) / data.ravel())/ err.ravel())
## -----------------------------------------------------------------------------------------
## Return the difference between a model and the data WITH ERRORS
## -----------------------------------------------------------------------------------------
def fitn1dgauss_chi2_err(par, x, data, err) :
    return np.sum(((1.0 - _n_centred_onedgaussian_Inorm(pars=par)(x) / data.ravel())/ err.ravel())**2)
## -----------------------------------------------------------------------------------------
## Return the difference between a model and the data
## -----------------------------------------------------------------------------------------
def fitn1dgauss_residuals(par, x, data) :
    return (1.0 - _n_centred_onedgaussian_Inorm(pars=par)(x) / data.ravel())
## ## -----------------------------------------------------------------------------------------
## ## Return the difference between a model and the data
## ## -----------------------------------------------------------------------------------------
## def fitn1dgauss_chi2(par, x, data) :
##     return np.sum((1.0 - _n_centred_onedgaussian_Inorm(pars=par)(x) / data.ravel())**2)
## 
## -------------------------------------------------------------------------------------
## Printing routine for mpfit
## -------------------------------------------------------------------------------------
class mpfitprint(object):
    def __init__(self):
        self.pars = []
        self.chi2 = []
        self.parinfo = []

    def __call__(self, mpfitfun, p, iter, fnorm, functkw=None, parinfo=None, quiet=0, dof=None):
        self.chi2.append(fnorm)
        self.pars.append(p)
        self.parinfo.append(parinfo)
        print "Chi2 = ", fnorm
## -------------------------------------------------------------------------------------

################################################################################
# MPFIT version of the multigauss 1D fitting routine
################################################################################
def multi_1dgauss_mpfit(xax, data, ngauss=1, err=None, params=None, 
        fixed=None, limitedmin=None, limitedmax=None, minpars=None, 
        maxpars=None, force_Sigma_bounds=False, factor_Chi2=1.01, verbose=True,
        veryverbose=False, linear_method="nnls", minSigma=None, maxSigma=None,
        mpfitprint=mpfitprint(), **fcnargs):
    """ An improvement on gaussfit.  Lets you fit multiple 1D gaussians.

    :param   xax: x axis
    :param   data: count axis
    :param   ngauss: How many gaussians to fit?  Default 1 
    :param   err: error corresponding to data
     :param  params: Fit parameters -- [width] * ngauss
         If len(params) == 0, ngauss will be set to len(params) 
         These parameters need to have length = ngauss.  If ngauss > 1 and length = 1 , they will
         be replicated ngauss times, otherwise they will be reset to defaults:
     :param  fixed: Is parameter fixed?
     :param  limitedmin/minpars: set lower limits on each parameter (default: width>0)
     :param  limitedmax/maxpars: set upper limits on each parameter
     :param  force_Sigma_bounds: force the Sigmas to be within the radii range with some margin
         default to True
     :param  factor_Chi2: if one Gaussian contribute less than (factor-1) to the Chi2, we remove it
         If set to 1, it means only zero Gaussians will be removed
         If set to default=1.01, it means any Gaussian contributing to less than 1% will be removed
     :param  minSigma, maxSigma: default to None but can be set for bounds for Sigma
     :param  linearmethod: Method used to solve the linear part of the problem
         Two methods are implemented: 
         "nnls" -> NNLS (default, included in scipy)
         "bvls" -> LLSP/BVLS in openopt (only if available)
         The variable Exist_OpenOpt is (internally) set to True if available

     :param  **fcnargs: Will be passed to MPFIT, you can for example use xtol, gtol, ftol, quiet
     :param  verbose: self-explanatory
     :param  veryverbose: self-explanatory

    :return Fit parameters:
    :return Model:
    :return Fit errors:
    :return chi2:

    """
    import copy

    ## Set up some default parameters for mpfit
    if "xtol" not in fcnargs.keys() : fcnargs["xtol"] = 1.e-10
    if "gtol" not in fcnargs.keys() : fcnargs["gtol"] = 1.e-10
    if "ftol" not in fcnargs.keys() : fcnargs["ftol"] = 1.e-10
    if "quiet" not in fcnargs.keys() : fcnargs["quiet"] = True

    ## Checking the method used for the linear part
    if linear_method == "bvls" and not Exist_OpenOpt :
        print "WARNING: you selected BVLS, but OpenOpt is not installed"
        print "WARNING: we will therefore use NNLS instead"
        linear_method == "nnls"

    ## Checking if the linear_method is implemented
    if linear_method.lower() not in dic_linear_methods.keys():
        print "ERROR: you should use one of the following linear_method: ", dic_linear_methods.keys()
        return 0, 0, 0, 0

    f_Get_Iamp = dic_linear_methods[linear_method.lower()]
    ## If no coordinates is given, create them
    if xax is None:
        xax = np.arange(len(data))

    if not isinstance(xax,np.ndarray): 
        xax = (np.asarray(xax)).ravel()

    sxax = xax[xax != 0]

    min_xax = np.min(sxax)
    max_xax = np.max(sxax)
#    DlSigma = 0.5 * np.log10(max_xax / min_xax) / ngauss
    DlSigma = 0.02
    lminS = np.log10(min_xax)
    lmaxS = np.log10(max_xax)

    if minSigma is None : minSigma = min_xax
    if maxSigma is None : maxSigma = max_xax / sqrt(2.)

    if isinstance(params,np.ndarray): params=params.tolist()
    if params is not None : 
        if (len(params) > ngauss): 
            ngauss = len(params)
            if verbose :
                print "WARNING: Your input parameters do not fit the Number of input Gaussians"
                print "WARNING: the new number of input Gaussians is: ", ngauss

    ## if no input parameters are given, we set up the guess as a log spaced sigma between min and max
    default_params = np.logspace(lminS + DlSigma, lmaxS - DlSigma, ngauss)

    newdefault_minpars = copy.copy(default_minpars)
    newdefault_maxpars = copy.copy(default_maxpars)
    if force_Sigma_bounds :
        newdefault_minpars[0] = minSigma
        newdefault_maxpars[0] = maxSigma
    else :
        newdefault_minpars[0] = minSigma / 100.
        newdefault_maxpars[0] = maxSigma * 100.

    ## Set up the default parameters if needed
    params = set_parameters_and_default_1D(params, default_params, ngauss)
    fixed = set_parameters_and_default_1D(fixed, default_fixed, ngauss)
    limitedmin = set_parameters_and_default_1D(limitedmin, default_limitedmin, ngauss)
    limitedmax = set_parameters_and_default_1D(limitedmax, default_limitedmax, ngauss)
    minpars = set_parameters_and_default_1D(minpars, newdefault_minpars, ngauss)
    maxpars = set_parameters_and_default_1D(maxpars, newdefault_maxpars, ngauss)

    ## -------------------------------------------------------------------------------------
    ## mpfit function which returns the residual from the best fit N 1D Gaussians
    ## Parameters are just sigma,q,pa - the amplitudes are optimised at each step
    ## Two versions are available depending on whether BVLS or NNLS is used (and available)
    ## -------------------------------------------------------------------------------------
    def mpfitfun(p, fjac=None, x=None, err=None, data=None, f_Get_Iamp=None):
        Iamp = f_Get_Iamp(p, x, data)
        newp = (np.vstack((Iamp, p.transpose()))).transpose()
        if err is None : 
            return [0,fitn1dgauss_residuals(newp, x, data)]
        else :
            return [0,fitn1dgauss_residuals_err(newp, x, data, err)]

    ## Information about the parameters
    if veryverbose :
        print "------------------"
        print "GUESS:      Sig   "
        print "------------------"
        for i in xrange(ngauss) :
            print "GAUSS %02d: %8.3e"%(i+1, params[i])
        print "--------------------------------------"

    ## Information about the parameters
    parnames = {0:"SIGMA"}
    parinfo = [ {'n':ii, 'value':params[ii], 'limits':[minpars[ii],maxpars[ii]], 
        'limited':[limitedmin[ii],limitedmax[ii]], 'fixed':fixed[ii], 
        'parname':parnames[0]+str(ii+1), 'error':ii} for ii in xrange(len(params)) ]

    ## Fit with mpfit of q, sigma, pa on xax, yax, and data (+err)
    fa = {'x': xax, 'data': data, 'err':err, 'f_Get_Iamp':f_Get_Iamp}

    if verbose: 
        print "------ Starting the minimisation -------"
    result = mpfit(mpfitfun, functkw=fa, iterfunct=mpfitprint, nprint=1, parinfo=parinfo, **fcnargs) 
    ## Getting these best fit values into the dictionnary
    bestparinfo = [ {'n':ii, 'value':result.params[ii], 'limits':[minpars[ii],maxpars[ii]], 
        'limited':[limitedmin[ii],limitedmax[ii]], 'fixed':fixed[ii], 
        'parname':parnames[0]+str(ii), 'error':ii} for ii in xrange(len(result.params)) ]

    ## Recompute the best amplitudes to output the right parameters
    ## And renormalising them
    Iamp = f_Get_Iamp(result.params, xax, data)
    Ibestpar_array = (np.vstack((Iamp, result.params.transpose()))).transpose()

    ## Getting rid of the non-relevant Gaussians
    ## If parameters factor_Chi2 is set we use it as a threshold to remove gaussians
    ## Otherwise we just remove the zeros
    if err is None : nerr = np.ones_like(data)
    else : nerr =  err
    ## First get the Chi2 from this round
    bestChi2 = fitn1dgauss_chi2_err(Ibestpar_array, xax, data, nerr)
    result.ind = range(ngauss)

    k = 0
    Removed_Gaussians = []
    for i in xrange(ngauss) :
        ## Derive the Chi2 WITHOUT the ith Gaussian
        newChi2 = fitn1dgauss_chi2_err(np.delete(Ibestpar_array, i, 0), xax, data, nerr)
        ## If this Chi2 is smaller than factor_Chi2 times the best value, then remove
        ## It just means that Gaussian is not an important contributor
        if newChi2 <= factor_Chi2 * bestChi2 :
            val = bestparinfo.pop(k)
            result.ind.pop(k)
            Removed_Gaussians.append(i+1)
        else : k += 1

    if veryverbose :
        if len(Removed_Gaussians) != 0 :
            print "WARNING Removed Gaussians ", Removed_Gaussians
            print "WARNING: (not contributing enough to the fit)"
    ngauss = len(result.ind)

    ## New minimisation after removing all the non relevant Gaussians
    newresult = mpfit(mpfitfun, functkw=fa, iterfunct=mpfitprint, nprint=1, parinfo=bestparinfo, **fcnargs) 

    newresult.ind = range(ngauss)
    bestfit_params = newresult.params

    ## We add the Amplitudes to the array and renormalise them
    Iamp = f_Get_Iamp(bestfit_params, xax, data)
    Ibestfit_params = (np.vstack((Iamp, bestfit_params.transpose()))).transpose()
    Ibestfit_params[:,0] /= (sq2pi * Ibestfit_params[:,1])
    ## And we sort them with Sigma
    Ibestfit_params = Ibestfit_params[Ibestfit_params[:,1].argsort()]

    if newresult.status == 0:
        raise Exception(newresult.errmsg)

    if verbose :
        print "============================="
        print "FIT:      Imax       Sig   "
        print "============================="
        for i in xrange(ngauss) :
            print "GAUSS %02d: %8.3e  %8.3f"%(i+1, Ibestfit_params[i,0], Ibestfit_params[i,1])

        print "Chi2: ",newresult.fnorm," Reduced Chi2: ",newresult.fnorm/len(data)

    return Ibestfit_params, newresult, n_centred_onedgaussian_Imax(pars=Ibestfit_params)(xax)
   
################################################################################
# LMFIT version of the multigauss 1D fitting routine
################################################################################
##------ Transform the Parameters class into a single array ---- ##
def extract_mult1dG_params(Params) :
    ind = Params.ind
    ngauss = len(ind)
    p = np.zeros(ngauss, np.float64)
    for i in xrange(ngauss) :
        p[i] = Params['Sigma%02d'%(ind[i]+1)].value
    return p

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
                print "Chi2 = %g" % chi2
                myinput.aprint = 0
            myinput.aprint += 1

def multi_1dgauss_lmfit(xax, data, ngauss=1, err=None, params=None, 
        fixed=None, limitedmin=None, limitedmax=None, minpars=None, 
        maxpars=None, force_Sigma_bounds=False, factor_Chi2=1.01, iprint=50, lmfit_method='leastsq',
        verbose=True, veryverbose=False, linear_method="nnls", minSigma=None,
        maxSigma=None, lmfit_iprint=lmfit_iprint(), **fcnargs):
    """
    An improvement on gaussfit.  Lets you fit multiple 1D gaussians.

    :param   xax: x axis
    :param   data: count axis
    :param   ngauss: How many gaussians to fit?  Default 1 
    :param   err: error corresponding to data
    :param  params: Fit parameters: [width, axis ratio, pa] * ngauss
         If len(params) == 0, ngauss will be set to len(params) 
         These parameters need to have length = ngauss.  If ngauss > 1 and length = 1 , they will
         be replicated ngauss times, otherwise they will be reset to defaults:
    :param  fixed: Is parameter fixed?
    :param  limitedmin/minpars: set lower limits on each parameter (default: width>0)
    :param  limitedmax/maxpars: set upper limits on each parameter
    :param  force_Sigma_bounds: force the Sigmas to be within the radii range with some margin
        default to True
    :param  factor_Chi2: if one Gaussian contribute less than (factor-1) to the Chi2, we remove it
        If set to 1, it means only zero Gaussians will be removed
        If set to default=1.01, it means any Gaussian contributing to less than 1% will be removed
    :param  minSigma, maxSigma: default to None but can be set for bounds for Sigma
    :param  lmfit_method: method to pass on to lmfit ('leastsq', 'lbfgsb', 'anneal')
        Default is leastsq (most efficient for the problem)
    :param  linearmethod: Method used to solve the linear part of the problem
        Two methods are implemented: 
        "nnls" -> NNLS (default, included in scipy)
        "bvls" -> LLSP/BVLS in openopt (only if available)
        The variable Exist_OpenOpt is (internally) set to True if available
    :param  **fcnargs: Will be passed to LMFIT, you can for example use: xtol , gtol, ftol,  etc
    :param  iprint: if > 0 and verbose, print every iprint iterations of lmfit. default is 50
    :param  verbose: self-explanatory
    :param  veryverbose: self-explanatory

    :return Fit parameters:
    :return Model:
    :return Fit errors:
    :return chi2:

    """
    import copy

    ## Default values
    lmfit_methods = ['leastsq', 'lbfgsb', 'anneal']

    ## Method check
    if lmfit_method not in lmfit_methods :
        print "ERROR: method must be one of the three following methods : ", lmfit_methods

    ## Setting up epsfcn if not forced by the user
    ## We removed epsfcn as it is usually just the machine precision
    ## if "epsfcn" not in fcnargs.keys() : fcnargs["epsfcn"] = 1.e-20
    if "xtol" not in fcnargs.keys() : fcnargs["xtol"] = 1.e-10
    if "gtol" not in fcnargs.keys() : fcnargs["gtol"] = 1.e-10
    if "ftol" not in fcnargs.keys() : fcnargs["ftol"] = 1.e-10

    ## Checking the method used for the linear part
    if linear_method == "bvls" and not Exist_OpenOpt :
        print "WARNING: you selected BVLS, but OpenOpt is not installed"
        print "WARNING: we will therefore use NNLS instead"
        linear_method == "nnls"

    ## Checking if the linear_method is implemented
    if linear_method.lower() not in dic_linear_methods.keys():
        print "ERROR: you should use one of the following linear_method: ", dic_linear_methods.keys()
        return 0, 0, 0

    f_Get_Iamp = dic_linear_methods[linear_method.lower()]
    ## If no coordinates is given, create them
    if xax is None:
        xax = np.arange(len(data))

    if not isinstance(xax,np.ndarray): 
        xax = (np.asarray(xax)).ravel()

    sxax = xax[xax != 0]
    min_xax = np.min(sxax)
    max_xax = np.max(sxax)
#    DlSigma = 0.5 * np.log10(max_xax / min_xax) / ngauss
    DlSigma = 0.02
    lminS = np.log10(min_xax)
    lmaxS = np.log10(max_xax)

    if minSigma is None : minSigma = min_xax
    if maxSigma is None : maxSigma = max_xax / sqrt(2.)

    if isinstance(params,np.ndarray): params=params.tolist()
    if params is not None : 
        if (len(params) > ngauss): 
            ngauss = len(params) 
            if verbose :
                print "WARNING: Your input parameters do not fit the Number of input Gaussians"
                print "WARNING: the new number of input Gaussians is: ", ngauss

    ## if no input parameters are given, we set up the guess as a log spaced sigma between min and max
    default_params = np.logspace(lminS + DlSigma, lmaxS - DlSigma, ngauss)

    newdefault_minpars = copy.copy(default_minpars)
    newdefault_maxpars = copy.copy(default_maxpars)
    if force_Sigma_bounds :
        newdefault_minpars[0] = minSigma
        newdefault_maxpars[0] = maxSigma
    else :
        newdefault_minpars[0] = minSigma / 100.
        newdefault_maxpars[0] = maxSigma * 100.

    ## Set up the default parameters if needed
    params = set_parameters_and_default_1D(params, default_params, ngauss)
    fixed = set_parameters_and_default_1D(fixed, default_fixed, ngauss)
    limitedmin = set_parameters_and_default_1D(limitedmin, default_limitedmin, ngauss)
    limitedmax = set_parameters_and_default_1D(limitedmax, default_limitedmax, ngauss)
    minpars = set_parameters_and_default_1D(minpars, newdefault_minpars, ngauss)
    maxpars = set_parameters_and_default_1D(maxpars, newdefault_maxpars, ngauss)

    class input_residuals() :
        def __init__(self, iprint, verbose) :
            self.iprint = iprint
            self.verbose = verbose
            self.aprint = 0

    ## -----------------------------------------------------------------------------------------
    ## lmfit function which returns the residual from the best fit N 1D Gaussians
    ## Parameters are just sigma,q,pa - the amplitudes are optimised at each step
    ## -----------------------------------------------------------------------------------------
    def opt_lmfit(pars, myinput=None, x=None, err=None, data=None, f_Get_Iamp=None):
        """ Provide the residuals for the lmfit minimiser
            for a Multi 1D gaussian
        """

        # We retrieve the parameters
        pars_array = extract_mult1dG_params(pars)

        ## Derive the Normalised Gaussians for this set of parameters
        Iamp = f_Get_Iamp(pars_array, x, data)
        newp = (np.vstack((Iamp, pars_array.transpose()))).transpose()
        if err is None :
            res = fitn1dgauss_residuals(newp, x, data)
        else :
            res = fitn1dgauss_residuals_err(newp, x, data, err)
        lmfit_iprint(res, myinput, pars)
        return res

    ## Information about the parameters
    nameParam = ['Sigma']
    Lparams = Parameters()
    if verbose :
        print "-------------------"
        print "GUESS:      Sig    "
        print "-------------------"
    for i in xrange(ngauss) :
        Lparams.add(nameParam[0]+"%02d"%(i+1), value=params[i], min=minpars[i], max=maxpars[i], vary= not fixed[i])
        if verbose :
            print "GAUSS %02d: %8.3e"%(i+1, params[i])
    if veryverbose : print "--------------------------------------"
    ## Adding indices to follow up the Gaussians we may remove
    Lparams.ind = range(ngauss)

    ## Setting up the printing option
    myinput = input_residuals(iprint, verbose)

    ####################################
    ## Doing the minimisation with lmfit
    ####################################
    if verbose: 
        print "------ Starting the minimisation -------"
    result = minimize(opt_lmfit, Lparams, method=lmfit_method, args=(myinput, xax, err, data, f_Get_Iamp), **fcnargs)
    ## Remove the Null Gaussians
    result.params.ind = range(ngauss)
    ngauss, Ind_ZGauss = Remove_Zero_1DGaussians(ngauss, nameParam, result, xax, data, err, factor_Chi2,
            f_Get_Iamp, niter=1, verbose=veryverbose) 

    ## Recall the Minimizer function for a second iteration to get the new chi2 etc
    newresult = minimize(opt_lmfit, result.params, method=lmfit_method, args=(myinput, xax, err, data, f_Get_Iamp), **fcnargs)
    ## Remove the Null Gaussians
    newresult.params.ind = result.params.ind 
    ngauss, Ind_ZGauss = Remove_Zero_1DGaussians(ngauss, nameParam, newresult, xax, data, err, factor_Chi2,
            f_Get_Iamp, niter=2, verbose=veryverbose) 

    ## We add the Amplitudes to the array and renormalise them
    bestfit_params = extract_mult1dG_params(newresult.params)
    Iamp = f_Get_Iamp(bestfit_params, xax, data)
    Ibestfit_params = (np.vstack((Iamp, bestfit_params.transpose()))).transpose()
    Ibestfit_params[:,0] /= (sq2pi * Ibestfit_params[:,1])
    ## And we sort them with Sigma
    Ibestfit_params = Ibestfit_params[Ibestfit_params[:,1].argsort()]

    if verbose :
        print "============================="
        print "FIT:      Imax       Sig   "
        print "============================="
        for i in xrange(ngauss) :
            print "GAUSS %02d: %8.3e  %8.3f "%(i+1, Ibestfit_params[i,0], Ibestfit_params[i,1])

        print "Chi2: ",newresult.chisqr," Reduced Chi2: ",newresult.redchi

    return Ibestfit_params, newresult, n_centred_onedgaussian_Imax(pars=Ibestfit_params)(xax)

def Remove_Zero_1DGaussians(ngauss, nameParam, result, xax, data, err, factor_Chi2, f_Get_Iamp, niter=1, verbose=False) :
    
    ## Recompute the best amplitudes and remove the ones that are zeros
    bestpar_array = extract_mult1dG_params(result.params)
    Iamp = f_Get_Iamp(bestpar_array, xax, data)
    ## New array including the Imax values
    Ibestpar_array = (np.vstack((Iamp, bestpar_array.transpose()))).transpose()

    ## Getting rid of the non-relevant Gaussians
    ## If parameters factor_Chi2 is set we use it as a threshold to remove gaussians
    ## Otherwise we just remove the zeros
    if err is None : nerr = np.ones_like(data)
    else : nerr =  err
    ## First get the Chi2 from this round
    bestChi2 = fitn1dgauss_chi2_err(Ibestpar_array, xax, data, nerr)

    k = 0
    Removed_Gaussians = []
    for i in xrange(ngauss) :
        ## Derive the Chi2 WITHOUT the ith Gaussian
        newChi2 = fitn1dgauss_chi2_err(np.delete(Ibestpar_array, i, 0), xax, data, nerr)
        ## If this Chi2 is smaller than factor_Chi2 times the best value, then remove
        ## It just means that Gaussian is not an important contributor
        if newChi2 <= factor_Chi2 * bestChi2 :
            result.params.pop(nameParam[0]+'%02d'%(result.params.ind[k]+1))
            result.params.ind.pop(k)
            Removed_Gaussians.append(i+1)
        else : k += 1
    New_ngauss = len(result.params.ind)

    if verbose :
        if len(Removed_Gaussians) != 0 :
            print "WARNING Removed Gaussians (iteration %d) : "%(niter), Removed_Gaussians
            print "WARNING: (not contributing enough to the fit)"

    return New_ngauss, Removed_Gaussians
