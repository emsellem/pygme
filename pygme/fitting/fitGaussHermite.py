""" This routine will create and fit Gauss-Hermite 
profiles using a simple description with an input given array GH, which 
includes the amplitude, mean velocity, dispersion (Gaussian) plus the higher 
order terms, h3, h4, h5, ...

Many versions of such routines exist, but for such 1d profiles they 
should provide similar results

This programme includes 2 minimising schemes: one based on mpfit 
(as translated in python) and one based on the scipy minimisation lmfit. 
Both provides similar (although not exactly identical) results (which could be a
precision issue, not an algorithm issue).
"""
#####################################################################
# VERSION of              fitGaussHermite
# 
# V.0.0.3    3 August 2015 - Changed call to GaussHermite
# V.0.0.2   21 August 2013 - Some docstrings and formatting changes
# V.0.0.1   16 Dec 2012 - Adaptation from the fitn1dgauss from pygme
#
#####################################################################
# Author : Eric Emsellem - @ ESO / CRAL  2012
#          eric.emsellem@eso.org
#####################################################################
# Specific development of pygme to be able to fit LOSVDs from N body model
######################################################################################
import numpy as np
from numpy import exp, sqrt, pi

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
    print("WARNING: Only mpfit is available an optimiser")
    print("WARNING: you may want to install lmfit!")
    print("WARNING: The routine will therefore use mpfit")
    print("         Available from pygme.fitting")
## Use mpfit for the non-linear least-squares
## if lmfit is not there

from pygme.mge_miscfunctions import GaussHermite

""" Note (from Adam Ginsburg) about mpfit/leastsq: 

I switched everything over to the Markwardt mpfit routine for a few reasons,
but foremost being the ability to set limits on parameters, not just force them
to be fixed.  As far as I can tell, leastsq does not have that capability.  

The version of mpfit I use can be found here:
    http://code.google.com/p/agpy/source/browse/trunk/mpfit
    This is included in the present distribution

Note (from Emsellem): this is now also solved using lmfit which is based on leastsq from scipy
  So the two methods are implemented

"""

# ==========================================================================
## Returns a function which does the calculation, given a set of parameters,
## of the sum of Gaussians, the I being the maximum intensities (not normed)
# ==========================================================================
def _set_GHparameters(parlist, default, degGH) :
    """ Set up the parameters given a default 

    if the Input does not have the right size (degGH) it fills in the rest
    with the default

    Input is :
    ==========
    : param parlist : the input parameters you wish to set
    : param default : the default when needed 
    : param degGH   : the right size (degree of Gauss-Hermite to fit)

    :returns : parameter list

    """
    if parlist is None : parlist = default

    ## Replicate the parlist with the right size first
    newparlist = [parlist[0]] * degGH
    ## Now computing the length of the input parlist
    lparlist = len(parlist)
    ## And completing it with the right default if needed
    newparlist[:lparlist] = parlist
    newparlist[lparlist:] = default[lparlist:]

    return newparlist

## -----------------------------------------------------------------------------------------
## Return the difference between a model and the data WITH ERRORS
## -----------------------------------------------------------------------------------------
def _fitGH_residuals_err(par, x, data, err) :
    """ Residual function for Gauss Hermite fit. Include Errors. """
    return ((data.ravel() - GaussHermite(x, par))/err.ravel())
## -----------------------------------------------------------------------------------------
## Return the difference between a model and the data WITH ERRORS
## -----------------------------------------------------------------------------------------
def _fitGH_chi2_err(par, x, data, err) :
    """ Sum of squares of residuals  function for Gauss Hermite fit. Include Errors. """
    return np.sum(((data.ravel() - GaussHermite(x, par))/err.ravel())**2)
## -----------------------------------------------------------------------------------------
## Return the difference between a model and the data
## -----------------------------------------------------------------------------------------
def _fitGH_residuals(par, x, data) :
    """ Residual function for Gauss Hermite fit. No Errors. """
    return (data.ravel() - GaussHermite(x, par))
## -----------------------------------------------------------------------------------------
## Return the difference between a model and the data
## -----------------------------------------------------------------------------------------
def _fitGH_chi2(par, x, data) :
    """ Sum of squares of residuals function for Gauss Hermite fit. No Errors. """
    return np.sum((data.ravel() - GaussHermite(x, par))**2)

## -----------------------------------------------------------------------------------------
## Return the two first moments of a 1D profile
## -----------------------------------------------------------------------------------------
def moment1d(xax, data) :
    """ Returns the first two moments of a 1D distribution

    e.g. a Line-of-Sight-Velocity-Distribution with arrays of V and amplitude

    Input:
    :param  xax : the input coordinates (1d array)
    :param data :  the input data (1d array, same size as xax)

    :returns m0, m1, m2 : the maximum amplitude and first two moments

    """
    m0 = np.max(data)
    m1 = np.average(xax, weights=data)
    m2 = sqrt(np.average(xax**2, weights=data) - m1**2)
    return m0, m1, m2

## -----------------------------------------------------------------------------------------
## Return the Amplitude which optimises the Chi2 for a simple linear sum
## -----------------------------------------------------------------------------------------
def _Solve_Amplitude(data, ufit, error=None) :
    """ Compute the amplitude needed to normalise the 1d profile which minimises
    the Chi2, given a x array, data and an error array

    The calculation follows a simple linear optimisation using
          
          Ioptimal = (dn x dn / dn x fn)

          where dn is the normalised data array = data / error
          and   fn is the normalised fitted array = fit / error
    Input:
    
    data     -   the input data (1D array, all input arrays should have the same 1d size)
    ufit     -   the un-normalised fitting profile
    error    -   the error array - Default is None (would use constant errors then)

    Output:

    Iamp     -   The normalisation needed to minimise Chi2
    """
    
    if error is None : error = np.ones_like(data)
    error = np.where(error == 0., 1., error)

    dn = data / error
    fn = ufit / error
    Iamp = np.sum(dn * dn) / np.sum(dn * fn)
    return Iamp

################################################################################
# MPFIT version of the 1D Gauss-Hermite fitting routine
################################################################################
def fitGH_mpfit(xax, data, degGH=2, err=None, params=None, fixed=None, limitedmin=None, 
        limitedmax=None, minpars=None, maxpars=None, verbose=True, veryverbose=False, **fcnargs):
    """ Fitting routine for a Gauss-Hermite function using mpfit

    :param xax: x axis
    :param data: count axis
    :param err: error corresponding to data
    :param degGH: order of the Gauss-Hermite moments to fit. Must be >=2
        If =2 (default), the programme will only fit a single Gaussian
        If > 2, it will add as many h_n as needed.
    :param params: Input fit parameters: Vgauss, Sgauss, h3, h4, h5...
        Length should correspond to degGH. If not, a guess will be used.
    :param fixed: Is parameter fixed?
    :param limitedmin/minpars: set lower limits on each parameter (default: width>0)
    :param limitedmax/maxpars: set upper limits on each parameter
    :param iprint: if > 0 and verbose, print every iprint iterations of lmfit. default is 50
    :param lmfit_method: method to pass on to lmfit ('leastsq', 'lbfgsb', 'anneal').
        Default is leastsq (most efficient for the problem)
    :param **fcnargs: Will be passed to MPFIT, you can for example use xtol, gtol, ftol, quiet
    :param verbose: self-explanatory
    :param veryverbose: self-explanatory

    :returns   Fit parameters:
    :returns   Model:
    :returns   Fit errors:
    :returns   chi2:

    """
    ## Set up some default parameters for mpfit
    if "xtol" not in list(fcnargs.keys()) : fcnargs["xtol"] = 1.e-10
    if "gtol" not in list(fcnargs.keys()) : fcnargs["gtol"] = 1.e-10
    if "ftol" not in list(fcnargs.keys()) : fcnargs["ftol"] = 1.e-10
    if "quiet" not in list(fcnargs.keys()) : fcnargs["quiet"] = True

    ## If no coordinates is given, create them and use pixel as a unit
    ## The x axis will be from 0 to N-1, N being the number of data points
    ## which will be considered as "ordered" on a regular 1d grid
    if xax is None:
        xax = np.arange(len(data))
    else :
        ## Make sure that x is an array (if given)
        if not isinstance(xax,np.ndarray): 
            xax = (np.asarray(xax)).ravel()

    ## Compute the moments of the distribution for later purposes
    ## This provides Imax, V and Sigma from 1d moments
    momProf = moment1d(xax, data)

    if isinstance(params,np.ndarray): params=params.tolist()
    if params is not None : 
        if (len(params) > degGH): 
            print("ERROR: input parameter array (params) is larger than expected")
            print("ERROR: It is %d while it should be smaller or equal to %s (degGH)", len(params), degGH)
            return 0., 0., 0., 0.
        elif (len(params) < degGH): 
            if verbose :
                print("WARNING: Your input parameters do not fit the Degre set up for the GH moments")
                print("WARNING: the given value of degGH (%d) will be kept ", degGH)
                print("WARNING: A guess will be used for the input fitting parameters ")

    default_params= np.zeros(degGH, dtype=np.float64) + 0.02
    default_params[:2] = momProf[1:]
    default_minpars= np.zeros(degGH, dtype=np.float64) - 0.2
    default_minpars[:2] = [np.min(xax), np.min(np.diff(xax)) / 3.]
    default_maxpars= np.zeros(degGH, dtype=np.float64) + 0.2
    default_maxpars[:2] = [np.max(xax), (np.max(xax) - np.min(xax)) / 3.]
    default_limitedmin = [True] * degGH
    default_limitedmax = [True] * degGH
    default_fixed = [False] * degGH

    ## Set up the default parameters if needed
    params = _set_GHparameters(params, default_params, degGH)
    fixed = _set_GHparameters(fixed, default_fixed, degGH)
    limitedmin = _set_GHparameters(limitedmin, default_limitedmin, degGH)
    limitedmax = _set_GHparameters(limitedmax, default_limitedmax, degGH)
    minpars = _set_GHparameters(minpars, default_minpars, degGH)
    maxpars = _set_GHparameters(maxpars, default_maxpars, degGH)

    ## -------------------------------------------------------------------------------------
    ## mpfit function which returns the residual from the best fit Gauss-Hermite
    ## Parameters are just V, Sigma, H3,... Hn - the amplitudes are optimised at each step
    ## -------------------------------------------------------------------------------------
    def mpfitfun(p, fjac=None, x=None, err=None, data=None):
        GH = np.concatenate(([1.0], p))
        ufit = GaussHermite(x, GH)
        GH[0] = _Solve_Amplitude(data, ufit, err)
        if err is None : 
            return [0, _fitGH_residuals(GH, x, data)]
        else :
            return [0, _fitGH_residuals_err(GH, x, data, err)]
    ## -------------------------------------------------------------------------------------
    ## Printing routine for mpfit
    ## -------------------------------------------------------------------------------------
    def mpfitprint(mpfitfun, p, iter, fnorm, functkw=None, parinfo=None, quiet=0, dof=None) :
        print("Chi2 = ", fnorm)
    ## -------------------------------------------------------------------------------------

    ## Information about the parameters
    parnames = {0:"V", 1:"S"}
    for i in range(2, degGH) : parnames[i] = "H_%02d"%(i+1)

    ## Information about the parameters
    if veryverbose :
        print("--------------------------------------")
        print("GUESS:            ")
        print("------")
        for i in range(degGH) :
            print(" %s :  %8.3f"%(parnames[i], params[i]))
        print("--------------------------------------")

    parinfo = [ {'n':ii, 'value':params[ii], 'limits':[minpars[ii],maxpars[ii]], 
        'limited':[limitedmin[ii],limitedmax[ii]], 'fixed':fixed[ii], 
        'parname':parnames[ii], 'error':ii} for ii in range(len(params)) ]

    ## Fit with mpfit of q, sigma, pa on xax, yax, and data (+err)
    fa = {'x': xax, 'data': data, 'err':err}

    if verbose: 
        print("------ Starting the minimisation -------")
    result = mpfit(mpfitfun, functkw=fa, iterfunct=mpfitprint, nprint=1, parinfo=parinfo, **fcnargs) 
    ## Recompute the best amplitudes to output the right parameters
    ## And renormalising them
    GH = np.concatenate(([1.0], result.params))
    ufit = GaussHermite(xax, GH)
    Iamp = _Solve_Amplitude(data, ufit, err)
    Ibestpar_array = np.concatenate(([Iamp], result.params))

    if result.status == 0:
        raise Exception(result.errmsg)

    if verbose :
        print("=====")
        print("FIT: ")
        print("=================================")
        print("        I         V         Sig   ")
        print("   %8.3f  %8.3f   %8.3f "%(Ibestpar_array[0], Ibestpar_array[1], Ibestpar_array[2]))
        print("=================================")
        for i in range(2, degGH) :
            print("GH %02d: %8.4f "%(i+1, Ibestpar_array[i+1]))
        print("=================================")

        print("Chi2: ",result.fnorm," Reduced Chi2: ",result.fnorm/len(data))

    return Ibestpar_array, result, GaussHermite(xax, Ibestpar_array)
   
################################################################################
# LMFIT version of the Gauss Hermite fitting routine
################################################################################
def _extract_GH_params(Params, parnames) :
    """ Extract the parameters from the formatted list """
    degGH = len(parnames)
    p = np.zeros(degGH, dtype=np.float64)
    for i in range(degGH) :
        p[i] = Params[parnames[i]].value
    return p

def fitGH_lmfit(xax, data, degGH=2, err=None, params=None, fixed=None, limitedmin=None, 
        limitedmax=None, minpars=None, maxpars=None, iprint=50, lmfit_method='leastsq',
        verbose=True, veryverbose=False, **fcnargs):
    """ Fitting routine for a Gauss-Hermite function using lmfit

    :param xax: x axis
    :param data: count axis
    :param err: error corresponding to data
    :param degGH: order of the Gauss-Hermite moments to fit. Must be >=2
        If =2 (default), the programme will only fit a single Gaussian
        If > 2, it will add as many h_n as needed.
    :param params: Input fit parameters: Vgauss, Sgauss, h3, h4, h5...
        Length should correspond to degGH. If not, a guess will be used.
    :param fixed: Is parameter fixed?
    :param limitedmin/minpars: set lower limits on each parameter (default: width>0)
    :param limitedmax/maxpars: set upper limits on each parameter
    :param iprint: if > 0 and verbose, print every iprint iterations of lmfit. default is 50
    :param lmfit_method: method to pass on to lmfit ('leastsq', 'lbfgsb', 'anneal'). Default is leastsq (most efficient for the problem)
    :param **fcnargs: Will be passed to LMFIT, you can for example use xtol, gtol, ftol, quiet
    :param verbose: self-explanatory
    :param veryverbose: self-explanatory

    :returns Fitparam: Fitted parameters
    :returns Result: Structure including the results
    :returns Fit: Fitted values

    """
    import copy

    ## Default values
    lmfit_methods = ['leastsq', 'lbfgsb', 'anneal']

    ## Method check
    if lmfit_method not in lmfit_methods :
        print("ERROR: method must be one of the three following methods : ", lmfit_methods)

    ## Setting up epsfcn if not forced by the user
    if "epsfcn" not in list(fcnargs.keys()) : fcnargs["epsfcn"] = 0.01
    if "xtol" not in list(fcnargs.keys()) : fcnargs["xtol"] = 1.e-10
    if "gtol" not in list(fcnargs.keys()) : fcnargs["gtol"] = 1.e-10
    if "ftol" not in list(fcnargs.keys()) : fcnargs["ftol"] = 1.e-10

    ## If no coordinates is given, create them and use pixel as a unit
    ## The x axis will be from 0 to N-1, N being the number of data points
    ## which will be considered as "ordered" on a regular 1d grid
    if xax is None:
        xax = np.arange(len(data))
    else :
        ## Make sure that x is an array (if given)
        if not isinstance(xax,np.ndarray): 
            xax = (np.asarray(xax)).ravel()

    ## Compute the moments of the distribution for later purposes
    ## This provides Imax, V and Sigma from 1d moments
    momProf = moment1d(xax, data)

    if isinstance(params,np.ndarray): params=params.tolist()
    if params is not None : 
        if (len(params) > degGH): 
            print("ERROR: input parameter array (params) is larger than expected")
            print("ERROR: It is %d while it should be smaller or equal to %s (degGH)", len(params), degGH)
            return 0., 0., 0., 0.
        elif (len(params) < degGH): 
            if verbose :
                print("WARNING: Your input parameters do not fit the Degre set up for the GH moments")
                print("WARNING: the given value of degGH (%d) will be kept ", degGH)
                print("WARNING: A guess will be used for the input fitting parameters ")

    default_params= np.zeros(degGH, dtype=np.float64) + 0.02
    default_params[:2] = momProf[1:]
    default_minpars= np.zeros(degGH, dtype=np.float64) - 0.2
    default_minpars[:2] = [np.min(xax), np.min(np.diff(xax)) / 3.]
    default_maxpars= np.zeros(degGH, dtype=np.float64) + 0.2
    default_maxpars[:2] = [np.max(xax), (np.max(xax) - np.min(xax)) / 3.]
    default_limitedmin = [True] * degGH
    default_limitedmax = [True] * degGH
    default_fixed = [False] * degGH

    ## Set up the default parameters if needed
    params = _set_GHparameters(params, default_params, degGH)
    fixed = _set_GHparameters(fixed, default_fixed, degGH)
    limitedmin = _set_GHparameters(limitedmin, default_limitedmin, degGH)
    limitedmax = _set_GHparameters(limitedmax, default_limitedmax, degGH)
    minpars = _set_GHparameters(minpars, default_minpars, degGH)
    maxpars = _set_GHparameters(maxpars, default_maxpars, degGH)

    class input_residuals() :
        def __init__(self, iprint, verbose) :
            self.iprint = iprint
            self.verbose = verbose
            self.aprint = 0

    ## -----------------------------------------------------------------------------------------
    ## lmfit function which returns the residual from the best fit Gauss Hermite function
    ## Parameters are V, S, h3, h4... - the amplitudes are optimised at each step
    ##                                  as this is a linear problem then
    ## -----------------------------------------------------------------------------------------
    def opt_lmfit(pars, myinput=None, x=None, err=None, data=None) :
        """ Provide the residuals for the lmfit minimiser
            in the case of a Gauss-Hermite function
        """
        p = _extract_GH_params(pars, parnames)
        GH = np.concatenate(([1.0], p))
        ufit = GaussHermite(x, GH)
        GH[0] = _Solve_Amplitude(data, ufit, err)
        if err is None : 
            res = _fitGH_residuals(GH, x, data)
        else :
            res = _fitGH_residuals_err(GH, x, data, err)
        lmfit_iprint(res, myinput)
        return res
    ## -----------------------------------------------------------------------------------------
    def lmfit_iprint(res, myinput) :
        """ Printing function for the iteration in lmfit
        """
        if (myinput.iprint > 0) & (myinput.verbose) :
            if myinput.aprint == myinput.iprint :
                print("Chi2 = %g"%((res*res).sum()))
                myinput.aprint = 0
            myinput.aprint += 1
    ## -----------------------------------------------------------------------------------------

    ## Information about the parameters
    parnames = {0:"V", 1:"S"}
    for i in range(2, degGH) : parnames[i] = "H_%02d"%(i+1)

    Lparams = Parameters()
    if veryverbose :
        print("-------------------")
        print("GUESS:     ")
        print("-------------------")
    for i in range(degGH) :
        Lparams.add(parnames[i], value=params[i], min=minpars[i], max=maxpars[i], vary= not fixed[i])
        if veryverbose :
            print("%s %02d: %8.3f"%(parnames[i], i+1, params[i]))
    if veryverbose : print("--------------------------------------")

    ## Setting up the printing option
    myinput = input_residuals(iprint, verbose)

    ####################################
    ## Doing the minimisation with lmfit
    ####################################
    if verbose: 
        print("------ Starting the minimisation -------")
    result = minimize(opt_lmfit, Lparams, method=lmfit_method, args=(myinput, xax, err, data), **fcnargs)

    ## Recompute the best amplitudes to output the right parameters
    ## And renormalising them
    p = _extract_GH_params(result.params, parnames)
    GH = np.concatenate(([1.0], p))
    ufit = GaussHermite(xax, GH)
    Iamp = _Solve_Amplitude(data, ufit, err)
    Ibestpar_array = np.concatenate(([Iamp], p))

    if verbose :
        print("=====")
        print("FIT: ")
        print("=================================")
        print("        I       V         Sig   ")
        print("  %8.3f  %8.3f   %8.3f "%(Ibestpar_array[0], Ibestpar_array[1], Ibestpar_array[2]))
        print("=================================")
        for i in range(2, degGH) :
            print("GH %02d: %8.4f "%(i+1, Ibestpar_array[i+1]))
        print("=================================")

        print("Chi2: ",result.chisqr," Reduced Chi2: ",result.redchi)

    return Ibestpar_array, result, GaussHermite(xax, Ibestpar_array)
   
################################################################################
# LMFIT version of the multigauss 1D fitting routine
################################################################################
