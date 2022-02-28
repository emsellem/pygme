#! /usr/bin/env python
""" This module provides functions useful for Sersic profiles, 
including a 3d deprojected approximation for a Sersic-n function.
"""
import os
from os.path import join as joinpath
import copy

### Importing the required stuff from different modules
import numpy as np
from numpy import pi, log10, log, exp, sqrt, cos, sin
from numpy import float64 as nfloat

import scipy
from scipy import interpolate, special, integrate, ndimage
from scipy.special import gamma, gammainc

from astropy.constants import G 
from astropy import units as u

from pygme.utils.iodata import get_package_data

# Eric Emsellem / ESO-CRAL. 
__version__ = '1.2.0 (August 10, 2020)'
# Version 1.2.0 : 2020/08/10 - adding MultSersic and cleaning properties
# Version 1.1.3 : Fixing some docstrings
# Version 1.1.2 : Fixing bug in Integrated Luminosity
# Version 1.1.0 : small fix on the get_i0 function

# Some useful values
Ggrav = G.to((u.km / u.s)**2 * u.pc / u.Msun)
facPC = nfloat(pi / 0.648)
_coef_bn_invn = [2194697.0 / 30690717750.0, 131. / 1148175.0, 
                      46.0 / 25515.0, 4.0 / 405.0, -1.0 / 3.0]
_coef_bn_n = [13.43, -19.67, 10.95, -0.8902, 0.01945]


def bn(n=1.0):
    if (n > 0.36): 
        bn_poly = np.poly1d(_coef_bn_invn)
        return (2.0 * n + bn_poly(1. / n))
    else: 
        bn_poly = np.poly1d(_coef_bn_n)
        return (bn_poly(n))

def get_data(datapath, asfile=False):
    """
    Retrieves a data file from a local file

    :param str datapath:
        The path of the data to be retrieved. 
    :param bool asfile:
        If True, a file-like object is returned that can be used to access the
        data. Otherwise, a string with the full content of the file is returned.
    :param localfn:
        The filename to use for saving (or loading, if the file is present)
        locally. If it is None, the filename will be inferred from the URL
        if possible. This file name is always relative to the astropysics data
        directory (see :func:`astropysics.config.get_data_dir`). This has no
        effect if :func:`set_data_store` is set to False.

    :returns: A file-like object or a loaded numpy array (with loadtxt)

    :raises IOError:
        If the datapath is requested as a local data file and not found.

    """
    ## The file is a local file - try to get it
    if not os.path.isfile(datapath) :
        print("The file %s you are trying to access does not exist" %(datapath))
        raise IOError
    print(f"Reading {datapath}")
    if asfile:
        return open(datapath)
    else:
        return np.loadtxt(datapath)

## Tabulated function to get deprojected Sersic ######################
## Derive the h1/2/3 functions
def hfunc_abel(n) :
    """
       Provides the parameters for an Abel integration of a Sersic function
       It returns a set of factors which depends on the input Sersic n
    """
    X = get_package_data("Abel_param.dat", asfile=False)
    ndata = X[:,0]
    nudata = X[:,1]
    kdata = X[:,2]
    a0data = X[:,3]
    a1data = X[:,4]
    a2data = X[:,5]
    a3data = X[:,6]
    a0 = interpolate.interp1d(ndata, a0data)
    a1 = interpolate.interp1d(ndata, a1data)
    a2 = interpolate.interp1d(ndata, a2data)
    a3 = interpolate.interp1d(ndata, a3data)
    k = interpolate.interp1d(ndata, kdata)
    nu = interpolate.interp1d(ndata, nudata)

    if (n > 10) | (n < 0.5) :
        print("Error: n is too high or too low here \n")
        return 0.,0.,0.,0.,0., 0.

    return nu(n), k(n), a0(n), a1(n), a2(n), a3(n)

# h1, h2, h3 function 
# Derive the h1/2/3 functions
def hfunc_Trujillo(n) :
    from . import __path__ as path
    X = get_package_data("Trujillo.dat", asfile=False)
    ndata = X[:,0]
    nudata = X[:,1]
    pdata = X[:,2]
    h1data = X[:,3]
    h2data = X[:,4]
    h3data = X[:,5]
    h1 = interpolate.interp1d(ndata, h1data)
    h2 = interpolate.interp1d(ndata, h2data)
    h3 = interpolate.interp1d(ndata, h3data)
    p = interpolate.interp1d(ndata, pdata)
    nu = interpolate.interp1d(ndata, nudata)

    if (n > 10) | (n < 0.5) :
        print("Error: n is too high or too low here \n")
        return 0.,0.,0.,0.,0.

    return nu(n), p(n), h1(n), h2(n), h3(n)


# Incomplete Gamma function as it should be (unnormalised)
# Derive the approximation for the Sersic b parameter
def inc_gamma(a,x) :
   return gamma(a) * gammainc(a,x)

# Ch function 
# Derive the Ch function given in Tersic +
def fac_ch(n, nu, p, a0, a1, a2, a3, x, bn) :
   if n != 1 :
      val = x**(p * (1.- n) / n) / (1. - (a0 + a1 * log10(x) + a2 * log10(x)**2 + a3 * log10(x)**3))
   else :
      val = 1.
   return val * scipy.special.kv(nu, bn * x**(1./n))

# Check lengths
def check_lengths(ldata, fill=None):
    """Check that lenghts of all data in ldata are the same
    """
    for i, l in enumerate(ldata):
        if l is None:
            ldata[i] = []
            
    cdata = copy.copy(ldata)
    nmax = np.max([len(l) for l in ldata])

    # Add fill value to the ones that are empty
    for i, l in enumerate(cdata):
        if len(l) == 0:
            ldata.remove(l)
            cdata[i] = [fill] * nmax

    return cdata, (len(set([len(l) for l in cdata])) == 1)

def logrspace(rscale, nr=100, facmin=100., facmax=10.):
    return np.logspace(log10(rscale / facmin), log10(facmax * rscale), nr)

# ----------------------------------------------------------------------
# Multi Sersic profiles
class  MultiSersic(object):
    def __init__(self, n=[], re=[], ie=[], i0=[], ltot=[], rsamp=None, nr=10):
        """Multiple Sersic profile
        """
        # initialise
        self.comp = []

        # Check if number of input is ok
        (n, re, ie, i0, ltot), status = check_lengths([n, re, ie, i0, ltot])
        if not status:
            print("Error: not all included lists have the same size")
            return

        if len(n) == 0:
            print("No component provided")
            return

        if rsamp is None:
            remax = np.max(np.array(re))
            self._r = logrspace(remax, nr=nr) 
        else:
            self._r = rsamp

        self._reset_comp(n, re, ie, i0, ltot)

    @property
    def r(self):
        if not hasattr(self, "_r"):
            return None
        return self._r

    @property
    def ncomp(self):
        return len(self.comp)

    def _reset_comp(self, n, re, ie, i0, ltot):
        for i, (sn, sre, sie, si0, sltot) in enumerate(zip(n, re, ie, i0, ltot)):
            self.add_comp(n=sn, re=sre, ie=sie, i0=si0, ltot=sltot)

    def remove_comp(self, i):
        """Removing one component, using its index
        """
        return self.comp.pop(i)

    def add_comp(self, **kwargs):
        newcomp = SersicProfile(**kwargs, rsamp=self.r)
        self.comp.append(newcomp)

    def get_sersic(self, r):
        return np.sum(np.array([self.comp[i].get_sersic(r) for i in range(self.ncomp)]),
                      axis=0)

    def get_rho_abel(self, r):
        return np.sum(np.array([self.comp[i].get_rho_abel(r) for i in range(self.ncomp)]),
                      axis=0)
    @property
    def rhop(self):
        return np.sum(np.array([self.comp[i].rhop for i in range(self.ncomp)]),
                      axis=0)

    @property
    def rho_abel(self):
        return np.sum(np.array([self.comp[i].rho_abel for i in range(self.ncomp)]),
                      axis=0)

    @property
    def rho_PS97(self):
        return np.sum(np.array([self.comp[i].rho_PS97 for i in range(self.ncomp)]),
                      axis=0)

    @property
    def mint(self):
        return np.sum(np.array([self.comp[i].mint for i in range(self.ncomp)]),
                      axis=0)

    @property
    def mint_abel(self):
        return np.sum(np.array([self.comp[i].mint_abel for i in range(self.ncomp)]),
                      axis=0)

    @property
    def mint_PS97(self):
        return np.sum(np.array([self.comp[i].mint_PS97 for i in range(self.ncomp)]),
                      axis=0)

    @property
    def n(self):
        return np.array([self.comp[i].n for i in range(self.ncomp)])

    @property
    def re(self):
        return np.array([self.comp[i].re for i in range(self.ncomp)])

    @property
    def i0(self):
        return np.array([self.comp[i].i0 for i in range(self.ncomp)])

    @property
    def ie(self):
        return np.array([self.comp[i].ie for i in range(self.ncomp)])

    @property
    def ltot(self):
        return np.array([self.comp[i].ltot for i in range(self.ncomp)])

    def combine_rhop(self, nameattr="age", log10=False):
        """Combine the weighted sum of a certain attribute
        per component, using rhop as the weight
        """

        ok = True
        data_comp = []
        for i in range(self.ncomp):
            if not hasattr(self.comp[i], nameattr):
                print(f"ERROR: cannot combine attribute {nameattr} - missing for component {i}")
                ok = False
            else:
                data_comp.append(getattr(self.comp[i], nameattr))
        
        def func(r):
            if log10:
                temp = np.log10(np.sum([10**(data_comp[i](r)) * self.comp[i].get_sersic(r) for i in range(self.ncomp)], axis=0) / self.get_sersic(r))
            else:
                temp = np.sum([data_comp[i](r) * self.comp[i].get_sersic(r) for i in range(self.ncomp)], axis=0) / self.get_sersic(r)
            return temp

        setattr(self, nameattr, func)


# Sersic fitting and tool function
class SersicProfile(object) :
    ## Derive the approximation for the Sersic b parameter
    def __init__(self, n=1., re=1.0, ie=None, i0=None, ltot=None, rsamp=None):
        """ Parameters of the Sersic law: n, ie, re
              n is adimensional and can be anything from 0 to infinity
                It usually has values between 1 and 8 for real galaxies
              re in parsec
              ie in Lsun.pc-2 the surface brightness at 1 re
                or
              i0 in Lsun.pc-2 the central surface brightness

              bn is adimensional and derived from n

              Other Parameters:
                  rhoL is in Lsun.pc-3
                  ltot in Lsun
                  rho_average is also in Lsun/pc3
        """
        self.n = n
        self.re = re

        # Now setting up in order of less importance in case these
        # are multiply defined
        self._reset_norm(ie, i0, ltot)
        self._reset_profiles(rsamp=rsamp)

    def _reset_norm(self, ie, i0, ltot):
        # first set the default
        self._ie = ie
        self._i0 = i0
        self._ltot = ltot
        # now see which ones are the priority
        self.ie = ie
        self.i0 = i0
        self.ltot = ltot

    def _reset_profiles(self, rsamp=None):
        # Compute some input numbers for the Prugniel-Simien function
        self.p_PS97 = 1.0 - 0.6097 / self.n + 0.05463 / (self.n * self.n)
        self.rho0_PS97 = self.i0 * self.bn**(self.n * (1. - self.p_PS97)) \
                         * gamma(2. * self.n) / (2. * self.re * gamma(self.n * (3. - self.p_PS97)))

        self.r, self.rre, self.nr = self.get_radii(rsamp)

        # We derive the projected Sersic profile (direct evaluation)
        self.rhop = self.get_sersic(self.r)

        # we then derive the spatial luminosity profile from Simonneau et al
        self.rhoL = self.get_rho_sersic(self.r)
        # we then derive the spatial luminosity profile from Abel deprojection
        self.rho_abel = self.get_rho_abel(self.r)
        # we then derive the spatial luminosity profile from Prugniel-Simien 97
        self.rho_PS97 = self.get_rho_PS97(self.r)

        # Now the integrated mass from the same models
        self.mint = self.get_mint_sersic(self.r)
        self.mint_PS97 = self.get_mint_PS97(self.r)
        self.mint_abel = self.get_mint_abel(self.r)

    @property 
    def n(self):
        if not hasattr(self, "_n"):
            self._n = 2.0
        return self._n

    @n.setter
    def n(self, val):
        if val is None:
            print("Using default n = 2.0")
            self._n = 2.0
        else:
            self._n = np.float(val)
        self.bn = self._get_bn()

    @property 
    def re(self):
        if not hasattr(self, "_re"):
            self._re = 2.0
        return self._re

    @re.setter
    def re(self, val):
        if val is None:
            print("Using default re = 1.0 value")
            self._re = 1.0
        else:
            self._re = val

    ####################################################################
    ## Getting the best approximation for b(n)
    ####################################################################
    def _get_bn(self, **kwargs):
        """
        Return the value of b_n, corresponding to a Sersic n index
        Using self.n is n is not provided

        b_n can be derived in two ways:
            * a linear function of n using coefficients present in self._coef_bn_n
            * a fraction representation using coefficients present in self._coef_bn_invn
        The first function is used for n < 0.36, the second one for all other values of n
        """
        n = kwargs.pop("n", self.n)
        return bn(n)

    #######################################################################
    # Function to derive either i0 or ie depending on which one is provided
    #######################################################################

    @property
    def ie(self):
        if not hasattr(self, "_ie"):
            self._ie = 1.0
        return self._ie

    @ie.setter
    def ie(self, val):
        if val is None:
            if self.i0 is None:
                if self.ltot is None:
                    print("Warning: setting ie to default of 1.0")
                    self._ie = 1.0
                else:
                    self._set_ie_from_ltot()
            else:
                self._set_ie()
        else:
            self._ie = val
            self._warn_change_i0ieltot()

        self._set_i0()
        self._set_ltot()

    @property
    def i0(self):
        if not hasattr(self, "_i0"):
            self._i0 = 1.0
        return self._i0

    @i0.setter
    def i0(self, val):
       if val is None:
           if self.ltot is None:
               if self.ie is None:
                   print("Warning: setting i0 to default of 1.0")
                   self._i0 = 1.0
               else:
                   self._set_i0()
           else:
               self._set_i0_from_ltot()
       else:
           self._i0 = val
           self._warn_change_i0ieltot()
       self._set_ie()
       self._set_ltot()

    @property
    def ltot(self):
        if not hasattr(self, "_ltot"):
            self._ltot = 1.0
        return self._ltot

    @ltot.setter
    def ltot(self, val):
        if val is None:
            if self.i0 is None:
                if self.ie is None:
                    print("Warning: setting ltot to default of 1.0")
                    self._ltot = 1.0
                else:
                    self._set_ltot_from_ie()
            else:
                self._set_ie()
                self._set_ltot_from_ie()
        else:
            self._ltot = val
            self._warn_change_i0ieltot()

        self._set_i0_from_ltot()
        self._set_ie()

    def _warn_change_i0ieltot(self):
        lpar = ['i0', 'ie', 'ltot']
        for l in lpar:
            if getattr(self, l) is None:
                clpar = copy.copy(lpar)
                clpar.remove(l)
                print(f"Value of {l} will be derived from other parameters")

    def _set_ie_from_ltot(self):
        self._ie = self.ltot / self._get_ltot_i01() / exp(self.bn)

    def _set_i0_from_ltot(self):
        self._i0 = self.ltot / self._get_ltot_i01()

    def _set_i0(self):
        """
        Provides i0 = central surface brightness value
        """
        self._i0 = self.ie * np.exp(self.bn)

    def _set_ie(self):
        """
        Provides ie = surface brightness at 1 re
        """
        self._ie = self.i0 / exp(self.bn)

    def _get_ltot_i01(self) :
        """
        Provides the total luminosity
        """
        return self.re**2 * pi * 2. * self.n * gamma(2. * self.n) / self.bn**(2.*self.n)

    def _set_ltot(self) :
        """
        Provides the total luminosity
        """
        self._ltot = self.i0 * self._get_ltot_i01()

    def _set_ltot_from_ie(self) :
        """
        Provides the total luminosity
        """
        self._ltot = self.ie * np.exp(self.bn) * self._get_ltot_i01()

    # Deriving Lambdas and Weights ######################################
    def _Lambda_Weight(self) :
        ## Deriving once the weights and abscissa plus the factors for integration
        n = self.n
        [Xquad, Wquad] = scipy.special.orthogonal.ps_roots(self._Nquad)
        self._Xquad = Xquad.astype(nfloat).reshape(self._Nquad,1)
        self._Wquad = Wquad.reshape(self._Nquad,1)
        if n !=1 :
            self._facj = Wquad * Xquad / sqrt(1. - (1. - Xquad*Xquad)**(2.*n/(n-1.)))
            self._lambdaj = 1./(1. - Xquad*Xquad)**(1./(n-1.))
        else :
            self._facj = np.zeros_like(Xquad)
            self._lambdaj = np.zeros_like(Xquad)

    def get_radii(self, r=None, nr=10) :
        if r is None :
            if not hasattr(self, 'r'):
                ## Setup a range for r based on the input variables
                r = np.logspace(log10(self.re / 100.), log10(10. * self.re), nr)
            else :
                r = self.r
        else :
            r = np.asarray(r).ravel()
        rre = r / self.re
        nr = len(r)
        return r, rre, nr

    # Normal PROJECTED Sersic law ##################################################
    def get_sersic(self, r=None) :
        """
        Derive the projected Sersic profile rhop, given an array r on input radii
        The object will thus return self.r and self.rhop accordingly
        """
        r, rre, nr = self.get_radii(r)
        return self.i0 * np.exp(-1. * self.bn * (r / self.re)**(1. / self.n))

    # Sersic rho (spatial) ##################################################
    def get_rho_sersic(self, r=None, Nquad=100) :
        """
        Derive the SPATIAL Sersic profile rho, given an array r on input radii
        The object will thus return self.r and self.rho accordingly
        """
        r, rre, nr = self.get_radii(r)

        ## Derive the necessary values for the integration
        self._Nquad = Nquad
        self._Lambda_Weight()

        ## General case
        if self.n != 1 :
            intrhoL = self._facj * exp(- self._lambdaj * self.bn * rre[...,np.newaxis]**(1./self.n))
            intrhoL *= self.bn * self.i0 * 2. / (pi * self.re * \
                         (self.n - 1.) * rre[...,np.newaxis]**((self.n - 1.) / self.n))
            intrhoL = intrhoL.sum(axis=1)
        ## Special case: n = 1
        else :
            intrhoL = self.bn * self.i0 * scipy.special.k0(self.bn * rre) / (pi * self.re)

        return intrhoL

    # Sersic Integrated mass (spatial) ######################################
    def get_mint_sersic(self, r=None) :
        r, rre, nr = self.get_radii(r.ravel())

        ## Resample r for finer interpolation of rho
        self.rfine = scipy.ndimage.interpolation.zoom(r, 3.0, output=None, order=1, mode='nearest')
        self.rhoLfine = self.get_rho_sersic(self.rfine)
        self.frhoL = interpolate.interp1d(self.rfine, self.rhoLfine)

        if self.n != 1 :
            Mint = self._facj * inc_gamma(2.*self.n+1.,self._lambdaj * self.bn * rre[...,np.newaxis]**(1./self.n)) /  self._lambdaj**(2.*self.n+1.)
            Mint = Mint * self.ltot * 4. / (pi * (self.n - 1.) * gamma(2. * self.n))
            Mint = Mint.sum(axis=0)
        else :
            Mint = np.zeros_like(self.r)
            Mint[0] = scipy.integrate.quad(self._intrhoSersicD, 0., self.r[0], args=(), limit=500)[0]
            for i in range(1, self.nr) :
                Mint[i] = scipy.integrate.quad(self._intrhoSersic, self.r[i-1], self.r[i], args=(), limit=500)[0] 
                Mint[i] += Mint[i-1]
 
        return Mint

    # Sersic rho (spatial) from interpolated function
    def _intrhoSersic(self, r) :
        return 4. * pi * r * r * self.frhoL(r)

    # Sersic rho (spatial) - direct calculation
    def _intrhoSersicD(self, r) :
        return 4. * pi * r * r * self.get_rho_sersic(r)
 
    # Sersic rho approx Trujillo 2002 (spatial) ####################################
    def comprho_Trujillo(self, r=None) :
        r, rre, nr = self.get_radii(r)
        self.nuT, self.pT, self.h1T, self.h2T, self.h3T  = hfunc_Trujillo(self.n)
        return self.i0 * self.bn * 2.0**((self.n-1.)/(2.*self.n)) \
             * fac_ch(self.n, self.nuT, self.pT, self.h3T, self.h2T, self.h1T, 0., rre, self.bn) / (self.re * self.n * pi)
    
    # Sersic rho approx Abel inversion - Glenn (spatial) ####################################
    def get_rho_abel(self, r=None) :
        r, rre, nr = self.get_radii(r)
        self.nuA, self.kA, self.a0A, self.a1A, self.a2A, self.a3A = hfunc_abel(self.n)
        return self._func_rho_abel(rre) 

    def _func_rho_abel(self, rre) :
        return self.i0 * self.bn * 2.0**((self.n-1.)/(2.*self.n)) \
             * fac_ch(self.n, self.nuA, self.kA, self.a0A, 
                     self.a1A, self.a2A, self.a3A, rre, self.bn) / (self.re * self.n * pi)

    # Sersic rho Glenn approx Abel (spatial)
    def _intrhoD_abel(self, r=None) :
        r, rre, Nr = self.get_radii(r)
        return 4. * pi * r * r * self._func_rho_abel(rre)

    # Sersic Abel approx Integrated mass (spatial) ######################################
    def get_mint_abel(self, r=None) :
        """
        Integrated Mass for an Abel approximation of a Sersic law
        """
        r, rre, nr = self.get_radii(r)
        Mint = np.zeros_like(r)
        if r[0] == 0.:
            Mint[0] = 0.
        else:
            Mint[0] = scipy.integrate.quad(self._intrhoD_abel, 0., r[0], args=(), limit=500)[0]

        for i in range(1, nr) :
            Mint[i] = scipy.integrate.quad(self._intrhoD_abel, r[i-1], r[i], args=())[0] 
            Mint[i] += Mint[i-1]
 
        return Mint
 
    # Computing the PS97 (spatial) profile using the right value of p ####################
    def get_rho_PS97(self, r=None) :
        r, rre, nr = self.get_radii(r)
        return rre**(-self.p_PS97) * exp(-self.bn * rre**(1. / self.n)) * self.rho0_PS97

    # Computing the integrated mass for PS97 profile ######################################
    def get_mint_PS97(self, r=None) :
        """
        Integrated Mass for a Prugniel-Simien 97 profile
        """
        r, rre, nr = self.get_radii(r)
        Mint = 4.0 * pi * self.re**3.0 * self.n * self.bn**(self.n * (self.p_PS97 - 3.)) \
                * inc_gamma(self.n * (3. - self.p_PS97), self.bn * rre**(1. / self.n))
        return Mint * self.rho0_PS97

    #=============================================================================================================
    # Return a realisation for a truncated 4. * pi * r^2 * Sersic function
    #=============================================================================================================
    def nbody_from_sersic(self, cutr=1., npoints=1, nsamp=10000):
        """
        Function which returns a sample of points (npoints) which follow
        a r^2 * Sersic distribution truncated at r=cutr
        This uses the cumulative function and the inverse of that function
        onto a random uniform function
    
        Input:
        ------
           n    :   Sersic index
           cutr     :   truncature (positive) in Re units
           npoints  :   Number of points for the output sample
           nsamp    : number of points used for the cumulative function
        """

        # Sampling in r with nsamp points - default to 10000 points to sample well the profile
        sampr = np.linspace(0., cutr * self.re, nsamp)
    
        # First compute the cumulative function of 4*pi*r2*Sersic_deprojected
        fSG = self.get_mint_abel(sampr)

        # Interpolation to get the inverse function
        invF = interpolate.interp1d(fSG, sampr)

        # Now calculating the variables on the Unity sphere at Radius r
        r = invF(np.random.uniform(0, fSG[-1], npoints))
        U = np.asarray(np.random.uniform(-1., 1., size=(npoints,)), dtype=np.float32)
        V = np.asarray(np.random.uniform(0.,1., size=(npoints,)), dtype=np.float32)
        sqU = np.sqrt(1. - U * U)
        theta = 2. * pi * V

        pos = np.zeros((3, npoints), dtype=np.float32)
        pos[0] = r * sqU * cos(theta)
        pos[1] = r * sqU * sin(theta)
        pos[2] = r * U 

        return pos
