#! /usr/bin/env python
"""
    This module provides functions useful for Sersic profiles, including a 3d deprojected 
    approximation for a Sersic-n function.
"""

### Importing the required stuff from different modules
import numpy as np
from numpy import pi, log10, log, exp, sqrt
from numpy import float64 as nfloat

import scipy
from scipy import interpolate, special, integrate, ndimage
from scipy.special import gamma, gammainc

from pygme.utils.iodata import get_package_data

__version__ = '1.1.2 (August 21, 2013)'
#__version__ = '1.1.2 (August 14, 2013)'
#__version__ = '1.1.1 (August 30, 2012)'
#__version__ = '1.1.0 (March 3, 2012)'

## Version 1.1.3 : Fixing some docstrings
## Version 1.1.2 : Fixing bug in Integrated Luminosity
## Version 1.1.0 : small fix on the get_I0 function

######################################################################
## Some useful values
######################################################################
Ggrav = 0.0043225524 # value from Remco in (km/s)2. Msun-1 . pc
facPC = nfloat(pi / 0.648)

######################################################################
## Tabulated function to get deprojected Sersic ######################
## Derive the h1/2/3 functions
######################################################################
def hfunc_Abel(n) :
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
#=========================================================================
######################################################################
## h1, h2, h3 function ###############################################
## Derive the h1/2/3 functions
######################################################################
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
#=========================================================================
#######################################################################
# Incomplete Gamma function as it should be (unnormalised)
#######################################################################
## Derive the approximation for the Sersic b parameter
def IncGamma(a,x) :
   return gamma(a) * gammainc(a,x)
#=========================================================================
## Ch function ##################################################
## Derive the Ch function given in Tersic +
def facCh(n, nu, p, a0, a1, a2, a3, x, bn) :
   if n != 1 :
      val = x**(p * (1.- n) / n) / (1. - (a0 + a1 * log10(x) + a2 * log10(x)**2 + a3 * log10(x)**3))
   else :
      val = 1.
   return val * scipy.special.kv(nu, bn * x**(1./n))
#=========================================================================

#######################################################################
# Sersic fitting and tool function
#######################################################################
class SersicProfile :
    def __init__(self, n=1., Re=1.0, Ie=None, I0=None, Ltot=None, rsamp=None, **kwargs) :
        """ Parameters of the Sersic law: n, Ie, Re
              n is adimensional and can be anything from 0 to infinity
                It usually has values between 1 and 8 for real galaxies
              Re in parsec
              Ie in Lsun.pc-2 the surface brightness at 1 Re
                or
              I0 in Lsun.pc-2 the central surface brightness

              bn is adimensional and derived from n

              Other Parameters:
                  rhoL is in Lsun.pc-3
                  Ltot in Lsun
                  rho_average is also in Lsun/pc3
        """

        self.n = nfloat(n)
        self.Re = Re
        self.bn = self.get_bn()
        # Now setting up in order of less importance in case these
        # are multiply defined
        self._Ltot = self._I0 = self._Ie = 1.0
        self.Ltot = Ltot
        self.Ie = Ie
        self.I0 = I0

        ## Compute some input numbers for the Prugniel-Simien function
        if 'pPS97' in kwargs :
            self.pPS97 = kwargs.pop('pPS97')
        else :
            self.pPS97 = 1.0 - 0.6097 / self.n + 0.05463 / (self.n * self.n)
        self.rho0PS97 = self.I0 * self.bn**(self.n * (1. - self.pPS97)) \
                         * gamma(2. * self.n) / (2. * self.Re * gamma(self.n * (3. - self.pPS97)))

        self.r, self.rre, self.Nr = self.get_radii(rsamp)

        ## We derive the projected Sersic profile (direct evaluation)
        self.rhop = self.get_Sersic(self.r)
        ## we then derive the spatial luminosity profile from Simonneau et al
        self.rhoL = self.get_rhoSersic(self.r)

        ## we then derive the spatial luminosity profile from Abel deprojection
        self.rhoAbel = self.get_rhoAbel(self.r)
        self.rhoPS97 = self.get_rhoPS97(self.r)
        self.Mint = self.get_MintSersic(self.r)
        self.MintPS97 = self.get_MintPS97(self.r)
        self.MintAbel = self.get_MintAbel(self.r)

    ## Derive the approximation for the Sersic b parameter
    _coef_bn_invn = [2194697.0 / 30690717750.0, 131. / 1148175.0, 46.0 / 25515.0, 4.0 / 405.0, -1.0 / 3.0]
    _coef_bn_n = [13.43, -19.67, 10.95, -0.8902, 0.01945]

    ####################################################################
    ## Getting the best approximation for b(n)
    ####################################################################
    def get_bn(self, n=None) :
        """
        Return the value of b_n, corresponding to a Sersic n index
        b_n can be derived in two ways:
            * a linear function of n using coefficients present in self._coef_bn_n
            * a fraction representation using coefficients present in self._coef_bn_invn
        The first function is used for n < 0.36, the second one for all other values of n
        """
        if n is None : n = self.n
        else : 
            n = nfloat(n)
            self.n = n
        if (n > 0.36) : 
            bn_poly = np.poly1d(self._coef_bn_invn)
            return (2.0 * n + bn_poly(1. / n))
        else : 
            bn_poly = np.poly1d(self._coef_bn_n)
            return (bn_poly(n))

    #=========================================================================
   #######################################################################
   # Function to derive either I0 or Ie depending on which one is provided
   #######################################################################

    @property
    def Ie(self):
        return self._Ie

    @Ie.setter
    def Ie(self, val):
        if val is None:
            if self.I0 is None:
                if self.Ltot is None:
                    print("Warning: setting Ie to default of 1.0")
                    self._Ie = 1.0
                else:
                    self._set_Ie_from_Ltot()
            else:
                self._set_Ie()
        else:
            self._Ie = val
            self._warn_change_I0IeLtot()

        self._set_I0()
        self._set_Ltot()

    @property
    def I0(self):
        return self._I0

    @I0.setter
    def I0(self, val):
       if val is None:
           if self.Ltot is None:
               if self.Ie is None:
                   print("Warning: setting I0 to default of 1.0")
                   self._I0 = 1.0
               else:
                   self._set_I0()
           else:
               self._set_I0_from_Ltot()
       else:
           self._I0 = val
           self._warn_change_I0IeLtot()
       self._set_Ie()
       self._set_Ltot()

    @property
    def Ltot(self):
        return self._Ltot

    @Ltot.setter
    def Ltot(self, val):
        if val is None:
            if self.I0 is None:
                if self.Ie is None:
                    print("Warning: setting Ltot to default of 1.0")
                    self._Ltot = 1.0
                else:
                    self._set_Ltot_from_Ie()
            else:
                self._set_Ltot_from_I0()
        else:
            self._Ltot = val
            self._warn_change_I0IeLtot()

        self._set_I0_from_Ltot()
        self._set_Ie()

    def _warn_change_I0IeLtot(self):
        lpar = ['I0', 'Ie', 'Ltot']
        for l in lpar:
            if getattr(self, l) is None:
                clpar = copy.copy(lpar)
                clpar.remove(l)
                print(f"Value of {l} will be derived from other parameters")

    def _set_Ie_from_Ltot(self):
        self._Ie = self.Ltot / self._get_Ltot_I01() / exp(self.bn)

    def _set_I0_from_Ltot(self):
        self._I0 = self.Ltot / self._get_Ltot_I01()

    def _set_I0(self):
        """
        Provides I0 = central surface brightness value
        """
        self._I0 = self.Ie * np.exp(self.bn)

    def _set_Ie(self):
        """
        Provides Ie = surface brightness at 1 Re
        """
        self._Ie = self.I0 / exp(self.bn)

    def _get_Ltot_I01(self) :
        """
        Provides the total luminosity
        """
        return self.Re**2 * pi * 2. * self.n * gamma(2. * self.n) / self.bn**(2.*self.n)

    def _set_Ltot(self) :
        """
        Provides the total luminosity
        """
        self._Ltot = self.I0 * self._get_Ltot_I01()

    def _set_Ltot_from_Ie(self) :
        """
        Provides the total luminosity
        """
        self._Ltot = self.Ie * np.exp(self.bn) * self._get_Ltot_I01()

    #=========================================================================
    ## Deriving Lambdas and Weights ######################################
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
    #=========================================================================
    def get_radii(self, r=None, Nr=1000) :
        if r is None :
            if 'r' not in dir(self) :
                ## Setup a range for r based on the input variables
                r = np.logspace(log10(self.Re/100.), log10(10. * self.Re), Nr)
            else :
                r = self.r
        else :
            r = np.asarray(r).ravel()
        rre = r / self.Re
        Nr = len(r)
        return r, rre, Nr
    #=========================================================================

    ## Normal PROJECTED Sersic law ##################################################
    def get_Sersic(self, r=None) :
        """
        Derive the projected Sersic profile rhop, given an array r on input radii
        The object will thus return self.r and self.rhop accordingly
        """
        r, rre, Nr = self.get_radii(r)
        return self.I0 * np.exp(-1. * self.bn * (r / self.Re)**(1. / self.n))
    #=========================================================================
    ## Sersic rho (spatial) ##################################################
    def get_rhoSersic(self, r=None, Nquad=100) :
        """
        Derive the SPATIAL Sersic profile rho, given an array r on input radii
        The object will thus return self.r and self.rho accordingly
        """
        r, rre, Nr = self.get_radii(r)

        ## Derive the necessary values for the integration
        self._Nquad = Nquad
        self._Lambda_Weight()

        ## General case
        if self.n != 1 :
            intrhoL = self._facj * exp(- self._lambdaj * self.bn * rre[...,np.newaxis]**(1./self.n))
            intrhoL *= self.bn * self.I0 * 2. / (pi * self.Re * \
                         (self.n - 1.) * rre[...,np.newaxis]**((self.n - 1.) / self.n))
            intrhoL = intrhoL.sum(axis=1)
        ## Special case: n = 1
        else :
            intrhoL = self.bn * self.I0 * scipy.special.k0(self.bn * rre) / (pi * self.Re)

        return intrhoL
    #=========================================================================
    ## Sersic Integrated mass (spatial) ######################################
    def get_MintSersic(self, r=None) :
        r, rre, Nr = self.get_radii(r.ravel())

        ## Resample r for finer interpolation of rho
        self.rfine = scipy.ndimage.interpolation.zoom(r, 3.0, output=None, order=1, mode='nearest')
        self.rhoLfine = self.get_rhoSersic(self.rfine)
        self.frhoL = interpolate.interp1d(self.rfine, self.rhoLfine)

        if self.n != 1 :
            Mint = self._facj * IncGamma(2.*self.n+1.,self._lambdaj * self.bn * rre[...,np.newaxis]**(1./self.n)) /  self._lambdaj**(2.*self.n+1.)
            Mint = Mint * self.Ltot * 4. / (pi * (self.n - 1.) * gamma(2. * self.n))
            Mint = Mint.sum(axis=0)
        else :
            Mint = np.zeros_like(self.r)
            Mint[0] = scipy.integrate.quad(self._intrhoSersicD, 0., self.r[0], args=(), limit=500)[0]
            for i in range(1, self.Nr) :
                Mint[i] = scipy.integrate.quad(self._intrhoSersic, self.r[i-1], self.r[i], args=(), limit=500)[0] 
                Mint[i] += Mint[i-1]
 
        return Mint
    #=========================================================================
    ## Sersic rho (spatial) from interpolated function
    def _intrhoSersic(self, r) :
        return 4. * pi * r * r * self.frhoL(r)
    #=========================================================================
    ## Sersic rho (spatial) - direct calculation
    def _intrhoSersicD(self, r) :
        return 4. * pi * r * r * self.get_rhoSersic(r)
    #=========================================================================
 
    ## Sersic rho approx Trujillo 2002 (spatial) ####################################
    def comprho_Trujillo(self, r=None) :
        r, rre, Nr = self.get_radii(r)
        self.nuT, self.pT, self.h1T, self.h2T, self.h3T  = hfunc_Trujillo(self.n)
        return self.I0 * self.bn * 2.0**((self.n-1.)/(2.*self.n)) \
             * facCh(self.n, self.nuT, self.pT, self.h3T, self.h2T, self.h1T, 0., rre, self.bn) / (self.Re * self.n * pi)
    #=========================================================================
    ## Sersic rho approx Abel inversion - Glenn (spatial) ####################################
    def get_rhoAbel(self, r=None) :
        r, rre, Nr = self.get_radii(r)
        self.nuA, self.kA, self.a0A, self.a1A, self.a2A, self.a3A = hfunc_Abel(self.n)
        return self._func_rhoAbel(rre) 

    def _func_rhoAbel(self, rre) :
        return self.I0 * self.bn * 2.0**((self.n-1.)/(2.*self.n)) \
             * facCh(self.n, self.nuA, self.kA, self.a0A, self.a1A, self.a2A, self.a3A, rre, self.bn) / (self.Re * self.n * pi)
    #=========================================================================
    ## Sersic rho Glenn approx Abel (spatial)
    def _intrhoD_Abel(self, r=None) :
        r, rre, Nr = self.get_radii(r)
        return 4. * pi * r * r * self._func_rhoAbel(rre)
    #=========================================================================
    ## Sersic Abel approx Integrated mass (spatial) ######################################
    def get_MintAbel(self, r=None) :
        """
        Integrated Mass for an Abel approxiamation of a Sersic law
        """
        r, rre, Nr = self.get_radii(r)
        Mint = np.zeros_like(r)
        Mint[0] = scipy.integrate.quad(self._intrhoD_Abel, 0., r[0], args=(), limit=500)[0]
        for i in range(1, self.Nr) :
            Mint[i] = scipy.integrate.quad(self._intrhoD_Abel, r[i-1], r[i], args=())[0] 
            Mint[i] += Mint[i-1]
 
        return Mint
    #=========================================================================
 
    ## Computing the PS97 (spatial) profile using the right value of p ####################
    def get_rhoPS97(self, r=None) :
        r, rre, Nr = self.get_radii(r)
        return rre**(-self.pPS97) * exp(- self.bn * rre**(1. / self.n)) * self.rho0PS97
    #=========================================================================
    ## Computing the integrated mass for PS97 profile ######################################
    def get_MintPS97(self, r=None) :
        """
        Integrated Mass for a Prugniel-Simien 97 profile
        """
        r, rre, Nr = self.get_radii(r)
        Mint = 4.0 * pi * self.Re**3.0 * self.n * self.bn**(self.n * (self.pPS97 - 3.)) \
                * IncGamma(self.n * (3.-self.pPS97), self.bn * rre**(1./self.n))
        return Mint * self.rho0PS97
    #=========================================================================
