#! /usr/bin/env python
"""
This module provides functions useful for Navarro-Frenk-White profile
"""

### Importing the required stuff from different modules
import numpy as np
from numpy import pi, log10, log, exp, sqrt

__version__ = '1.1.0 (March 3, 2012)'

########################################
## Some constant - approximation...
########################################
H0 = 72 # km/s/Mpc
Ggrav = 0.0043225524 # value in (km/s)2. Msun-1 . pc
rhocrit = 3. * H0**2 / (8. * pi * Ggrav * 1.0e3) # in Msun kpc**-3

###################################################
## Profiles in 1D which could be useful
###################################################
## Navarro Frenk White profile
class NFWProfile :
    def __init__(self, c=None, Mvir=None, alpha=-0.12, kc=12.0, **kwargs) :
        """
        NFW profile with Rs in kpc and rho0 in Msun/kpc-3

        We need as input, either:
            c : concentration parameter (adimensional)
               or
            Mvir: virial mass (total) in Msun

        And optionally :
            rsamp: input radial array (radius in kpc) 
                   Default is None: using a default range between Rs/1000 and Rs*1000
                       where Rs is the scale radius (which is derived)

        Will be derived:
           Rs: scale radius (in kpc)
           rho0: normalised density (in Msun/kpc-3)
           rho: density values (in Msun/kpc-3)
           Rvir : virial radius (in kpc)

        Already provided:
           alpha and kc : provides the relation between Mvir and c
               Defaults are alpha=-0.12, and kc=12.0, respectively
        """

        ## Approximation for the transformation between Mvir and c
        self.kc = kc
        self.alpha = alpha

        if c is None :
            if Mvir is None :
                print("ERROR: either c or Mvir should be provided")
                return [0.]
            self.Mvir = Mvir
            self.c = self.kc * (self.Mvir / 1.e12)**(self.alpha)
        else :
            self.c = c
            if Mvir is not None :
                print("ERROR: either c or Mvir should be provided, not BOTH")
                return [0.]
            self.Mvir = 1.0e12 * (self.c / self.kc)**(1. / self.alpha) 

        self.Rvir = (self.Mvir * 3.0 / (4. * pi * 360. * rhocrit * 0.27))**(1./3.)
        self.Rs = self.Rvir / self.c

        if 'rsamp' in kwargs :
            rsamp = kwargs.pop('rsamp')
        else :
            rsamp = None

        self.get_radii(rsamp)
        sel_radius = (self.r < 1000. * self.Rs)
        self.rho0 = self.Mvir / (4. * pi * self.Rs**3 * (log(1. + self.c) - self.c/ (1. + self.c)))
        self.rho = np.zeros_like(self.r)
        self.rho[sel_radius] = self.rho0 / ((self.r[sel_radius] / self.Rs) * (1. + self.r[sel_radius] / self.Rs)**2)

    ###########################################################################
    def get_radii(self, r=None, Nr=1000) :
        if r is None :
            if 'r' not in dir(self) :
                ## Setup a range for r based on the input variables
                self.r = np.logspace(log10(self.Rs/1000.), log10(1000. * self.Rs), Nr)
        else :
            self.r = np.ravel(np.asarray(r))
        self.Nr = len(self.r)
    #=========================================================================

