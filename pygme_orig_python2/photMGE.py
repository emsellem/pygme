#!/usr/bin/python
"""
This module specifically deals with the photometric quantities
related to a Multi Gaussian Expansion models (Monnet et al.  1992, Emsellem et al. 1994). It
makes use of the MGE class defined in pygme.py module.
It includes the derivation of projected and deprojected photometry.

WARNING: this module is being uptdated regularly and may still contains some obvious bugs. A stable version will
be available hopefully before the end of 2012.

For questions, please contact Eric Emsellem at eric.emsellem@eso.org
"""

"""
Importing the most import modules
This MGE module requires NUMPY and SCIPY
"""
try:
    import numpy as np
except ImportError:
    raise Exception("numpy is required for pygme")

from numpy import exp, sqrt

try:
    from scipy import special
except ImportError:
    raise Exception("scipy is required for pygme")

from rwcfor import floatMGE
from paramMGE import paramMGE
from mge_miscfunctions import print_msg

__version__ = '1.1.1 (21/08/2013)'
#__version__ = '1.1.0 (28/08/2012)'
#__version__ = '1.0.0 (08/01/2012)'

# Version 1.1.1 Changed imin imax into ilist
# Version 1.1.0 Revised to only have R and Z for visible modules
#               and not R2 and Z2
# Version 1.0.0 extracted from the older pygme.py

class photMGE(paramMGE):
    def __init__(self, infilename=None, indir=None, saveMGE=None, **kwargs) :
        paramMGE.__init__(self, infilename=infilename, indir=indir, saveMGE=saveMGE, **kwargs)

    ##################################################################
    ### Derive the spatial and projected densities
    ##################################################################
    ### MASS - 1 G  --------------------------------------------------
    ### Deriving the spatial mass density for 1 gaussian
    def _rho3D_1G_fromR2Z2(self, R2, Z2, ind) :
        """
        Spatial Mass Density in Mass/pc-2/arcsec-1
          for 1 Gaussian only: ind is the indice of that gaussian
          R2 and Z2 are grids of R*R and Z*Z (should have the same size) [in arcseconds]
        """
        if self._findGauss3D == 0 : 
            print_msg("No Spatial Gaussians yet", 2)
            return R2*0.
        return self.Parc[ind] * exp(- (R2 + Z2 / self.QxZ2[ind]) / self._pParam.dSig3Darc2[ind])  # in Mass/pc-2/arcsec-1

    ### Deriving the projected mass density for 1 gaussian
    def _rho2D_1G_fromX2Y2(self, X2, Y2, ind) :
        """
        Projected Mass Density in Mass/pc-2
          for 1 Gaussian only: ind is the indice of that gaussian
          R2 and Z2 are grids of R*R and Z*Z (should have the same size) [in arcseconds]
        """
        if self._findGauss2D == 0 : 
            print_msg("No projected Gaussians yet", 2)
            return X2*0.
        return self.Pp[ind] * exp(- (X2 + Y2 / self.Q2D2[ind]) / self.dSig2Darc2[ind])  # in Mass/pc-2
    ###=================================================================
    ##################################################################
    ### Derive the spatial and projected densities
    ##################################################################
    ### MASS - 1 G  --------------------------------------------------
    ### Deriving the spatial mass density for 1 gaussian
    def rho3D_1G(self, R, Z, ind) :
        """
        Spatial Mass Density in Mass/pc-2/arcsec-1
          for 1 Gaussian only: 
              ind is the indice of that gaussian
              R and Z are cylindrical coordinates [in arcseconds]
        """
        return self._rho3D_1G_fromR2Z2(R*R, Z*Z, ind)

    ### Deriving the projected mass density for 1 gaussian
    def rho2D_1G(self, X, Y, ind) :
        """
        Projected Mass Density in Mass/pc-2
          for 1 Gaussian only: 
              ind is the indice of that gaussian
              X, Y are projected coordinates [in arcseconds]
        """
        return self._rho2D_1G_fromX2Y2(X*X, Y*Y, ind)
    ###=================================================================

    ### MASS - ALL-----------------------------------------------------
    ### Deriving the spatial mass density for all
    def _rho3D_fromR2Z2(self, R2, Z2, ilist=None) :
        """
        Spatial Mass Density in Mass/pc-2/arcsec-1
          R2 and Z2 are grids of R*R and Z*Z (should have the same size) [in arcseconds]
        """
        if self._findGauss3D == 0 : 
            print_msg("No Spatial Gaussians yet", 2)
            return R2*0.
        rho = np.zeros_like(R2)
        for i in ilist :
            rho += self._rho3D_1G_fromR2Z2(R2, Z2, i)
        return rho  # in Mass/pc-2/arcsec-1

    ### Deriving the projected mass density for all
    def _rho2D_fromX2Y2(self, X2, Y2, ilist=None) :
        """
        Projected Mass Density in Mass/pc-2
          R2 and Z2 are grids of R*R and Z*Z (should have the same size) [in arcseconds]
        """
        if self._findGauss2D == 0 : 
            print_msg("No projected Gaussians yet", 2)
            return X2*0.
        rho2D = np.zeros_like(X2)
        for i in ilist :
            rho2D += self._rho2D_1G_fromX2Y2(X2, Y2, i)
        return rho2D  # in Mass/pc-2
    ###=============================================================================
    ### MASS - ALL-----------------------------------------------------
    ### Deriving the spatial mass density for all
    def rho3D(self, R, Z, ilist=None) :
        """
        Spatial Mass Density in Mass/pc-2/arcsec-1

        Input :
             R, Z the cylindrical coordinates [in arcseconds]
             ilist indices for the Gaussians
        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)

        return self._rho3D_fromR2Z2(R*R, Z*Z, ilist=ilist)

    ### Deriving the projected mass density for all
    def rho2D(self, X, Y, ilist=None) :
        """
        Projected Mass Density in Mass/pc-2

        Input :
             X, Y the projected coordinates [in arcseconds]
             ilist indices for the Gaussians
        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)

        return self._rho2D_fromX2Y2(X*X, Y*Y, ilist=ilist)
    ###=============================================================================

    ### LUMINOSITY - 1 G ------------------------------------------------------------
    ###=============================================================================
    ### Deriving the spatial luminosity density for 1 gaussian
    def _rhoL3D_1G_fromR2Z2(self, R2, Z2, ind) :
        """
        Spatial LUMINOSITY distribution in Lum.pc-2/arcsec-1
           for 1 Gaussian only: ind is the indice of that gaussian
          R2 and Z2 are grids of R*R and Z*Z (should have the same size) [in arcseconds]
        """
        if self._findGauss3D == 0 : 
            print_msg("No Spatial Gaussians yet", 2)
            return R2*0.
        return self.Imax3Darc[ind] * exp(- (R2 + Z2 / self.QxZ2[ind]) / self._pParam.dSig3Darc2[ind])  # I in Lum.pc-2/arcsec-1
    ### Deriving the spatial luminosity density for 1 gaussian
    def rhoL3D_1G(self, R, Z, ind) :
        """
        Spatial LUMINOSITY distribution in Lum.pc-2/arcsec-1 for 1 Gaussian only

        Input :
            ind is the indice of that gaussian
            R, Z are the cylindrical coordinates [in arcseconds]
        """
        return self._rhoLspatial_1G_fromR2Z2(R*R, Z*Z, ind)
    ###=============================================================================
    ### Deriving the projected luminosity density for 1 gaussian
    def _rhoL2D_1G_fromX2Y2(self, X2, Y2, ind) :
        """
        Projected LUMINOSITY distribution in Lum.pc-2
           for 1 Gaussian only: ind is the indice of that gaussian
          R2 and Z2 are grids of R*R and Z*Z (should have the same size) [in arcseconds]
        """
        if self._findGauss2D == 0 : 
            print_msg("No projected Gaussians yet", 2)
            return X2*0.
        return self.Imax2D[ind] * exp(- (X2 + Y2 / self.Q2D2[ind]) / self.dSig2Darc2[ind])  # I in Lum.pc-2
    ###=============================================================================
    ### Deriving the projected luminosity density for 1 gaussian
    def rhoL2D_1G(self, X, Y, ind) :
        """
        Projected LUMINOSITY distribution in Lum.pc-2
           for 1 Gaussian only

        Input :
            ind is the indice of that gaussian
            X, Y are the projected coordinates [in arcseconds]
        """
        return self._rhoL2D_1G_fromX2Y2(X*X, Y*Y, ind)
    ###=============================================================================

    ### LUMINOSITY - ALL ------------------------------------------------------------
    ### Deriving the spatial luminosity density for all
    def _rhoL3D_fromR2Z2(self, R2, Z2, ilist=None) :
        """
        Spatial LUMINOSITY distribution in Lum.pc-2/arcsec-1
            R2, Z2 are the squares of the cylindrical coordinates [in arcseconds]
            ilist: indices for the Gaussians
        """
        if self._findGauss3D == 0 : 
            print_msg("No Spatial Gaussians yet", 2)
            return R2*0.
        rhoL = np.zeros_like(R2)
        for i in ilist :
            rhoL += self._rhoLspatial_1G_fromR2Z2(R2, Z2, i)
        return rhoL  # I in Lum.pc-3
    ### Deriving the spatial luminosity density for all
    def rhoL3D(self, R, Z, ilist=None) :
        """
        Spatial LUMINOSITY distribution in Lum.pc-2/arcsec-1
        
        Input :
            R, Z = cylindrical coordinates [in arcseconds]
            ilist: indices for the Gaussians
        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)

        return self._rhoL3D_fromR2Z2(R*R, Z*Z, ilist=ilist)
    ###=============================================================================

    ### Deriving the projected luminosity density for all
    def _rhoL2D_fromX2Y2(self, X2, Y2, ilist=None):
        """
        Projected LUMINOSITY distribution in Lum.pc-2
            X2, Y2 are the squares of the projected coordinates [in arcseconds]
            ilist: indices for the Gaussians
        """
        if self._findGauss2D == 0 : 
            print_msg("No projected Gaussians yet", 2)
            return X2*0.
        rhoL2D = np.zeros_like(X2)
        for i in ilist :
            rhoL2D += self._rhoL2D_1G_fromX2Y2(X2, Y2, i)
        return rhoL2D  # I in Lum.pc-2
    ###=============================================================================
    ### Deriving the projected luminosity density for all
    def rhoL2D(self, X, Y, ilist=None):
        """
        Projected LUMINOSITY distribution in Lum.pc-2
        
        Input :
            X, Y = projected coordinates [in arcseconds]
            ilist: indices for the Gaussians
        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)

        return self._rhoL2D_fromX2Y2(X*X, Y*Y, ilist=ilist)
     ###=============================================================================

    ### INTEGRATED LUMINOSITY - ONLY IN Z -------------------------------------------------
    ### Deriving the surface Lum for 1 gaussian, R is in arcsec
    def FluxSurf_1G(self, R, ind) :
        """
        Flux Surface density in Lum.pc-2
           for 1 Gaussian only: ind is the indice of that gaussian
           R is a grid of radii in arcseconds
        """
        if self._findGauss3D == 0 : 
            print_msg("No Spatial Gaussians yet", 2)
            return R * 0.
        R2 = R * R
        return self.Imax3Darc[ind] * sqrt(2. * np.pi) * self.qSig3Darc[ind] * exp(- R2 / self._pParam.dSig3Darc2[ind])  # in Lum.pc-2

    ### Deriving the integrated Lum (Zcut) for 1 gaussian, R and Z are in arcsec
    def _rhointLZ_1G(self, R, Zcut, ind):
        """
        Integrated luminosity (within Zcut in arcsec) in Lum.pc-2
           for 1 Gaussian only: ind is the indice of that gaussian
           R is a grid of radii in arcseconds
        """
        if self._findGauss3D == 0 : 
            print_msg("No Spatial Gaussians yet", 2)
            return R * 0.
        return self.FluxSurf_1G(R, ind) * floatMGE(special.erf(Zcut / self._pParam.dqSig3Darc[ind]))  # in Lum.pc-2

    ### Deriving the integrated Lum (Zcut) for all, R and are in arcsec
    def rhointLZ(self, R, Zcut, ilist=None) :
        """
        Integrated luminosity (within Zcut in arcsec) in Lum.pc-2
           R is a grid of radii in arcseconds
        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)

        if self._findGauss3D == 0 : 
            print_msg("No Spatial Gaussians yet", 2)
            return R * 0.
        rhointL = 0.
        for i in ilist :
            rhointL += self._rhointLZ_1G(R, Zcut, i)
        return rhointL  # in Lum.pc-2
    ###=============================================================================

    ### INTEGRATED MASS - ONLY IN Z --------------------------------------------------------
    ### Deriving the surface Mass for 1 gaussian, R is in arcsec
    def MassSurf1(self, R, ind) :
        """
        Mass Surface density in Mass.pc-2
           for 1 Gaussian only: ind is the indice of that gaussian
           R is a grid of radii in arcseconds
        """
        if self._findGauss3D == 0 : 
            print_msg("No Spatial Gaussians yet", 2)
            return R * 0.
        R2 = R * R
        return self.Parc[ind] * sqrt(2. * np.pi) * self.qSig3Darc[ind] * exp(- R2 / self._pParam.dSig3Darc2[ind])  # in Mass.pc-2

    ### Deriving the integrated Mass (Zcut) for 1 gaussian, R and are in arcsec
    def rhointMZ1(self, R, Zcut, ind) :
        """
        Integrated Mass (within Zcut in arcsec) in Mass.pc-2
           for 1 Gaussian only: ind is the indice of that gaussian
           R is a grid of radii in arcseconds
        """
        if self._findGauss3D == 0 : 
            print_msg("No Spatial Gaussians yet", 2)
            return R * 0.
        return self.MassSurf1(R, ind) * float(special.erf(Zcut / self._pParam.dqSig3Darc[ind]))  # in Mass.pc-2

    ### Deriving the integrated Mass (Rcut, Zcut) for all, R and are in arcsec
    def rhointMZ(self, R, Zcut, ilist=None) :
        """
        Integrated Mass (within Zcut in arcsec) in Mass.pc-2
           R is a grid of radii in arcseconds
        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)

        if self._findGauss3D == 0 : 
            print_msg("No Spatial Gaussians yet", 2)
            return R * 0.

        rhointM = 0.
        for i in ilist :
            rhointM += self.rhointMZ1(R, Zcut, i)
        return rhointM  # in Mass.pc-2
    ###=============================================================================

    ### INTEGRATED LUMINOSITY - SPHERE ALL -------------------------------------------------

    ### Deriving the integrated Lum (mcut) for all, m in arcsec
    def rhoSphereintL(self, mcut, ilist=None) :
        """
        Integrated LUMINOSITY truncated within a spheroid of m=mcut (in arcsec)
        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)

        if self._findGauss3D == 0 : 
            print_msg("No Spatial Gaussians yet", 2)
            return 0.

        rhointL = 0.
        for i in ilist :
            rhointL += self.rhoSphereintL_1G(mcut, i)
        return rhointL
    ###=============================================================================

    ### Deriving the integrated Mass (mcut) for all, m in arcsec
    def rhoSphereintM(self, mcut, ilist=None) :
        """
        Integrated Mass truncated within a spheroid of m=mcut (in arcsec)
        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)

        if self._findGauss3D == 0 : 
            print_msg("No Spatial Gaussians yet", 2)
            return 0.

        rhointM = 0.
        for i in ilist :
            rhointM += self.rhoSphereintM_1G(mcut, i)
        return rhointM
    ###=============================================================================

    ### Deriving the integrated Lum (Rcut, Zcut) for all, R and are in arcsec
    def rhointL(self, Rcut, Zcut, ilist=None) :
        """
        Integrated LUMINOSITY truncated within a cylindre defined by Rcut, Zcut (in arcsec)
        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)

        if self._findGauss3D == 0 : 
            print_msg("No Spatial Gaussians yet", 2)
            return 0.

        rhointL = 0.
        for i in ilist :
            rhointL += self.rhointL_1G(Rcut, Zcut, i)
        return rhointL
    ###=============================================================================

    ### INTEGRATED MASS - ALL --------------------------------------------------------
    ### Deriving the integrated Mass (Rcut, Zcut) for all, R and are in arcsec
    def rhointM(self, Rcut, Zcut, ilist=None) :
        """
        Integrated Mass truncated within a cylindre defined by Rcut, Zcut (in arcsec)
        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)

        if self._findGauss3D == 0 : 
            print_msg("No Spatial Gaussians yet", 2)
            return 0.

        rhointM = 0.
        for i in ilist :
            rhointM += self.rhointM_1G(Rcut, Zcut, i)
        return rhointM
    ###=============================================================================
