#!/usr/bin/python
"""
This module specifically deals with the dynamical quantities
related to Multi Gaussian Expansion models (Monnet et al.  1992, Emsellem et al. 1994).
It includes the derivation of projected and deprojected photometry, and
the derivation of velocity moments via the Jeans Equations.

"""

"""
Importing the most import modules
This MGE module requires numpy and scipy
"""
try:
    import numpy as np
except ImportError:
    raise Exception("numpy is required for pygme")

from numpy import shape
from numpy import cos, sin, exp, sqrt

try:
    from scipy import special
except ImportError:
    raise Exception("scipy is required for pygme")

from numpy import inf

from .rwcfor import floatMGE, floatG

from pygme.photMGE import photMGE
from pygme.mge_miscfunctions import quadrat_ps_roots

__version__ = '1.1.2 (21/08/2013)'
#__version__ = '1.1.1 (22/01/2013)'
#__version__ = '1.1.0 (27/08/2012)'
#__version__ = '1.0.0 (08/01/2012)'

# Version 1.1.2 Changed imin,imax into ilist
# Version 1.1.1 Small minor fixes
# Version 1.1.0 Number of small changes including for visible modules
# Version 1.0.0 extracted from the older pygme.py

## This is a maximum value to include in the derivation of exponential(x^2) and erfc(x) functions
## Beyond this value, an analytic approximation is replacing the exact expression
_Maximum_Value_forEXPERFC = 20.0

class dynMGE(photMGE) :
    """
    This class contains all the dynamics-related quantities, from circular velocities, epicycle
    frequencies, Jeasn Equations.
    """
    def __init__(self, infilename=None, indir=None, saveMGE=None, **kwargs) :
        photMGE.__init__(self, infilename=infilename, indir=indir, saveMGE=saveMGE, **kwargs)

    ##########################################################################################################
    ### Compute the  terms for Jeans ###
    ######################################
    ## First order moment ===================================================================
    ## Two integrals: line of sight and adimensional one with variable T between 0 and 1
    def _IntMu1(self, T, R2, Z2, ilist=None) :
        T2 = T * T
        T2Bij_soft = 1. - self._dParam.Bij_soft * T2
        facdenom = 1. - self.e2 * T2

        Integrand = np.zeros_like(R2)
        for j in range(self.nStarGauss) :
            expfac = self._pParam.qParc[j] * exp(- (R2 + Z2 / facdenom[j]) * T2 / self._dParam.dSig3Darc2_soft[j]) / sqrt(facdenom[j])
            for i in ilist :
                Integrand +=  self.rhoL[i] * expfac * (self.e2[i] - self.e2[j] * T2) / T2Bij_soft[i,j] # L*L*pc-4*arcsec-2

        return Integrand * T2

    def _IntlosMu1(self, LOS, X2, Y, cosi, sini, Xquad, Wquad, ilist=None) :
        R2 = -Y * cosi + LOS * sini
        R2 = X2 + R2 * R2
        Z2 = Y * sini + LOS * cosi
        Z2 = Z2 * Z2

        ## INTEGRAL via quadrature
        result = Wquad[0] * self._IntMu1(Xquad[0], R2, Z2, ilist)
        for i in range(1,len(Xquad)) :
            result += Wquad[i] * self._IntMu1(Xquad[i], R2, Z2, ilist)
        Integrand = sqrt(self.rhoLT * result)

        return Integrand

    def _Mu1(self, X, Y, inclin=90., ilist=None) :
        X2 = X * X
        Y2 = Y * Y
        inclin_rad = inclin * np.pi / 180.
        sini = sin(inclin_rad)
        cosi = cos(inclin_rad)
        [Xquad, Wquad] = quadrat_ps_roots(self.Nquad)
        rhop = self._rhoL2D_fromX2Y2(X2, Y2, ilist)

        result = np.zeros_like(X2)
        for i in range(shape(X2)[0]) :
            for j in range(shape(X2)[1]) :
                print("%f %f \n" %(X[i,j], Y[i,j]))
                ### INTEGRAL between -infinity and infinity along the line of sight
                result[i,j] = float(scipy.integrate.quad(self._IntlosMu1, -inf, inf, epsabs=1.e-01, epsrel=1.e-01, args=(X2[i,j], Y[i,j], cosi, sini, Xquad, Wquad, ilist))[0])

        return sqrt(4. * np.pi * self.G) * result * sini * X / rhop
    ###=============================================================================

    ## Second order moment ========================================================
    ## Only 1 integral: variable T between 0 and 1
    def _IntMu2(self, T, X2, Y2, cosi2, sini2, ilist=None) :
        T2 = T * T
        T4 = T2 * T2
        T2Bij_soft = 1. - self._dParam.Bij_soft * T2
        facdenom = 1. - self.e2 * T2
        e2T4 = T4 * self.e2 / (self._pParam.dSig3Darc2 * facdenom)

        Integrand = np.zeros_like(X2)
        for j in range(self.nGauss) :
            for i in ilist :
                A = T2 / self._dParam.dSig3Darc2_soft[j] + 1. / self._pParam.dSig3Darc2[i]  # in arcsec-2
                B = e2T4[j] + self.e2q2dSig3Darc2[i]              # in arcsec-2
                ABcosi2 = A + B * cosi2                      # in arcsec-2
                varexp = -A * (X2 + Y2 * (A + B) / ABcosi2)  # adimensionless
                denom = T2Bij_soft[i,j] * sqrt(facdenom[j] * ABcosi2) # in arcsec-1
                num = self._dParam.q2Sig3Darc2[i] + X2 * sini2 * (self.e2[i] - T2 * self.e2[j]) # in arcsec^2
                Integrand += self._pParam.qParc[j] * self.Imax3Darc[i] * num * exp(varexp) / denom # L*M*pc-4*arcsec

        return Integrand * T2

    def _Mu2(self, X, Y, inclin=90., ilist=None) :
        ilist = self._set_ilist(ilist)
        ### First compute the gaussian quadrature points, and weights
        [Xquad, Wquad] = quadrat_ps_roots(self.Nquad)
        X2 = X * X
        Y2 = Y * Y
        self.rhop = np.sum(self.Imax2D[ilist] * exp(- (X2[...,np.newaxis] + Y2[...,np.newaxis] / self.Q2D2[ilist]) / self.dSig2Darc2[ilist]))

        inclin_rad = inclin * np.pi / 180.
        sini = sin(inclin_rad)
        cosi = cos(inclin_rad)
        sini2 = sini * sini
        cosi2 = cosi * cosi

        result = np.zeros_like(X)
        for i in range(self.Nquad) :
            result += Wquad[i] * self._IntMu2(Xquad[i], X2, Y2, cosi2, sini2, ilist)
        return 4. * np.pi**1.5 * self.G * result / self.rhop # en km^2.s-2
    ###=============================================================================

    ######################################################################################
    ### Compute the gravitational potential  ###
    ############################################
    def _IntPot(self, T, R2, Z2, ilist=None) :
        """ Integrand for the Gravitational potential
        """
        ilist = self._set_ilist(ilist)
        T2 = T * T
        denom = 1. - self.e2[ilist] * T2
        Integrand = self._dParam.Sig3Darc2_soft[ilist] * self._pParam.qParc[ilist] * exp(- (R2[...,np.newaxis] + Z2[...,np.newaxis] / denom) * T2 / self._dParam.dSig3Darc2_soft[ilist]) / sqrt(denom)
        return np.sum(Integrand, axis=-1)

    def Potential(self, R, Z, ilist=None) :
        """
        Return, for a grid of R and Z the Gravitational potential in km^2.s^2
        R and Z should be in arcseconds

        :param R: cylindrical radius (float or array) in arcseconds
        :param Z: vertical height (float or array) in arcseconds
        :param ilist: list of indices for the Gaussians to consider (0 to Ngauss-1)

        :returns: Gravitational potential :math:`\Phi` in Units of km^2.s-2
        :rtype: float or array of float depending on input
        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)
        ### First compute the gaussian quadrature points, and weights
        [Xquad, Wquad] = quadrat_ps_roots(self.Nquad)
        R2 = R*R
        Z2 = Z*Z
        result = np.sum(Wquad[i] * self._IntPot(Xquad[i], R2, Z2, ilist) for i in range(self.Nquad))

        if (self.Mbh > 0.) :
            mask = (R2 + Z2 == 0.)
            result[~mask] += self.facMbh / sqrt(R2[~mask] + Z2[~mask] + self.SoftarcMbh2)

        return -4. * np.pi * self.G * result  # en km^2.s-2

    ############# ESCAPE VELOCITY = SQRT(2 * PHI) ####################################
    def Vescape(self, R, Z, ilist=None) :                     # R and Z should be in arcsec
        """
        Return, for a grid of R and Z the escape velocity in Km/s
        R and Z should be in arcseconds

        :param R: cylindrical radius (float or array) in arcseconds
        :param Z: vertical height (float or array) in arcseconds

        :returns: float/array -- Escape velocity [km.s-1]
        """
        return sqrt(-2. * self.Potential(R, Z, ilist))  # en km.s-1

    ############ CIRCULAR VELOCITY  ##################################################
    def _IntVc(self, T, R2, ilist=None) :
        T2 = T * T
        Integrand = np.zeros_like(R2)
        denom = 1. - self.e2 * T2
        for i in ilist :
            Integrand += self._pParam.qParc[i] * exp(- R2 * T2 / self._dParam.dSig3Darc2_soft[i]) / sqrt(denom[i])
        return Integrand * T2

    def Vcirc(self, R, ilist=None, Mbh=True) :
        """
        Derive the circular velocity for the MGE model taking into account
        Only the Gaussians from the indice list (ilist) - counting from 0
        A softening can be included (eps in pc)

        :param R: cylindrical radius (float or array) in arcseconds
        :param ilist: list of indices of Gaussians to count

        :returns: float/array -- Circular velocity [km.s-1]
        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)
        ### First compute the gaussian quadrature points, and weights
        [Xquad, Wquad] = quadrat_ps_roots(self.Nquad)
        R2 = R*R
        result = R2 * np.sum(Wquad[i] * self._IntVc(Xquad[i], R2, ilist) for i in range(self.Nquad))

        if (self.Mbh > 0.) & Mbh:
            mask = (R == 0.)
            result[mask] += self.facMbh / np.maximum(1.e-2, self.SoftarcMbh)
            result[~mask] += self.facMbh / sqrt(R2[~mask] + self.SoftarcMbh2)

        return sqrt(result * self.PIG)  # en km.s-1
    ### ==============================================================================

    ##################################################################
    ### Compute the acceleration in R and Z for orbit integration  ###
    ##################################################################
    def _IntaccR(self, T, R2, Z2, ilist=None) :
        T2 = T * T
        Integrand = np.zeros_like(R2)
        denom = 1. - self.e2 * T2
        for i in ilist :
            Integrand += self._pParam.qParc[i] * exp(- (R2 + Z2 / denom[i]) * T2 / self._dParam.dSig3Darc2_soft[i]) / sqrt(denom[i])
        return Integrand * T2
    #===========================================================

    #######################################################
    def _IntaccZ(self, T, R2, Z2, ilist=None) :
        T2 = T * T
        Integrand = np.zeros_like(R2)
        denom = 1. - self.e2 * T2
        for i in ilist :
            Integrand += self._pParam.qParc[i] * exp(- (R2 + Z2 / denom[i]) * T2 / self._dParam.dSig3Darc2_soft[i]) / (denom[i])**(1.5)
        return Integrand * T2
    #===========================================================

    #######################################################
    def _accR(self, R, Z, ilist=None) :
        ### First compute the gaussian quadrature points, and weights
        [Xquad, Wquad] = quadrat_ps_roots(self.Nquad)
        result = np.zeros_like(R)
        R2 = R*R
        Z2 = Z*Z
        for i in range(self.Nquad) :
            result += Wquad[i] * self._IntaccR(Xquad[i], R2, Z2, ilist)
        return self.PIG * result * R / self.pc_per_arcsec  # en km^2.s-2.pc-1
    #===========================================================

    #######################################################
    def _accZ(self, R, Z, ilist=None) :
        ### First compute the gaussian quadrature points, and weights
        [Xquad, Wquad] = quadrat_ps_roots(self.Nquad)
        result = np.zeros_like(R)
        R2 = R*R
        Z2 = Z*Z
        for i in range(self.Nquad) :
            result += Wquad[i] * self._IntaccZ(Xquad[i], R2, Z2, ilist)
        return self.PIG * result * Z / self.pc_per_arcsec  # en km^2.s-2.pc-1
    #===========================================================
    ##################################################################
    ### Compute the second derivative of the potential with R
    ##################################################################
    def _Intd2Potd2R(self, T, R2, Z2, ilist=None) :
        T2 = T * T
        Integrand = np.zeros_like(R2)
        denom = 1. - self.e2 * T2
        for i in ilist :
            Integrand += self._pParam.qParc[i] * exp(- (R2 + Z2 / denom[i]) * T2 / self._dParam.dSig3Darc2_soft[i]) * (1. - R2 * T2 / self._dParam.Sig3Darc2_soft[i]) / sqrt(denom[i])
        return Integrand * T2
    #===========================================================
    #######################################################
    def _d2Potd2R(self, R, Z, ilist=None) :
        ### First compute the gaussian quadrature points, and weights
        [Xquad, Wquad] = quadrat_ps_roots(self.Nquad)
        result = np.zeros_like(R)
        R2 = R*R
        Z2 = Z*Z
        for i in range(self.Nquad) :
            result += Wquad[i] * self._Intd2Potd2R(Xquad[i], R2, Z2, ilist)
        return self.PIG * result / (self.pc_per_arcsec*self/pc_per_arcsec)  # en km^2.s-2.pc-2
    #===========================================================
    #######################################################
    # FluxDensity for certain gaussians
    #######################################################
    def _FluxDensity(self, R2, Z2, ilist=None) :
        """
        Function useful for the integration of dynamical quantities
        """
        ### Compute .rho and .rhoT the individual and total M density on the grid
        rhoL = np.zeros((self.nGauss, len(R2)), floatMGE)
        rhoLT = np.zeros_like(R2)
        for i in ilist :
            rhoL[i] = self._rhoL3D_1G_fromR2Z2(R2, Z2, i)
        rhoLT = np.sum(rhoL, axis=0)
        return rhoL, rhoLT
        ## WARNING: rho is in Mass/pc-2/arcsec-1
    #===========================================================

    #######################################################
    # MassDensity for certain gaussians
    #######################################################
    def _MassDensity(self, R2, Z2, ilist=None) :
        """
        Function useful for the integration of dynamical quantities: QToomre
        """
        ### Compute .rho and .rhoT the individual and total M density on the grid
        rho = np.zeros((self.nGauss, len(R2)), floatMGE)
        rhoT = np.zeros_like(R2)
        for i in ilist :
            rho[i] = self._rho3D_1G_fromR2Z2(R2, Z2, i)
        rhoT = np.sum(rho, axis=0)
        return rho, rhoT
        ## WARNING: rho is in Mass/pc-2/arcsec-1
    #===========================================================
    ############ OmegaR  ##################################################
    def _IntOmega(self, T, R2, ilist=None) :
        T2 = T * T
        Integrand = np.zeros_like(R2)
        denom = 1. - self.e2 * T2
        for i in ilist :
            Integrand += self._pParam.qParc[i] * exp(- R2 * T2 / self._dParam.dSig3Darc2_soft[i]) / sqrt(denom[i])
        return Integrand * T2

    def Omega(self, R, ilist=None) :
        """ Returns :math:`\Omega`, the circular frequency, for a grid of R.
        R should be in arcseconds

        :param R: cylindrical radius (float or array) in arcseconds
        :param Z: vertical height (float or array) in arcseconds

        :returns: :math:`\Omega` Circular frequency [km.s-1.pc-1]
        :rtype: float or array of float depending on input

        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)
        ### First compute the gaussian quadrature points, and weights
        [Xquad, Wquad] = quadrat_ps_roots(self.Nquad)
        R2 = R*R
        result = np.sum(Wquad[i] * self._IntVc(Xquad[i], R2, ilist) for i in range(self.Nquad))

        if (self.Mbh > 0.) :
            mask = (R == 0.)
            result[mask] += self.facMbh / np.maximum(1.e-4, self.SoftarcMbh**3)
            result[~mask] += self.facMbh / (R2[~mask] + self.SoftarcMbh2)**(3.0/2.0)

        return sqrt(self.PIG * result) / self.pc_per_arcsec  # en km.s-1.pc-1
    ### ==============================================================================
    ##################################################################
    ### Compute the epicycle frequency kappa (squared)
    ##################################################################
    def _Intkappa(self, T, R2, ilist=None) :
        """
        Integrand for kappa - from an MGE model
        """
        T2 = T * T
        Integrand = np.zeros_like(R2)
        denom = 1. - self.e2 * T2
        for i in ilist :
            Integrand += self._pParam.qParc[i] * exp(- R2 * T2 / self._pParam.dSig3Darc2[i]) * (4. - R2 * T2 / self._dParam.Sig3Darc2_soft[i]) / sqrt(denom[i])
        return Integrand * T2
    #===========================================================
    #######################################################
    def kappa(self, R, ilist=None) :
        """
        Return :math:`\kappa`, the epicycle radial frequency for an MGE model

        :param R: cylindrical radius (float or array) in arcseconds
        :param Z: vertical height (float or array) in arcseconds

        :returns: :math:`\kappa` Radial Epicycle frequency [km.s-1.pc-1]
        :rtype: float or array of float depending on input
        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)
        ### First compute the gaussian quadrature points, and weights
        [Xquad, Wquad] = quadrat_ps_roots(self.Nquad)
        result = np.zeros_like(R)
        R2 = R*R
        for i in range(self.Nquad) :
            result += Wquad[i] * self._Intkappa(Xquad[i], R2, ilist)
        return sqrt(self.PIG * result) / self.pc_per_arcsec  # en km.s-1.pc-1
    #===========================================================
    #######################################################
    def EpicycleRatio(self, R, ilist=None) :
        """ Derive :math:`\Omega / \left(2 \\times \kappa\\right)` as the epicycle approximation
            for the ratio between sigma_R and sigma_Z

        :param R: cylindrical radius (float or array) in arcseconds

        :returns: The Epicycle ratio :math:`\Omega / 2 \\times \kappa`

        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)
        ### Then compute the two frequencies
        k = self.kappa(R, ilist)  # in km.s-1.pc-1
        O = self.Omega(R, ilist)  # in km.s-1.pc-1
        return O / (2. * k)
    #===========================================================
    #######################################################
    def QToomre(self, R, Zcut, ilist=None) :
        """ Derive the Toomre criterion

        :param R: cylindrical radius (float or array) in arcseconds
        :param Zcut: cut in Z to evaluate QToomre (in arcsecconds)

        :return : The Toomre parameter :math:`Q`

        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)

        ### Get the frequencies
        k = self.kappa(R)  # in km.s-1.pc-1
        O = self.Omega(R)  # in km.s-1.pc-1

        Z = np.zeros_like(R)
        R2 = R*R
        Z2 = Z*Z
        rhoSigmaR2 = np.zeros_like(R)
        SigmaZ = np.zeros_like(R)
        rhoT = 0.
        for i in ilist :
            self.rho, self.rhoT = self._MassDensity(R2, Z2, ilist=[i])
            rhoSigmaR2 += self._dParam.kRZ2[i] * self.rho[i] * self._sigma_z2_fromR2Z2(R2,Z2, ilist=[i]) # in km.s-1
            rhoT += self.rho[i]
        SigmaR = sqrt(rhoSigmaR2 / rhoT)
        self.rho, self.rhoT = self._MassDensity(R2, Z2, ilist)
        SigmaZ = sqrt(self._sigma_z2_fromR2Z2(R2,Z2, ilist)) # in km.s-1
        SurfDensity = self._rhointMZ(R, Zcut, ilist)
        ## self.G in (km/s)2. Msun-1 . pc2 . arcsec-1
        ## SurfDensity in Msun.pc-2
        ## So QT in pc-1 * arcsec, so we multiply by pc_per_arcsec
        QT = k * SigmaR * self.pc_per_arcsec / (3.36 * self.G * SurfDensity)
        return SigmaR, SigmaZ, O, k, QT
    #===========================================================

    #######################################################
    ### Compute some components for the Jeans modelling  ##
    #######################################################
    def _intsigma_z2(self, T, R2, Z2, ilist=None) :
        """
        Integrand for SigmaZ**2 from an MGE model
        """
        T2 = T * T
        T2Bij_soft = 1. - self._dParam.Bij_soft * T2
        q2Sig3Darc2T2 = self._dParam.q2Sig3Darc2 * T2
        Integrand = np.zeros_like(R2)        # this has the dimension of the particules array
        denom = 1. - self.e2 * T2
        expfac = self._pParam.qParc * exp(- (R2[...,np.newaxis] + Z2[...,np.newaxis] / denom) * T2 / self._dParam.dSig3Darc2_soft) / sqrt(denom)
        for j in range(self.nGauss) :
#         expfac = self._pParam.qParc[j] * exp(- (R2 + Z2 / denom[j]) * T2 / self._pParam.dSig3Darc2[j]) / sqrt(denom[j])
#         Integrand = Integrand + np.sum(self.rho * self._dParam.q2Sig3Darc2 * expfac / T2Bij_soft[:,j], axis=-1)
            Integrand += np.sum(self.rho[ilist].transpose() * (expfac[...,j])[...,np.newaxis] * q2Sig3Darc2T2[ilist] / T2Bij_soft[ilist,j], axis=-1)

        return Integrand # in rho * M arcsec pc-2 / 4 PI G
    #===========================================================
    #######################################################
    def _intvtheta2(self, T, R2, Z2, ilist=None) :
        """
        Integrand for Vtheta**2 from an MGE model
        """
        T2 = T * T
        T2Bij_soft = 1. - self._dParam.Bij_soft * T2
        Integrand = np.zeros_like(R2)
        denom = 1. - self.e2 * T2
#      T2e2j = T2 * self.e2
        qParcT2 = self._pParam.qParc * T2
        expfac = qParcT2 * exp(- (R2[...,np.newaxis] + Z2[...,np.newaxis] / denom) * T2 / self._pParam.dSig3Darc2) / sqrt(denom)
        for j in range(self.nGauss) :
            for i in ilist :
#            Integrand += self.rho[i] * (R2 * (self.e2[i] - T2e2j[j]) + self._dParam.q2Sig3Darc2[i]) * expfac[...,j] / T2Bij_soft[i,j]
                Integrand += self.rho[i] * (R2 * (self._dParam.mkRZ2q2[i] - self._dParam.Dij_soft[i,j] * T2) + self._dParam.kRZ2[i] \
                                 * self._dParam.q2Sig3Darc2[i]) * expfac[...,j] / T2Bij_soft[i,j]

        return Integrand
    #===========================================================
    #######################################################
    def _sigma_z2_fromR2Z2(self, R2, Z2, ilist=None) :
        """
        Compute SigmaZ**2 : the second centred velocity moment from an MGE model

        WARNING: this function takes R2 and Z2 as input, not R and Z

        Input : R2 and Z2: squares of the R and Z coordinates
                ilist: indices for Gaussians to take into account
        """
        r2 = R2 + Z2
        r = sqrt(r2)
        r2soft = r2 + self.SoftarcMbh2
        rsoft = sqrt(r2soft)
        ## Compute the mass density for individual gaussians as well as the sum
        [Xquad, Wquad] = quadrat_ps_roots(self.Nquad)
        sigz2 = np.sum(Wquad[i] * self._intsigma_z2(Xquad[i], R2, Z2, ilist) for i in range(self.Nquad))

        # Contribution from the BH
        if self.Mbh > 0. :
            for i in ilist:
                var = (r / self._pParam.dqSig3Darc[i]).astype(floatG)
                mask = (var < _Maximum_Value_forEXPERFC)
                lasterm = np.empty_like(var)
                # facMbh in M arcsec2 pc-2 / 4PI G
                lasterm[mask] = self._dParam.sqpi2s[i] * special.erfc(var[mask]) * np.exp(var[mask]**2)
                lasterm[~mask] = 2. / (r[~mask] + sqrt(r2[~mask] + self._dParam.qq2s2[i]))
                sigz2 += self.rho[i] * self.facMbh * (1. / rsoft - lasterm) # in rho * M arcsec pc-2 / 4 PI G

        return sigz2 * self.PIG / self.rhoT
    #===========================================================
    #######################################################
    def sigma_z2(self, R, Z, ilist=None) :
        """ Compute SigmaZ^2 : the second centred velocity moment from an MGE model

        :param R: input Radial coordinate
        :param Z: input Vertical coordinate
        :param ilist: indices for the Gaussians to take into account
        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)

        R2 = R*R
        Z2 = Z*Z
        r2 = R2 + Z2
        r = sqrt(r2)
        r2soft = r2 + self.SoftarcMbh2
        rsoft = sqrt(r2soft)
        ## Compute the mass density for individual gaussians as well as the sum
        [Xquad, Wquad] = quadrat_ps_roots(self.Nquad)
        sigz2 = np.sum(Wquad[i] * self._intsigma_z2(Xquad[i], R2, Z2, ilist) for i in range(self.Nquad))

        # Contribution from the BH
        if self.Mbh > 0. :
            for i in ilist :
                # facMbh in M arcsec2 pc-2 / 4PI G
                var = (r / self._pParam.dqSig3Darc[i]).astype(floatG)
                mask = (var < _Maximum_Value_forEXPERFC)
                lasterm = np.empty_like(var)
                # facMbh in M arcsec2 pc-2 / 4PI G
                lasterm[mask] = self._dParam.sqpi2s[i] * special.erfc(var[mask]) * np.exp(var[mask]**2)
                lasterm[~mask] = 2. / (r[~mask] + sqrt(r2[~mask] + self._dParam.qq2s2[i]))
                sigz2 += self.rho[i] * self.facMbh * (1. / rsoft - lasterm) # in rho * M arcsec pc-2 / 4 PI G

        return sigz2 * self.PIG / self.rhoT
    #===========================================================
    #######################################################
    def _vtheta2_fromR2Z2(self, R2, Z2, ilist=None) :
        """
        Compute Vtheta**2 : the first velocity moment from an MGE model

        WARNING: This function uses R2 and Z2 (squares) as input, not R and Z

        Input : R2, Z2 as input coordinates
                ilist: indices of the Gaussians
        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)

        r2 = R2 + Z2
        r = sqrt(r2)
        r2soft = r2 + self.SoftarcMbh2
        rsoft = sqrt(r2soft)
        ## Compute the mass density for individual gaussians as well as the sum
        [Xquad, Wquad] = quadrat_ps_roots(self.Nquad)
        # MU2
        VT2 = np.sum(Wquad[i] * self._intvtheta2(Xquad[i], R2, Z2, ilist) for i in range(self.Nquad))

        # Contribution from the BH
        if self.Mbh > 0. :
            for i in ilist :
                var = (r / self._pParam.dqSig3Darc[i]).astype(floatG)
                mask = (var < _Maximum_Value_forEXPERFC)
                lasterm = np.empty_like(var)
                # facMbh in M arcsec2 pc-2 / 4PI G
                lasterm[mask] = self._dParam.sqpi2s[i] * special.erfc(var[mask]) * np.exp(var[mask]**2)
                lasterm[~mask] = 2. / (r[~mask] + sqrt(r2[~mask] + self._dParam.qq2s2[i]))
                VT2 += (1. + self._dParam.e2q2Sig3Darc2[i] * R2) * self.rho[i] * self.facMbh \
                          * (1. / rsoft - lasterm)  # in rhoT * M arcsec pc-2 / 4 PI G

        return VT2 * self.PIG / self.rhoT
    #===========================================================
    #######################################################
    def vtheta2(self, R, Z, ilist=None) :
        """
        Compute Vtheta**2 : the first velocity moment from an MGE model
        Input : R, Z as input coordinates
                ilist: indices of the Gaussians
        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)

        R2 = R*R
        Z2 = Z*Z
        r2 = R2 + Z2
        r = sqrt(r2)
        r2soft = r2 + self.SoftarcMbh2
        rsoft = sqrt(r2soft)
        ## Compute the mass density for individual gaussians as well as the sum
        [Xquad, Wquad] = quadrat_ps_roots(self.Nquad)
        # MU2
        VT2 = np.sum(Wquad[i] * self._intvtheta2(Xquad[i], R2, Z2, ilist=ilist) for i in range(self.Nquad))

        # Contribution from the BH
        if self.Mbh > 0. :
            for i in ilist :
                var = (r / self._pParam.dqSig3Darc[i]).astype(floatG)
                mask = (var < _Maximum_Value_forEXPERFC)
                lasterm = np.empty_like(var)
                # facMbh in M arcsec2 pc-2 / 4PI G
                lasterm[mask] = self._dParam.sqpi2s[i] * special.erfc(var[mask]) * np.exp(var[mask]**2)
                lasterm[~mask] = 2. / (r[~mask] + sqrt(r2[~mask] + self._dParam.qq2s2[i]))
                VT2 += (1. + self._dParam.e2q2Sig3Darc2[i] * R2) * self.rho[i] * self.facMbh \
                          * (1. / rsoft - lasterm)  # in rhoT * M arcsec pc-2 / 4 PI G

        return VT2 * self.PIG / self.rhoT
    #===========================================================
    #######################################################
    def _sigmaz2_muTheta2_fromR2Z2(self, R2, Z2, ilist=None) :
        """
        Compute both Sigma_Z**2 and Mu_Z**2 the centred and non-centred
        second order velocity moments from an MGE model
        op can be "all" or "sigma" or "mu" depending on which quantity is needed

        Input : R2 and Z2 the squares of R and Z
                ilist : list of indices of Gaussians to take into account
        """
        r2 = R2 + Z2
        r = sqrt(r2)
        r2soft = r2 + self.SoftarcMbh2
        rsoft = sqrt(r2soft)
        ## Compute the mass density for individual gaussians as well as the sum
        [Xquad, Wquad] = quadrat_ps_roots(self.Nquad)
        sigz2 = np.zeros_like(R2)
        # sigmaz2
        for i in range(self.Nquad) :
            sigz2 += Wquad[i] * self._intsigma_z2(Xquad[i],  R2, Z2, ilist)

        # MU2
        muTheta2 = np.zeros_like(R2)
        for i in range(self.Nquad) :
            muTheta2 += Wquad[i] * self._intvtheta2(Xquad[i], R2, Z2, ilist)

        # Contribution from the BH
        if self.Mbh > 0. :
            for i in ilist :
                var = (r / self._pParam.dqSig3Darc[i]).astype(floatG)
                mask = (var < _Maximum_Value_forEXPERFC)
                lasterm = np.empty_like(var)
                # facMbh in M arcsec2 pc-2 / 4PI G
                lasterm[mask] = self._dParam.sqpi2s[i] * special.erfc(var[mask]) * np.exp(var[mask]**2)
                lasterm[~mask] = 2. / (r[~mask] + sqrt(r2[~mask] + self._dParam.qq2s2[i]))
                sigz2 += self.rho[i] * self.facMbh * (1. / rsoft - lasterm) # in rho * M arcsec pc-2 / 4 PI G
                muTheta2 += (1. + self._dParam.e2q2Sig3Darc2[i] * R2) * self.rho[i] * self.facMbh \
                         * (1. / rsoft - lasterm)  # in rhoT * M arcsec pc-2 / 4 PI G

        sigz2 *= self.PIG / self.rhoT
        muTheta2 *= self.PIG / self.rhoT
        return sigz2, muTheta2
    #===========================================================
    #######################################################
    def sigmaz2_muTheta2(self, R, Z, ilist) :
        """
        Compute both Sigma_Z**2 and Mu_Z**2 the centred and non-centred
        second order velocity moments from an MGE model
        op can be "all" or "sigma" or "mu" depending on which quantity is needed

        Input : R and Z the coordinates
                ilist : Gaussian indices to take into account
        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)

        R2 = R*R
        Z2 = Z*Z
        r2 = R2 + Z2
        r = sqrt(r2)
        r2soft = r2 + self.SoftarcMbh2
        rsoft = sqrt(r2soft)
        ## Compute the mass density for individual gaussians as well as the sum
        [Xquad, Wquad] = quadrat_ps_roots(self.Nquad)
        sigz2 = np.zeros_like(R2)
        # sigmaz2
        for i in range(self.Nquad) :
            sigz2 += Wquad[i] * self._intsigma_z2(Xquad[i],  R2, Z2, ilist)

        # MU2
        muTheta2 = np.zeros_like(R2)
        for i in range(self.Nquad) :
            muTheta2 += Wquad[i] * self._intvtheta2(Xquad[i], R2, Z2, ilist)

        # Contribution from the BH
        if self.Mbh > 0. :
            for i in ilist :
                var = (r / self._pParam.dqSig3Darc[i]).astype(floatG)
                mask = (var < _Maximum_Value_forEXPERFC)
                lasterm = np.empty_like(var)
                # facMbh in M arcsec2 pc-2 / 4PI G
                lasterm[mask] = self._dParam.sqpi2s[i] * special.erfc(var[mask]) * np.exp(var[mask]**2)
                lasterm[~mask] = 2. / (r [~mask]+ sqrt(r2[~mask] + self._dParam.qq2s2[i]))
                sigz2 += self.rho[i] * self.facMbh * (1. / rsoft - lasterm) # in rho * M arcsec pc-2 / 4 PI G
                muTheta2 += (1. + self._dParam.e2q2Sig3Darc2[i] * R2) * self.rho[i] * self.facMbh \
                         * (1. / rsoft - lasterm)  # in rhoT * M arcsec pc-2 / 4 PI G

        sigz2 *= self.PIG / self.rhoT
        muTheta2 *= self.PIG / self.rhoT
        return sigz2, muTheta2
    #===========================================================
