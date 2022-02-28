try:
    import numpy as np
except ImportError:
    raise Exception("numpy is required for pygme")

from numpy import asarray
from numpy import cos, sin, sqrt, arctan

try:
    from scipy import interpolate
except ImportError:
    raise Exception("scipy is required for pygme")

import os
from .rwcfor import floatMGE
from pygme.dynMGE import dynMGE
from pygme.paramMGE import dynParamMGE
from pygme.mge_miscfunctions import sample_trunc_r2gauss, sample_trunc_gauss

__version__ = '2.0.4 (24/10/2014)'  # Changed default value for SigmaGas and fixed comment in realise_Nbody
#__version__ = '2.0.3 (21/08/2013)' 
#__version__ = '2.0.2 (16/01/2013)'
# Version 2.0.3: Changed imin imax into ilist
# Version 2.0.2: 16/01/2013 - Simplification in the derivation of sigR, sigZ, sigTheta
# Version 2.0.1: 18/12/2012 - Adding the FacBetaEps factor as a parameter of the realise_Nbody routine

class nbodyMGE(dynMGE) :
    def __init__(self, infilename=None, indir=None, saveMGE=None, **kwargs) :
        dynMGE.__init__(self, infilename=infilename, indir=indir, saveMGE=saveMGE, **kwargs)

###########################################  N BODY   #############################################
    ################################################################
    ### Generate N bodies consistent with the existing MGE model ###
    ################################################################
    def realise_Nbody(self, **kwargs):
        """ Generate particles within the potential defined by the MGE model
                   Cuts in R and Z, in pc, are defined by Rcut and Zcut
                   The number of particles and the way the particles have their
                   dynamics derived is specified in the Ascii input MGE model
                   (e.g. NGROUP, NDYNCOMP, NPARTGROUP1, 2, ...)
                   Anisotropy can be specified in the input Ascii Model with
                   numbers (if negative, the Spin will be reversed), 'epicycle' or 'betaeps'
         Rcut:     cut in R, in pc - default is 50000
         Zcut:     cut in Z, in pc - default is 50000
         mcut:     cut in ellipsoidal coordinates, in pc (think of this as an ellipsoid with major-axis max radius = mcut )
                   Default is 50000
         ComputeV: Boolean (True/False), if True (default) velocities are derived, otherwise only the positions
         GasDisk: Boolean (True/False), if True (default) the Gas component will have velocities compatible with a thin disk
                  Otherwise, we will follow the prescription given by the kRZ and kRTheta components in the mge file
         SigmaGas:  SigmaR, SigmaTheta and SigmaZ for the Gas, in km/s - default to 10 km/s for all 3 values
         TruncationMethod : Method to sample the positions.
                    "Ellipsoid" (default): will follow the isosurface of each Gaussians at that radius as a cut
                                        mcut will be used (in pc)
                    "Cylindre" means an R, Z Cylindrical cal (Rcut, Zcut will be used - in pc)
         Add_BHParticle : boolean, if defined (Default is True):
                            True means that a BH particle is added if Mbh > 0
                            False means that if Mbh > 0, the potential will take it
                               into account but no particle is added
         Softening:  in pc, softening added in quadrature to the gaussian Sigmas for the potential, Default is 0 (no softening)
         FacBetaEps : factor involved when using the BETAEPS option as an anisotropy parameter for the
                      Gaussians. When one of the Gaussian component is using BETAEPS for K_R_Z, we fix the
                      anisotropy to -> delta = FacBetaEps * Epsilon where delta = 1 - Sigma_Z^2/Sigma_R^2 and
                      Epsilon is the intrinsic ellipticity of that Gaussian. Setting FacBetaEps >= 0.8 is not
                      permitted (as this would break the requirement on the second order moments).

         verbose: default is 1, will print some more information
        """
        import time

        ## Checking a Few things before starting ########################
        if self.nGauss <= 0 :
            print('ERROR: NGAUSS is not right (= %d)' %self.nGauss)
            return
        if self.TtruncMass <= 0:
            print('ERROR: Mass of the model (= %g) is not right' %self.TtruncMass)
            return
        opGAS = (self.nGasGauss != 0)
        opSTAR = (self.nStarGauss != 0)
        opHALO = (self.nHaloGauss != 0)

        ## Number of Groups -------------------------##
        if self.nGroup == 0:
            print("ERROR: nGroup is 0")
            return
        if self.nDynComp == 0:
            print("ERROR: nDynComp is 0")
            return

        ## Some options from kwargs -- INITIALISATION -------------------------------------- ##
        ##--- Compute only positions or also velocities ? ---##
        ComputeV = kwargs.get('ComputeV', True)
        GasDisk = kwargs.get('GasDisk', True)
        ## Get the dispersion for the gas in km/s -----------##
        (self.SigRGas, self.SigThetaGas, self.SigZGas) =  kwargs.get('SigmaGas',(10.0,10.0,10.0))
        ## Add a BH particle or not? --- ##
        self.Add_BHParticle = kwargs.get('Add_BHParticle', True)
        ## Overwrite mode : 'o' or None ------------------------ ##
        self.overwrite = kwargs.get('overwrite', None)
        ## First Realised Particle, and Max number of Particle -- ##
        self.FirstRealisedPart = np.int(kwargs.get('FirstRealisedPart', 0))
        self.nMaxPart = np.int(kwargs.get('nMaxPart', 0))
        ## Softening -- default is 0 (no softening)--------- ##
        self.Softening = kwargs.get('Softening', 0.0)
        ## Verbose: default is 1 ----------##
        verbose = kwargs.get('verbose', 1)
        ## -------------------------------------------------------------------------------------##

        ## Softening in pc----------------------------------##
        if self.Softening > 0. :
            print("WARNING: Softening will be %g (pc) !!!"%(self.Softening))
        self.Softarc = self.Softening / self.pc_per_arcsec   # Softening in Arcseconds
        self.SoftarcMbh = self.Softarc  # best approx for Mbh smoothing
        self.SoftarcMbh2 = self.SoftarcMbh**2

        ## -- Method for Truncating the Density distribution of particles  ---##
        self.TruncationMethod = kwargs.get('TruncationMethod', 'Ellipsoid')
        if self.TruncationMethod == "Cylindre" :
            self.Rcut = kwargs.get('Rcut', 50000)
            self.Zcut = kwargs.get('Zcut', 50000)
            Xcut = self.Rcut
            self.Rcutarc = self.Rcut / self.pc_per_arcsec
            self.Zcutarc = self.Zcut / self.pc_per_arcsec
        elif self.TruncationMethod == "Ellipsoid" :
            self.mcut = kwargs.get('mcut', 50000)
            Xcut = self.mcut
            self.mcutarc = self.mcut / self.pc_per_arcsec
        else :
            print("ERROR: TruncationMethod should be Cylindre or Ellipsoid. not %s" %(self.TruncationMethod))
            return

        ## We first save the MGE file for archival purposes, as well as the initial parameters
        self.RealisationTime = time.time()
        dest_filename = self.saveMGE + "/" + "%s_"%(str(self.RealisationTime)) + self.MGEname
        if os.path.isfile(dest_filename) & (str(self.overwrite).lower() != "o") :
            print("ERROR: filename already exists in Archival Directory %s"%(dest_filename))
            print("       Please use overwrite mode (O) or provide a different output directory (saveMGE)")
            return
        os_command = "cp %s %s"%(self.fullMGEname, dest_filename)
        os.system(os_command)
        #--------------------------------------------------------------------------------------#

        ## Save the command into a file with the same time
        text = "init_nbody(Rcut=%g, Zcut=%g, mcut=%g, ComputeV=%d, GasDisk=%s, SigRGas=%g, SigThetaGas=%g, SigZGas=%g, TruncationMethod=%s, Add_BHParticle=%r, FirstRealisedPart=%r, nMaxPart=%r, overwrite=%r)\n"%(self.Rcut, self.Zcut, self.mcut, ComputeV, GasDisk, self.SigRGas, self.SigThetaGas, self.SigZGas, self.TruncationMethod, self.Add_BHParticle, self.FirstRealisedPart, self.nMaxPart, self.overwrite)
        fout = open(self.saveMGE + "/" + "%s"%(str(self.RealisationTime)) + ".MGE_CI", "w+")
        fout.write(text)
        fout.close()
        #-------------------------------------------------#

        ## Get all parameters right and the number of particles too
        self._comp_Nparticles()

        #==============================================================================================================
        ## End of parameter initialisation
        #==============================================================================================================
        ## Beginning of allocation
        #==============================================================================================================

        self.R = np.zeros(self.nRealisedPart, floatMGE)
        self.theta = np.zeros(self.nRealisedPart, floatMGE)
        self.z = np.zeros(self.nRealisedPart, floatMGE)   ## in Parsec
        self.x = np.zeros(self.nRealisedPart, floatMGE)   ## in Parsec
        self.y = np.zeros(self.nRealisedPart, floatMGE)   ## in Parsec
        self.BodGroup = np.zeros(self.nRealisedPart, int)
        self.BodGauss = np.zeros(self.nRealisedPart, int)
        self.BodMass = np.zeros(self.nRealisedPart, floatMGE)
        ## Add the mass of the particle at 0,0,0 0,0,0 (last particle)
        if self.nRealisedPartBH == 1 :
            self.BodMass[-1] = self.Mbh

        ## Allocation for particles dynamics ############################
        self.NSpin = np.ones(self.nRealisedPart, floatMGE)
        self.NkRTheta = np.zeros(self.nRealisedPart, floatMGE)
        self.NkRZ = np.zeros(self.nRealisedPart, floatMGE)

        # Now: how do we derive sigma_R or sigma_Theta
        if self.epicycle.any() :  ## Theta will be derived from sigma_R with the epicycle approximation
            R = np.linspace(0., Xcut, 1000) ## Derive a range of R in parsec
            epiratio = self.EpicycleRatio(R / self.pc_per_arcsec) # R is passed in arcsec
            # Function to have from R in pc, sigma_R / sigma_Theta from the epicycle approximation
            funcEpiratio = interpolate.interp1d(R, epiratio)

        ## Now we implement (if betaeps=1) the relation beta = 0.6 * eps
        ## Only if specified
        if 'FacBetaEps' in kwargs :
            self.FacBetaEps = kwargs.get('FacBetaEps', 0.6)
            self._init_BetaEps(verbose=True)

        ## Derive required values from the anisotropy kRZ2 (sig_R2/ sig_z2)
        self._dParam = dynParamMGE(self)

        ############### Computing POSITIONS for the N body realisation ##################
        # for each Gaussian, derive initial positions for particles
        ## Only do this if it is axisymmetric
        if self.axi == 1 :

            ##################################### BEGIN STARS, GAS, HALO ######################################
            self.Spin = np.ones(self.nGauss, np.int)
            for i in range(self.nGauss) :
                sigma = self.Sig3D[i]

                if self.TruncationMethod == "Cylindre" :
                    self.x[self.nRealisedPartCum[i]:self.nRealisedPartCum[i+1]] = sample_trunc_gauss(sigma=sigma, cutX=self.Rcut, npoints=self.nRealisedPartGauss[i], even=1)
                    self.y[self.nRealisedPartCum[i]:self.nRealisedPartCum[i+1]] = sample_trunc_gauss(sigma=sigma, cutX=self.Rcut, npoints=self.nRealisedPartGauss[i], even=1)
                    sigma = self.Sig3D[i]*self.QxZ[i]
                    self.z[self.nRealisedPartCum[i]:self.nRealisedPartCum[i+1]] = sample_trunc_gauss(sigma=sigma, cutX=self.Zcut, npoints=self.nRealisedPartGauss[i], even=1)
                    self.theta[self.nRealisedPartCum[i]:self.nRealisedPartCum[i+1]] = asarray(np.random.uniform(0., 2.*np.pi, size=(self.nRealisedPartGauss[i],)), dtype=floatMGE)
                elif self.TruncationMethod == "Ellipsoid" :
                    r = sample_trunc_r2gauss(sigma=sigma, cutr=self.mcut, npoints=self.nRealisedPartGauss[i])
                    U = asarray(np.random.uniform(-1., 1., size=(self.nRealisedPartGauss[i],)), dtype=floatMGE)
                    V = asarray(np.random.uniform(0.,1., size=(self.nRealisedPartGauss[i],)), dtype=floatMGE)
                    sqU = np.sqrt(1. - U*U)
                    theta = 2. * np.pi * V
                    self.x[self.nRealisedPartCum[i]:self.nRealisedPartCum[i+1]] = r*sqU*cos(theta)
                    self.y[self.nRealisedPartCum[i]:self.nRealisedPartCum[i+1]] = r*sqU*sin(theta)
                    self.z[self.nRealisedPartCum[i]:self.nRealisedPartCum[i+1]] = r * U * self.QxZ[i]
                    self.theta[self.nRealisedPartCum[i]:self.nRealisedPartCum[i+1]] = theta

                self.BodGauss[self.nRealisedPartCum[i]:self.nRealisedPartCum[i+1]] = i+1
                self.BodGroup[self.nRealisedPartCum[i]:self.nRealisedPartCum[i+1]] = self.GaussDynCompNumber[i]
                self.BodMass[self.nRealisedPartCum[i]:self.nRealisedPartCum[i+1]] = self.pmassGauss[i]

                ## We set up things so that at the end we have kRZ and kRTheta
                ## First we test if one of the set up variable is negative, which means that we should inverse the Spin
                if (self.kRTheta[i] < 0) :
                    self.kRTheta[i] = np.abs(self.kRTheta[i])
                    self.Spin[i] = -1
                    self.NSpin[self.nRealisedPartCum[i]:self.nRealisedPartCum[i+1]] = - np.ones(self.nRealisedPartGauss[i], dtype=floatMGE)

                self.NkRZ[self.nRealisedPartCum[i]:self.nRealisedPartCum[i+1]] = np.zeros(self.nRealisedPartGauss[i], dtype=floatMGE) + self.kRZ[i]
                if self.epicycle[i] :
                    self.NkRTheta[self.nRealisedPartCum[i]:self.nRealisedPartCum[i+1]] = funcEpiratio(self.R[self.nRealisedPartCum[i]:self.nRealisedPartCum[i+1]])
                else :
                    self.NkRTheta[self.nRealisedPartCum[i]:self.nRealisedPartCum[i+1]] = np.zeros(self.nRealisedPartGauss[i], dtype=floatMGE) + self.kRTheta[i]

            print("NStar = %d particles Realised over a total of %d" %(self.nRealisedPartStar, self.nPartStar))
            print("NGas = %d particles Realised over a total of %d" %(self.nRealisedPartGas, self.nPartGas))
            print("NHalo = %d particles Realised over a total of %d" %(self.nRealisedPartHalo, self.nPartHalo))
            if self.nRealisedPartBH == 1:
                print("Adding a BH particle of %e Msun" %(self.Mbh))
            firstStar = 0                      # index for the first Star particle
            firstGas = lastStar = self.nRealisedPartStar          # index for the first Gas particle - last Star particle
            firstHalo = lastGas = firstGas + self.nRealisedPartGas  # index for the first Halo particle - last Gas particle
            firstBH = lastHalo = firstHalo + self.nRealisedPartHalo   # index for the BH particle - last Halo particle
            ##################################### END STARS, GAS, HALO ######################################

            ##  Computing some important quantities : R, r, theta, xarc etc ------------------------- ##
            self.R = sqrt(self.x**2 + self.y**2)
            ## And r spherical
            self.r = sqrt(self.x**2 + self.y**2+self.z**2)

            ## Now computing the true theta
            self.theta[(self.x == 0.) & (self.y >= 0.)] = np.pi / 2.
            self.theta[(self.x == 0.) & (self.y < 0.)] = -np.pi / 2.
            self.theta[(self.x < 0.)] = arctan(self.y[(self.x < 0.)] / self.x[(self.x < 0.)]) + np.pi
            self.theta[(self.x > 0.)] = arctan(self.y[(self.x > 0.)] / self.x[(self.x > 0.)])

            ### Transforming in arcsecond
            self.xarc = self.x / self.pc_per_arcsec  ### Normalisation using the distance of the galaxy
            self.yarc = self.y / self.pc_per_arcsec  ### Normalisation using the distance of the galaxy
            self.zarc = self.z / self.pc_per_arcsec  ### Normalisation using the distance of the galaxy
            self.Rarc = self.R / self.pc_per_arcsec  ### Normalisation using the distance of the galaxy
            self.rarc = self.r / self.pc_per_arcsec  ### Normalisation using the distance of the galaxy

            R2 = (self.Rarc)**2 ## R in arcsec
            Z2 = (self.zarc)**2 ## z in arcsec

            ############### Computing velocities for the N body realisation ##################
            if ComputeV :
                ###    Integration using gaussian quadrature   ###
                ### First compute the gaussian quadrature points, and weights
                print("Starting the derivation of velocities")
                self.muTheta2 = np.zeros(self.nRealisedPart, floatMGE)
                self.sigz = np.zeros(self.nRealisedPart, floatMGE)
                self.sigR = np.zeros(self.nRealisedPart, floatMGE)
                self.sigT = np.zeros(self.nRealisedPart, floatMGE)
                self.vt = np.zeros(self.nRealisedPart, floatMGE)
                if verbose :
                    print("End of memory alloc")

##### OPTION REMOVE     if self.GLOBAL_Sigma == False :
                ## Doing it in Dynamical groups #################################
                if verbose :
                    print("STARTING Local Sigma for each Dynamical Group")
                ## First check that Dynamical Groups are ordered
                setGauss_Stars = list(range(self.nStarGauss))
                setGauss_Halo = list(range(self.nStarGauss + self.nGasGauss, self.nGauss))
                setGauss = np.concatenate((setGauss_Stars, setGauss_Halo))
                nRealisedPart = self.nRealisedPartStar + self.nRealisedPartHalo
                ## First derive the equations for each INDIVIDUAL DYNAMICAL GROUP for SIGMA_Z
                if nRealisedPart != 0 :
                    for i in range(self.nDynComp) :
                        iminG = np.min(self.listGaussDynComp[i])
                        imaxG = np.max(self.listGaussDynComp[i])
                        if (iminG >= self.nStarGauss) & (imaxG < self.nStarGauss+self.nGasGauss) & GasDisk:
                            continue
                        for j in range(iminG+1, imaxG) :
                            if j not in self.listGaussDynComp[i] :
                                print("ERROR: Dynamical Group %d should included ordered Gaussians"%(i+1))
                                print("ERROR: Dynamical Group %d is "%(i+1),self.listGaussDynComp[i])
                                return

                        startI, endI  = self.nRealisedPartCum[iminG], self.nRealisedPartCum[imaxG+1]
                        if endI <= startI :
                            continue
                        R2comp = R2[startI: endI]
                        Z2comp = Z2[startI: endI]
                        self.rho, self.rhoT = self._MassDensity(R2comp, Z2comp, ilist=list(range(iminG,imaxG+1)))
                        self.rhoT = np.where(self.rhoT > 0., self.rhoT, 1.0)
                        temp1, temp2 = self._sigmaz2_muTheta2_fromR2Z2(R2comp, Z2comp, ilist=list(range(iminG,imaxG+1)))
                        self.sigz[startI: endI] = sqrt(temp1)
                        self.muTheta2[startI: endI] = temp2
                        if verbose :
                            print("End of sigz2 and mu2 derivation for Dynamical Group %02d"%(i+1))

#####   REMOVING THIS OPTION - NOT REQUIRED CONSIDERING THE INPUT ASCII FILE WITH DYN GROUPS  ###### else :
####    OPTION REMOVED     ######    if verbose :
####    OPTION REMOVED     ######       print "STARTING GLOBAL Sigma for All Stars and then Halo"
####    OPTION REMOVED     ###### ## STARS ####################
####    OPTION REMOVED     ######    R2Star = R2[firstStar:lastStar]
####    OPTION REMOVED     ######    Z2Star = Z2[firstStar:lastStar]
####    OPTION REMOVED
####    OPTION REMOVED     ######    imin = 0
####    OPTION REMOVED     ######    imax = self.nStarGauss-1     # Include all Gaussians, including Halo ones
####    OPTION REMOVED     ######    self.rho, self.rhoT = self._MassDensity(R2Star, Z2Star, imin=imin, imax=imax)
####    OPTION REMOVED
####    OPTION REMOVED     ######    ## Compute both sigmaz2 and mu2 for the Stars
####    OPTION REMOVED     ######    temp1, temp2 = self.sigmaz2_mut2(R2Star, Z2Star, imin=imin, imax=imax)
####    OPTION REMOVED     ######    self.sigz2[firstStar:lastStar] = temp1
####    OPTION REMOVED     ######    self.mut2[firstStar:lastStar] = temp2
####    OPTION REMOVED     ######    if verbose :
####    OPTION REMOVED     ######       print "End of sigz2 and mu2 derivation for Stars"
####    OPTION REMOVED
####    OPTION REMOVED     ######    ## HALO ####################
####    OPTION REMOVED     ######    R2Halo = R2[firstHalo:lastHalo]
####    OPTION REMOVED     ######    Z2Halo = Z2[firstHalo:lastHalo]
####    OPTION REMOVED
####    OPTION REMOVED     ######    imin = self.nStarGauss + self.nGasGauss
####    OPTION REMOVED     ######    imax = self.nGauss-1     # Include all Gaussians, including Halo ones
####    OPTION REMOVED     ######    self.rho, self.rhoT = self._MassDensity(R2Halo, Z2Halo, imin=imin, imax=imax)
####    OPTION REMOVED     ######    self.rhoT = np.where(self.rhoT > 0., self.rhoT, 1.0)
####    OPTION REMOVED
####    OPTION REMOVED     ######    ## Compute both sigmaz2 and mu2 for the Halos
####    OPTION REMOVED     ######    temp1, temp2 = self.sigmaz2_mut2(R2Halo, Z2Halo, imin=imin, imax=imax)
####    OPTION REMOVED     ######    self.sigz2[firstHalo:lastHalo] = temp1
####    OPTION REMOVED     ######    self.mut2[firstHalo:lastHalo] = temp2
####    OPTION REMOVED     ######    if verbose :
####    OPTION REMOVED     ######       print "End of sigz2 and mu2 derivation for Halo"

                ## Using only kRZ and kRTheta
                sigR = self.sigz * self.NkRZ
                sigTheta = np.minimum(sqrt(self.muTheta2), sigR / self.NkRTheta)  # sigma Theta from sigma R
                vt = sqrt(np.clip(self.muTheta2 - sigTheta**2, 0., np.inf))
                self.sigR[firstStar:lastStar] = sigR[firstStar:lastStar]  # sigma R from sigma Z
                self.sigR[firstHalo:lastHalo] = sigR[firstHalo:lastHalo]  # sigma R from sigma Z
                self.sigT[firstStar:lastStar] = sigTheta[firstStar:lastStar]  # sigma Theta from sigma R
                self.sigT[firstHalo:lastHalo] = sigTheta[firstHalo:lastHalo]   # sigma Theta from sigma R
                # Mean V theta
                self.vt[firstStar:lastStar] = vt[firstStar:lastStar]
                self.vt[firstHalo:lastHalo] = vt[firstHalo:lastHalo]
                if not GasDisk :
                    self.sigR[firstGas:lastGas] = sigR[firstGas:lastGas]  # sigma R from sigma Z
                    self.sigT[firstGas:lastGas] = sigTheta[firstGas:lastGas]   # sigma Theta from sigma R
                    self.vt[firstGas:lastGas] = vt[firstGas:lastGas]
                if verbose :
                    if GasDisk :
                        print("End of sigz2 and mu2 derivation for All Stars and Halo particles")
                    else :
                        print("End of sigz2 and mu2 derivation for All Stars, Gas and Halo particles")

                ## GAS ######################
                if opGAS & GasDisk:
                    self.vt[firstGas:lastGas] = self.Vcirc(self.Rarc[firstGas:lastGas])
                    self.muTheta2[firstGas:lastGas] = self.vt[firstGas:lastGas]**2 + self.SigThetaGas**2
                    temp = np.zeros_like(self.sigR[firstGas:lastGas])
                    self.sigR[firstGas:lastGas] = temp + self.SigRGas       # sigma R for the Gas
                    self.sigT[firstGas:lastGas] = temp + self.SigThetaGas   # sigma Theta for the Gas
                    self.sigz[firstGas:lastGas] = temp + self.SigZGas      # sigma Z for the Gas
                    if verbose :
                        print("End of sigz2 and mu2 derivation for Gas")

                ## Changing the spin of the component
                self.vt *= self.NSpin

                ## Starting the randomization of velocities using the derived V and Sigma values
                print("Randomizing the Velocities")
                Vescape = self.Vescape(self.Rarc,self.zarc)       # Vescape : cut it if the total velocity is higher
                Nrejected = 0
                Nstart = 0
                Nremain = self.nRealisedPart
                ind = list(range(self.nRealisedPart))
                self.Vz = np.zeros(self.nRealisedPart, floatMGE)
                self.VR = np.zeros(self.nRealisedPart, floatMGE)
                self.Vtheta = np.zeros(self.nRealisedPart, floatMGE)
                self.Vtot = np.zeros(self.nRealisedPart, floatMGE)
                iter = 0
                while Nremain != 0 :
                ### Randomize the positions taking into account the 3D width of the Gaussian
                    self.Vz[ind] = asarray(np.random.normal(0., 1., Nremain), dtype=floatMGE) * self.sigz[ind]
                    self.VR[ind] = asarray(np.random.normal(0., 1., Nremain), dtype=floatMGE) * self.sigR[ind]
                    self.Vtheta[ind] = asarray(np.random.normal(0., 1., Nremain), dtype=floatMGE) * self.sigT[ind] + self.vt[ind]

                    self.Vtot[ind] = sqrt(self.Vz[ind]**2 + self.VR[ind]**2 + self.Vtheta[ind]**2)

                    ind = np.ravel(np.where(self.Vtot[ind] > Vescape[ind]))  # indices which are NOT ok with Vesc
                    nrealised = Nremain - ind.size
                    Nstart = Nstart+nrealised
                    Nremain = ind.size
                    iter += 1
                    print("NtotalV = %d, Nrealised = %d, Nremaining = %d, Iter = %d" %(Nstart, nrealised, Nremain, iter))
                    Nrejected += Nremain

                print("Rejected (recalculated) points above Vescape: %d" %(Nrejected))

                self.Vx = self.VR * cos(self.theta) - self.Vtheta * sin(self.theta)
                self.Vy = self.VR * sin(self.theta) + self.Vtheta * cos(self.theta)

        return

############################################################################################################
####################################### END OF NBODY REALIZATION ###########################################
############################################################################################################

    def comp_Pot(self) :
        self.EcPot = self.Pot(self.Rarc, self.zarc)
        self.EcPotT = np.sum(self.EcPot)
        return

    def comp_Ep(self) :
        print("==== Potential Energy ====")
        print("WARNING: this is a direct computation of the potential energy: can be time consuming!")
        self.Ep = np.zeros(self.nRealisedPart, floatMGE)
        for i in range(self.nRealisedPart) :
            Ep = np.sum(concatenate((1./sqrt((self.x[:i] - self.x[i])**2 + (self.y[:i] - self.y[i])**2 + (self.z[:i] - self.z[i])**2), 1./sqrt((self.x[i+1:] - self.x[i])**2 + (self.y[i+1:] - self.y[i])**2 + (self.z[i+1:] - self.z[i])**2))),axis=0)
            self.Ep[i] = - Ep * self.Gorig * self.BodMass**2

        self.EpT = np.sum(self.Ep,axis=0) / 2.
        return

    def comp_Ec(self) :
        print("==== Kinetic Energy ====")
        self.Ec = 0.5 * self.BodMass * (self.Vx**2 + self.Vy**2 + self.Vz**2)
        self.EcT = np.sum(self.Ec,axis=0)
        return

    ################## Projection of the MGE model  ################
    def projpart(self, inclin=90.) :
        """ Projection of an MGE realization (N particles) using a defined inclination
        inclin: inclination in degrees, 90 being edge-on, 0 being face-on
        """

        inclin_rad = inclin * np.pi / 180.
        self.Xp = self.x
        self.Yp = self.y * cos(inclin_rad) + self.z * sin(inclin_rad)
        self.Zp = - self.y * sin(inclin_rad) + self.z * cos(inclin_rad)
        self.Xparc = self.Xp / self.pc_per_arcsec
        self.Yparc = self.Yp / self.pc_per_arcsec
        self.Zparc = self.Zp / self.pc_per_arcsec

        self.Vrad = self.Vy * sin(inclin_rad) - self.Vz * cos(inclin_rad)

        return
    #===================================================================

    ##################################################################
    ### Save the Nbody coordinates x,y,z,Vx,Vy,Vz in an ascii file   #
    ##################################################################
    def save_nbody(self, outdir=None, outfilename=None, overwrite=False, arcsec=False) :
        """ Save the N body realizationof an MGE model into an ascii file
          name : string defining the name of the output file
          overwrite: if file exists, overwrite or not - default = False
          arcsec: save the positions in arcseconds or pc - default= False (pc)
        """
        if outfilename is None :
            print("You must specify an output ascii file")
            return

        if outdir is not None :
            outfilename = outdir + outfilename

        if os.path.isfile(outfilename) and overwrite==False :  # testing the existence of the file
            print('WRITING ERROR: File %s already exists, use overwrite=True if you wish' %outfilename)
            return

        ascii_file = open(outfilename, mode="w")

        if arcsec == True :
            outx = self.xarc
            outy = self.yarc
            outz = self.zarc
        else :
            outx = self.x
            outy = self.y
            outz = self.z

        for i in range(self.nRealisedPart) :
            line = "%12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e \n" %(outx[i], outy[i], outz[i], self.Vx[i], self.Vy[i], self.Vz[i], self.BodMass[i])
            ascii_file.write(line)

        ascii_file.close
        return
    #===================================================================
