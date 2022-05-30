#!/usr/bin/python
"""
This module reads and writes the parameters of a Multi Gaussian Expansion model (Monnet et al.
1992, Emsellem et al. 1994). It can read and write MGE input ascii files and
computes a number of basic parameters for the corresponding models.  

uptdated regularly and may still contains some obvious bugs. A stable version will
be available hopefully before the end of 2012.
For questions, please contact Eric Emsellem at eric.emsellem@eso.org
"""

"""
Importing the most import modules
This MGE module requires NUMPY and SCIPY
"""

import os

try:
    import numpy as np
except ImportError:
    raise Exception("numpy is required for pygme")

try:
    from scipy import special
except ImportError:
    raise Exception("scipy is required for pygme")

from numpy import asarray
from numpy import cos, sin, copy, sqrt, exp

from .rwcfor import floatMGE
from .mge_miscfunctions import print_msg

__version__ = '1.1.6 (22 Dec 2014)'

## Version 1.1.6 : EE - Fixed found2D
## Version 1.1.5 : EE - Fixed mcut input parameter
## Version 1.1.4 : EE - Fixed a typo on indices
## Version 1.1.3 : EE - Added BetaEps, M/L etc also in the 2D Gauss just in case
## Version 1.1.2 : EE - Changed imin,imax into ilist
## Version 1.1.1 : EE - Removed the condition for comp_Nparticles when reading an mge
## Version 1.1.0 : EE - Some serious cleanup in the naming of the variables
## Version 1.0.2 : EE - few minor changes including adding saveMGE
## Version 1.0.1 : EE - replaces ones to zeros in initialisation of GaussGroupNumber

############################################################################
# Class to define dynamical MGE parameters useful for calculation purposes #
############################################################################
class dynParamMGE():
    """ 
    Class to add some parameters which are useful for dynamical routines
    """
    def __init__(self, MGEmodel):
        """ 
        Initialisation of the additional dynamical parameters
        """
        if (MGEmodel._findGauss3D > 0):
            self.Sig3Darc2_soft = MGEmodel.Sig3Darc**2  + MGEmodel.Softarc**2  # Sigma softened in arcsec
            self.dSig3Darc2_soft = 2. * self.Sig3Darc2_soft
            # Deriving some more numbers
            self.Bij = np.zeros((MGEmodel.nGauss, MGEmodel.nGauss), floatMGE)
            self.Bij_soft = np.zeros((MGEmodel.nGauss, MGEmodel.nGauss), floatMGE)
            self.e2q2dSig3Darc2 = np.zeros(MGEmodel.nGauss, floatMGE)
            self.e2q2Sig3Darc2 = np.zeros(MGEmodel.nGauss, floatMGE)
            self.sqpi2s = sqrt(np.pi / 2.) / MGEmodel.qSig3Darc
            self.qq2s2 = 4. * MGEmodel.QxZ2 * MGEmodel.Sig3Darc2
            self.q2Sig3Darc2 = MGEmodel.QxZ2 * MGEmodel.Sig3Darc2
            for i in range(MGEmodel.nGauss) :
                if self.q2Sig3Darc2[i] != 0. :
                    self.e2q2dSig3Darc2[i] = MGEmodel.e2[i] / (2. * self.q2Sig3Darc2[i])
                    self.e2q2Sig3Darc2[i] = MGEmodel.e2[i] / self.q2Sig3Darc2[i]
                else :
                    print("WARNING: %d component has q2*Sig2=0" %(i+1))
                for j in range(MGEmodel.nGauss) :
                    self.Bij[i,j] = MGEmodel.e2[j] - self.q2Sig3Darc2[i] / MGEmodel.Sig3Darc2[j]
                    self.Bij_soft[i,j] = MGEmodel.e2[j] - self.q2Sig3Darc2[i] / self.Sig3Darc2_soft[j]

            self.kRZ2 = MGEmodel.kRZ**2
            self.mkRZ2q2 = 1. - self.kRZ2 * MGEmodel.QxZ2
            self.mkRZ2 = 1. - self.kRZ2
            self.Dij = np.zeros((MGEmodel.nGauss,MGEmodel.nGauss), floatMGE)
            self.Dij_soft = np.zeros((MGEmodel.nGauss,MGEmodel.nGauss), floatMGE)
            for i in range(MGEmodel.nGauss) :
                for j in range(MGEmodel.nGauss) :
                    self.Dij[i,j] = self.mkRZ2[i] * self.Bij[i,j] + MGEmodel.e2[j] * self.kRZ2[i]
                    self.Dij_soft[i,j] = self.mkRZ2[i] * self.Bij_soft[i,j] + MGEmodel.e2[j] * self.kRZ2[i]

## ===========================================================================================

############################################################################
# Class to define photometric MGE parameters useful for calculation purposes #
############################################################################
class photParamMGE():
    """ 
    Class to add some parameters which are useful for photometric routines
    """
    def __init__(self, MGEmodel):
        """ 
        Initialisation of the additional photometric parameters
            These are hidden in this class
        """
        if (MGEmodel._findGauss3D > 0):
            self.dSig3Darc = sqrt(2.) * MGEmodel.Sig3Darc
            self.dSig3Darc2 = 2. * MGEmodel.Sig3Darc2
            self.qParc = MGEmodel.QxZ * MGEmodel.Parc
            self.dqSig3Darc = sqrt(2.) * MGEmodel.qSig3Darc

## ===========================================================================================

class paramMGE(object) :
    def __init__(self, infilename=None, saveMGE=None, indir=None, **kwargs) :
        """
        Initialisation of the MGE model - reading the input file

        infilename : input MGE ascii file defining the MGE model
        indir: directory where to find the mge file
        saveMGE: directory in which some MGE model will be saved automatically during the
                 realisation of the Nbody sample
                 If saveMGE is None (default), it will be defined as ~/MGE
                 This will be created by default (if not existing)
                 
        Additional Input (not required):
            nTotalPart: total number of particles
            nPartStar : number of Stellar particles
            nPartHalo: number of Dark Matter particles
            nPartGas : number of Gas particles

            FirstRealisedPart : number for the first realised Particle 
                                This is useful if we wish to realise the model in chunks
            nMaxPart : Max number of particles to be realised for this run

            mcut : cut in pc, Default is 50 000 (50 kpc)
                   Used for the Ellipsoid truncation

            Rcut : cut in pc, Default is 50 000 (50 kpc)
            Zcut : cut in pc, Default is 50 000 (50 kpc)
                   Used for the Cylindre truncation
 
            FacBetaEps : Coefficient for : Beta = Coef * Epsilon
                         Default if Coef = 0.6
                         Can also be a vector (one for each Gaussian)

            MaxFacBetaEps: maximum value allowed for FacBetaEps. Default is 0.8.

        """

        ## Now checking if saveMGE has been defined and act accordingly
        if saveMGE is None :
            ## This is the default dir (~/MGE) if none is given
            saveMGE = os.path.expanduser("~/MGE")
            if not os.path.isdir(saveMGE) :
                ## Creating the default saveMGE directory
                os.system("mkdir ~/MGE")

        ## Test now if this exists
        if not os.path.isdir(saveMGE) :
            print("ERROR: directory for Archival does not exist = %s"%(saveMGE))
            return
        ## Finally save the value of saveMGE in the structure
        self.saveMGE = saveMGE

        ## Setting up some fixed variable #####################################
        ## G is in (km/s)2. Msun-1 . pc .
        ## OLD VALUE WAS:  self.Gorig = 0.0043225821
        self.Gorig = floatMGE(0.0043225524) # value from Remco van den Bosch

        self.nPart = np.int(kwargs.get("nTotalPart", 0))   # TOTAL Number of n bodies
        self.nPartStar = np.int(kwargs.get("nPartStar", 0))   # TOTAL Number of n bodies
        self.nPartHalo = np.int(kwargs.get("nPartHalo", 0))   # TOTAL Number of n bodies
        self.nPartGas = np.int(kwargs.get("nPartGas", 0))   # TOTAL Number of n bodies
        self.Add_BHParticle = True   # Add a BH if Mbh > 0 when realising particles

        self.FirstRealisedPart = np.int(kwargs.get("FirstRealisedPart", 0))   # First Realised Particle
        self.nMaxPart = np.int(kwargs.get("nMaxPart", 0))   # Max number of particles to be realised

        self.Euler = np.array([0., 90., 0.]) # Inclination - Default is 90 degrees = edge-on
        self.TruncationMethod = "Ellipsoid"   # Default method to truncate Gaussians (other = Cylindre)
        self.mcut = kwargs.get("Mcut", 50000.)   # Default truncation in pc - Default is 50kpc
        self.Rcut = kwargs.get("Rcut", 50000.)   # Default truncation in pc - Default is 50kpc
        self.Zcut = kwargs.get("Zcut", 50000.)   # Default truncation in pc - Default is 50kpc

        self.Mbh = 0.           # Black hole mass
        self.axi = 1

        self.Nquad = 100              # Number of Points for the Quadrature, default is 100
        self._findGauss3D = 0
        self._findGauss2D = 0

        self.FacBetaEps = kwargs.get("FacBetaEps", 0.6)   # Coefficient for the BETAEPS option: Beta = Coef * Epsilon
        self.MaxFacBetaEps = kwargs.get("MaxFacBetaEps", 0.8)   # Max value the BETAEPS Factor
        self.DummyFacBetaEps = 0.6

        ## Test if infilename is None. If this is the case reset MGE with 0 Gaussians
        self.nGauss = self.nGroup = self.nDynComp = 0
        self._reset(All=True)
        if infilename is not None :
            self.read_mge(infilename, indir=indir)

    def _reset(self, **kwargs) :
        """
        Reset values of the MGE model

        Possible options:
           nGauss
           nGroup
           NDynComp
           Dist
           Softening
           infilename
           pwd

           All : will set all to None, or 0 (and Dist to 10 Mpc)
        """
        AllReset = kwargs.get("All", False)
        if AllReset :
            for key in ["infilename", "pwd"] :
                kwargs[key] = ""
            for key in ["nGauss", "nGroup", "nDynComp"] :
                kwargs[key] = 0
            self._reset_Dist()
            self._reset_Softening()
            kwargs["Dist"] = self.Dist
            kwargs["Softening"] = self.Softening

        for key in kwargs :
            if key == "nGauss" :
                nGauss = kwargs.get("nGauss", None)
                self._reset_nGauss(nGauss)  # Set nGauss 
            elif key == "nGroup" :
                nGroup = kwargs.get("nGroup", None)
                self._reset_nGroup(nGroup)  # Set nGroup
            elif key == "Dist" :
                Dist = kwargs.get("Dist", None)
                self._reset_Dist(Dist)     # Distance in Mpc - Default is 10 Mpc
            elif key == "Softening" :
                Softening = kwargs.get("Softening", None)
                self._reset_Softening(Softening)  # Set Softening 
            elif key == "nDynComp" :
                self.nDynComp = kwargs.get("nDynComp", None)
            elif key == "infilename" :
                self.infilename = kwargs.get("infilename", None)
            elif key == "pwd" :
                self.pwd = kwargs.get("pwd", None)

    def _reset_nGroup(self, nGroup=None) :
        ## nGroup Reset
        if nGroup is not None :
            self.nGroup = nGroup    # Number of Groups
            self.nPartGroup = np.zeros((self.nGroup,), np.int)    # Number of particles per Group
            self.nRealisedPartGroup = np.zeros((self.nGroup,), np.int)    # Number of REALISED particles per Group

    ## =============================================================
    def _reset_nGauss(self, nGauss=0, verbose=0) :
        ## nGauss reset
        if nGauss is not None :
            if np.size(nGauss) == 3 :
               self.nStarGauss = int(nGauss[0])
               self.nGasGauss = int(nGauss[1])
               self.nHaloGauss = int(nGauss[2])
               self.nGauss = self.nStarGauss + self.nGasGauss + self.nHaloGauss 
            elif np.size(nGauss) == 1 :
               self.nGauss = nGauss    # Number of Gaussians
               self.nStarGauss = nGauss
               self.nGasGauss = self.nHaloGauss = 0
            else :
               print_msg("With nGauss which should contain 1 or 3 integers", 2)
               return
            self._findGauss3D = 0
            self._findGauss2D = 0
            self.Imax2D = np.zeros((self.nGauss,), floatMGE)   # In Lsun  pc-2
            self.Sig2Darc = np.zeros((self.nGauss,), floatMGE)  # in arcsecond
            self.Q2D = np.zeros((self.nGauss,), floatMGE)
            self.PAp = np.zeros((self.nGauss,), floatMGE)
            self.Imax3D = np.zeros((self.nGauss,), floatMGE) # In Lsun pc-2 arcsec-1
            self.Sig3Darc = np.zeros((self.nGauss,), floatMGE)  # in arcsecond
            self.QxZ = np.zeros((self.nGauss,), floatMGE)
            self.QyZ = np.zeros((self.nGauss,), floatMGE)
            self.ML = np.ones((self.nGauss,), floatMGE)
            self.kRTheta = np.ones((self.nGauss,), floatMGE)   # sigma_R / sigma_Theta
            self.kRZ = np.ones((self.nGauss,), floatMGE)       # sigma_R / sigma_Z
            self.betaeps = np.zeros((self.nGauss,), np.int)   # betaeps option (1 or 0)
            self.epicycle = np.zeros((self.nGauss,), np.int)   # epicycle option (1 or 0)
            self.truncFlux = np.zeros((self.nGauss,), floatMGE)
            self.MGEFlux = np.zeros((self.nGauss,), floatMGE)
            self.truncMass = np.zeros((self.nGauss,), floatMGE)
            self.MGEMass = np.zeros((self.nGauss,), floatMGE)
            self.MGEFluxp = np.zeros((self.nGauss,), floatMGE)
            self.GaussGroupNumber = np.ones((self.nGauss,), np.int)     # Group Number for that Gaussian
            self.GaussDynCompNumber = np.ones((self.nGauss,), np.int)     # Dynamical Group Number for that Gaussian
            self.TtruncMass = 0.         # Total mass  in Nbody
            self.TtruncFlux = 0.         # Total flux  in Nbody
            self.TMGEMass = 0.         # Total mass of MGE model
            self.TMGEFlux = 0.         # Total flux of MGE model
            self.axi = 1

    ## Change the Distance of the model ###########################
    def _reset_Dist(self, Dist=None, verbose=True) :
        if Dist is None :
            if hasattr(self, "Dist"):
                Dist = self.Dist
            else:
                Dist = 10.0 ## Setting the default in case the Distance is negative
                print("WARNING: dummy Dist value for reset")

        if Dist <= 0. :
            if verbose:
                print("WARNING: you provided a negative Dist value")
                print("WARNING: it will be set to the default (10 Mpc)")
            Dist = 10.0 ## Setting the default in case the Distance is negative

        self.Dist = floatMGE(Dist)
        self.pc_per_arcsec = floatMGE(np.pi * self.Dist / 0.648)
        self.mcutarc = self.mcut / self.pc_per_arcsec   #Default truncation - in arcseconds at 10 Mpc
        self.Rcutarc = self.Rcut / self.pc_per_arcsec   #Default truncation - in arcseconds at 10 Mpc
        self.Zcutarc = self.Zcut / self.pc_per_arcsec   #Default truncation - in arcseconds at 10 Mpc
        ## G is in (km/s)2. Msun-1 . pc .
        ## We multiply it by pc / arcsec
        ## so it becomes:
        ##  (km/s)2. Msun-1 . pc2 . arcsec-1
        ## OLD VALUE WAS:  self.Gorig = 0.0043225821
        self.G = self.Gorig * self.pc_per_arcsec
        self.PIG = floatMGE(4. * np.pi * self.G)
        ## Adding the standard parameters
        self._add_PhotometricParam()
    ## =============================================================

    ## Change the softening of the model ###########################
    def _reset_Softening(self, Softening=0.0, verbose=0) :
        """
        Change the softening value of the model (in pc)
        """
        if Softening is not None :
            self.Softening = Softening              # softening  in pc
            self.Softarc = self.Softening / self.pc_per_arcsec    # Softening in arcsec
            self.SoftarcMbh = self.Softarc
            self.SoftarcMbh2 = self.SoftarcMbh**2
            ## Add dynamics parameters: this is needed since the softening just changed
            self._dParam = dynParamMGE(self)
    ## ============================================================

    ## List the Gaussians in the different Groups #################
    def _listGroups(self) :
        # Reinitialise the list of Gaussians in the Groups
        self.listGaussGroup = []
        for i in range(self.nGroup) :
            self.listGaussGroup.append(np.where(self.GaussGroupNumber == (i+1))[0])
    ## ============================================================

    ## List the Gaussians in the different Dynamics Groups #################
    def _listDynComps(self) :
        # Reinitialise the list of Gaussians in the Groups
        self.listGaussDynComp = []
        for i in range(self.nDynComp) :
            self.listGaussDynComp.append(np.where(self.GaussDynCompNumber == (i+1))[0])
    ## ============================================================


    ## Decode the SGAUSS and associated lines in mge File #############
    def _read_GAUSS2D(self, linesplit, findGauss2D) :
        self.Imax2D[findGauss2D] = floatMGE(linesplit[1])                # I in Lum.pc-2
        self.Sig2Darc[findGauss2D] = floatMGE(linesplit[2])              # Sigma in arcsec
        self.Q2D[findGauss2D] = floatMGE(linesplit[3])
        self.PAp[findGauss2D] = floatMGE(linesplit[4])
        lelines = len(linesplit)
        if lelines >= 6 :
            self.ML[findGauss2D] = floatMGE(linesplit[5])
        if lelines >= 7 :
            if linesplit[6][:3] == "EPI" :
                self.kRTheta[findGauss2D] = -1.0
                self.epicycle[findGauss2D] = 1
            else :
                self.kRTheta[findGauss2D] = floatMGE(linesplit[6])
                self.epicycle[findGauss2D] = 0
            if linesplit[7][:4] == "BETA" :
                self.betaeps[findGauss2D] = 1
            else :
                self.kRZ[findGauss2D] = floatMGE(linesplit[7])
                self.betaeps[findGauss2D] = 0
        if lelines >= 9 :
            self.GaussGroupNumber[findGauss2D] = int(linesplit[8])
        if lelines >= 10 :
            self.GaussDynCompNumber[findGauss2D] = int(linesplit[9])
        return

    ## Decode the SGAUSS and associated lines in mge File #############
    def _read_GAUSS3D(self, linesplit, findGauss3D) :
        self.Imax3D[findGauss3D] = floatMGE(linesplit[1])            # I in Lum.pc-2.arcsec-1
        self.Sig3Darc[findGauss3D] = floatMGE(linesplit[2])             # Sigma in arcsec
        self.QxZ[findGauss3D] = floatMGE(linesplit[3])
        self.QyZ[findGauss3D] = floatMGE(linesplit[4])
        self.ML[findGauss3D] = floatMGE(linesplit[5])
        lelines = len(linesplit)
        if lelines >= 8 :
            if linesplit[6][:3] == "EPI" :
                self.kRTheta[findGauss3D] = -1.0
                self.epicycle[findGauss3D] = 1
            else :
                self.kRTheta[findGauss3D] = floatMGE(linesplit[6])
                self.epicycle[findGauss3D] = 0
            if linesplit[7][:4] == "BETA" :
                self.kRZ[findGauss3D] = 1. / sqrt(1. - (self.FacBetaEps[findGauss3D] * (1. - self.QxZ[findGauss3D])))
                self.betaeps[findGauss3D] = 1
            else :
                self.kRZ[findGauss3D] = floatMGE(linesplit[7])
                self.betaeps[findGauss3D] = 0
        if lelines >= 9 :
            self.GaussGroupNumber[findGauss3D] = int(linesplit[8])
        if lelines >= 10 :
            self.GaussDynCompNumber[findGauss3D] = int(linesplit[9])
        if (self.QxZ[findGauss3D] != self.QyZ[findGauss3D]) :
            self.axi = 0
            print('Detected triaxial component %d: self.axi set to 0'%(findGauss3D))
        return
     ## ============================================================
    def _init_BetaEps(self, verbose=True) :
        """
        We initialise here the BetaEps vector using the input value
        If a scalar, it is transformed into a vector of constant values.
        It will only be used for components that have the betaeps option =1.
        """
        if np.size(self.FacBetaEps) == 1 :
            self.FacBetaEps = np.array([self.FacBetaEps] * self.nGauss)
        elif np.size(self.FacBetaEps) != self.nGauss :
            print("WARNING: FacBetaEps has a dimension which is not consistent with the number of Gaussians")
            print("WARNING: Should be a scalar or a 1D array of size nGauss")
            print("WARNING: We will therefore use the fixed default value = 0.6 instead.")
            self.FacBetaEps = np.array([0.6] * self.nGauss)
 
        self.FacBetaEps = np.asarray(self.FacBetaEps)
        ## Checking that no value goes beyond MaxFacBetaEps
        if np.any(self.FacBetaEps > self.MaxFacBetaEps) : 
            print("WARNING: FacBetaEps cannot be set to values higher than %5.3f"%(self.MaxFacBetaEps))
            print("WARNING: Input FacBetaEps = ", self.FacBetaEps)
            print("WARNING: We will change these values to 0.6.")
            self.FacBetaEps = np.where(self.FacBetaEps > self.MaxFacBetaEps, self.MaxFacBetaEps, self.FacBetaEps)
 
        if verbose: 
            print("The BetaEps vector (beta = FacBetaEps * Epsilon) is fixed to  ")
            print("                   ", self.FacBetaEps)
 
        if self.betaeps.any() :
            self.kRZ[self.betaeps == 1] = np.zeros(np.sum(self.betaeps, dtype=np.int), floatMGE) + 1. / sqrt(1. - (self.FacBetaEps[self.betaeps == 1] * (1. - self.QxZ[self.betaeps == 1])))

    ##################################################################
    ### Reading an ascii MGE file and filling the MGE class object ###
    ##################################################################
    def read_mge(self, infilename=None, indir=None) :

        if (infilename is not None) :                       # testing if the name was set
            if indir is not None :
                infilename = indir + infilename

            if not os.path.isfile(infilename) :          # testing the existence of the file
                print('OPENING ERROR: File %s not found' %infilename)
                return

            ################################
            # Opening the ascii input file #
            ################################
            self.pwd = os.getcwd()
            self.fullMGEname = os.path.abspath(infilename)
            self.MGEname = os.path.basename(self.fullMGEname)
            self.pathMGEname = os.path.dirname(self.fullMGEname)

            mge_file = open(self.fullMGEname)

            lines = mge_file.readlines()
            nlines = len(lines)

            ########################################
            ## First get the Number of gaussians  ##
            ## And the global set of parameters   ##
            ########################################
            keynGauss = keynStarGauss = keynGasGauss = keynHaloGauss = keynGroup = 0
            findGauss2D = findGauss3D = findStarGauss2D = findStarGauss3D = findGasGauss2D = findGasGauss3D = findHaloGauss2D = findHaloGauss3D = findGroup = 0

            for i in range(nlines) :
                if lines[i][0] == "#" or lines[i] == "\n" :
                    continue
                sl = lines[i].split()
                keyword = sl[0]
                if (keyword[:6] == "NGAUSS") :
                    if len(sl) == 2 :
                       nStarGauss = int(sl[1])
                       nGasGauss = nHaloGauss = 0
                    elif len(sl) == 4 :
                       nStarGauss = int(sl[1])
                       nGasGauss = int(sl[2])
                       nHaloGauss = int(sl[3])
                    self.nStarGauss = nStarGauss
                    self.nGasGauss = nGasGauss
                    self.nHaloGauss = nHaloGauss
                    keynStarGauss = 1
                    keynGasGauss = 1
                    keynHaloGauss = 1
                    if nStarGauss < 0 or nGasGauss < 0 or nHaloGauss < 0:
                        print('ERROR: Keyword NGAUSS has some negative values: %d %d %d' %(nStarGauss, nGasGauss, nHaloGauss))
                        continue
                    nGauss = nStarGauss + nGasGauss + nHaloGauss
                    if nGauss <= 0 :
                        print('ERROR: Keyword NGAUSS is less than or equal to 0: %d' %nGauss)
                        continue
                    self._reset(nGauss=(nStarGauss, nGasGauss, nHaloGauss))
                    keynGauss = 1
                elif (keyword[:4] == "DIST") :
                    Dist = floatMGE(sl[1])
                    self._reset_Dist(Dist)
                elif (keyword[:6] == "NGROUP") :
                    nGroup = int(sl[1])
                    if nGroup < 0 :
                        print('ERROR: Keyword NGROUP is less than 0: %d' %nGroup)
                        continue
                    self._reset(nGroup=nGroup)
                    keynGroup = 1
                elif (keyword[:9] == "NDYNCOMP") :
                    nDynComp = int(sl[1])
                    if nDynComp < 0 :
                        print('ERROR: Keyword NDYNCOMP is less than 0: %d' %nDynComp)
                        continue
                    self._reset(nDynComp=nDynComp)

            if (keynGauss == 0) :
                print('Could not find NGAUSS keyword in the MGE input File %s' %self.MGEname)
                return
            listStarGauss2D = []
            listStarGauss3D = []
            listGasGauss2D = []
            listGasGauss3D = []
            listHaloGauss2D = []
            listHaloGauss3D = []

            ## We initialise the BetaEps Values using the input one
            self._init_BetaEps()

            ##================================================================================##
            ## Then really decoding the lines and getting all the details from the ascii file ##
            ##================================================================================##
            for i in range(nlines) :
                if (lines[i][0] == "#")  or (lines[i] == "\n") :
                    continue
                sl = lines[i].split()
                keyword = sl[0]
                if (keyword[:6] == "NGAUSS") or (keyword[:4] == "DIST") or (keyword[:9] == "NGASGAUSS") or (keyword[:10] == "NHALOGAUSS") or (keyword[:11] == "NGROUP") or (keyword[:11] == "NDYNCOMP"):
                    continue
                ## projected gaussians
                elif (keyword[:11] == "STARGAUSS2D") :
                    if findGauss2D == self.nGauss  or keynStarGauss == 0 :
                        print('Line ignored (STARS: NGAUSS = %d): %s' %(self.nGauss,lines[i]))
                        continue
                    if findStarGauss2D == self.nStarGauss :
                        print('Line ignored (STAR: NSTARGAUSS = %d): %s' %(self.nStarGauss,lines[i]))
                        continue
                    self._read_GAUSS2D(sl, findGauss2D)
                    listStarGauss2D.append(findGauss2D)
                    findGauss2D += 1
                    findStarGauss2D += 1
                elif (keyword[:10] == "GASGAUSS2D") :
                    if findGauss2D == self.nGauss or keynGasGauss == 0:
                        print('Line ignored (GAS: NGAUSS = %d): %s' %(self.nGauss,lines[i]))
                        continue
                    if findGasGauss2D == self.nGasGauss :
                        print('Line ignored (GAS: NGASGAUSS = %d): %s' %(self.nGasGauss,lines[i]))
                        continue
                    self._read_GAUSS2D(sl, findGauss2D)
                    listGasGauss2D.append(findGauss2D)
                    findGauss2D += 1
                    findGasGauss2D += 1
                elif (keyword[:11] == "HALOGAUSS2D") :
                    if findGauss2D == self.nGauss or keynHaloGauss == 0:
                        print('Line ignored (HALO: NGAUSS = %d): %s' %(self.nGauss,lines[i]))
                        continue
                    if findHaloGauss2D == self.nHaloGauss :
                        print('Line ignored (HALO: NHALOGAUSS = %d): %s' %(self.nHaloGauss,lines[i]))
                        continue
                    self._read_GAUSS2D(sl, findGauss2D)
                    listHaloGauss2D.append(findGauss2D)
                    findGauss2D += 1
                    findHaloGauss2D += 1

                ## spatial gaussians
                elif (keyword[:11] == "STARGAUSS3D") :
                    if findGauss3D == self.nGauss :
                        print('Line ignored (NGAUSS = %d): %s' %(self.nGauss,lines[i]))
                        continue
                    if findStarGauss3D == self.nStarGauss :
                        print('Line ignored (STAR: NSTARGAUSS = %d): %s' %(self.nStarGauss,lines[i]))
                        continue
                    self._read_GAUSS3D(sl, findGauss3D)
                    listStarGauss3D.append(findGauss3D)
                    findGauss3D += 1
                    findStarGauss3D += 1
                elif (keyword[:10] == "GASGAUSS3D") :
                    if findGauss3D == self.nGauss or keynGasGauss == 0:
                        print('Line ignored (GAS: NGAUSS = %d): %s' %(self.nGauss,lines[i]))
                        continue
                    if findGasGauss3D == self.nGasGauss :
                        print('Line ignored (GAS: NGASGAUSS = %d): %s' %(self.nGasGauss,lines[i]))
                        continue
                    self._read_GAUSS3D(sl, findGauss3D)
                    listGasGauss3D.append(findGauss3D)
                    findGauss3D += 1
                    findGasGauss3D += 1
                elif (keyword[:11] == "HALOGAUSS3D") :
                    if findGauss3D == self.nGauss or keynHaloGauss == 0:
                        print('Line ignored (HALO: NGAUSS = %d): %s' %(self.nGauss,lines[i]))
                        continue
                    if findHaloGauss3D == self.nHaloGauss :
                        print('Line ignored (HALO: NHALOGAUSS = %d): %s' %(self.nHaloGauss,lines[i]))
                        continue
                    self._read_GAUSS3D(sl, findGauss3D)
                    listHaloGauss3D.append(findGauss3D)
                    findGauss3D += 1
                    findHaloGauss3D += 1

                ## Center and other parameters
                elif (keyword[:6] == "CENTER") :
                    self.Center = np.zeros((2,), floatMGE)
                    self.Center[0] = floatMGE(sl[1])
                    self.Center[1] = floatMGE(sl[2])
                elif (keyword[:5] == "EULER") :
                    self.Euler = np.zeros((3,), floatMGE)
                    self.Euler[0] = floatMGE(sl[1])
                    self.Euler[1] = floatMGE(sl[2])
                    self.Euler[2] = floatMGE(sl[3])
                elif (keyword[:3] == "MBH") :
                    self.Mbh = floatMGE(sl[1])
                elif (keyword[:10] == "NPARTGROUP") :
                    GroupNumber = int(keyword[10:])
                    if GroupNumber > self.nGroup or GroupNumber < 0 or findGroup == self.nGroup or keynGroup == 0 or (len(sl) > 3) or (int(sl[1]) < 0) :
                        print('Line ignored (NPARTGROUP%2d: NGROUP = %d) = Wrong Entry %s' %(GroupNumber, self.nGroup, lines[i]))
                        continue
                    if len(sl) == 3 :
                        if (int(sl[2]) < 0) or (int(sl[2]) > int(sl[1])) :
                            print('Line ignored (NPARTGROUP: NGROUP = %d) = second entry should be greater than 0 and less than the first entry: %s' %(self.nGroup,lines[i]))
                            continue
                        self.nRealisedPartGroup[GroupNumber - 1] = int(sl[2])             # Number of particles in Group to be realised

                    self.nPartGroup[GroupNumber - 1] = int(sl[1])             # Number of particles in Group

                    findGroup += 1
                else :
                    print('Could not decode the following keyword: %s' %keyword)
                    mge_file.close
                    break
            ################################
            # CLOSING the ascii input file #
            ################################
            mge_file.close

            ##============ Ascii file is not closed ====================##


            ## Reorganising the read parameters and data ##
            ## And setting this up into the structure ##
            self._findGauss2D = findGauss2D
            self._findGauss3D = findGauss3D
            self.nGauss = max(findGauss3D, findGauss2D)
            self.nGasGauss = max(findGasGauss3D, findGasGauss2D)
            self.nHaloGauss = max(findHaloGauss3D, findHaloGauss2D)
            self.nStarGauss = max(findStarGauss3D, findStarGauss2D)

            ## Reorganizing things to have the gas then halo components at the end
            ## ORDER OF GAUSSIANS IS THEREFORE: STARS, GAS, HALO
            tempImax2D = copy(self.Imax2D)
            tempSig2Darc = copy(self.Sig2Darc)
            tempQ2D = copy(self.Q2D)
            tempPAp = copy(self.PAp)
            tempImax3D = copy(self.Imax3D)
            tempSig3Darc = copy(self.Sig3Darc)
            tempQxZ = copy(self.QxZ)
            tempQyZ = copy(self.QyZ)
            tempML = copy(self.ML)
            tempkRTheta = copy(self.kRTheta)
            tempkRZ = copy(self.kRZ)
            tempbetaeps = copy(self.betaeps)
            tempepicycle = copy(self.epicycle)
            tempGaussGroup = copy(self.GaussGroupNumber)
            tempGaussDynComp = copy(self.GaussDynCompNumber)
            ## Projected components
            k = 0
            j = findGauss2D - self.nHaloGauss - self.nGasGauss
            l = findGauss2D - self.nHaloGauss
            for i in range(findGauss2D) :
                if i not in listGasGauss2D :
                    if i not in listHaloGauss2D :
                        ind = k
                        k += 1
                    else :
                        ind = l
                        l += 1
                else :
                    ind = j
                    j += 1
                self.Imax2D[ind] = tempImax2D[i]  # I in Lum.pc-2
                self.Sig2Darc[ind] = tempSig2Darc[i]
                self.Q2D[ind] = tempQ2D[i]
                self.PAp[ind] = tempPAp[i]
            ## Spatial components
            k = 0
            j = findGauss3D - self.nHaloGauss - self.nGasGauss
            l = findGauss3D - self.nHaloGauss
            self.listGasGauss = listGasGauss3D
            self.listHaloGauss = listHaloGauss3D
            self.listStarGauss = listStarGauss3D
            for i in range(findGauss3D) :
                if i not in listGasGauss3D :
                    if i not in listHaloGauss3D :
                        ind = k
                        k += 1
                    else :
                        ind = l
                        l += 1
                else :
                    ind = j
                    j += 1
                self.Imax3D[ind] = tempImax3D[i]
                self.Sig3Darc[ind] = tempSig3Darc[i]
                self.QxZ[ind] = tempQxZ[i]
                self.QyZ[ind] = tempQyZ[i]
                self.ML[ind] = tempML[i]
                self.kRTheta[ind] = tempkRTheta[i]
                self.kRZ[ind] = tempkRZ[i]
                self.betaeps[ind] = tempbetaeps[i]
                self.epicycle[ind] = tempepicycle[i]
                self.GaussGroupNumber[ind] = tempGaussGroup[i]
                self.GaussDynCompNumber[ind] = tempGaussDynComp[i]
            #########################################

            # Testing if all axis ratios are axisymmetric or not
            self.axi = 1
            for i in range(findGauss3D) :
                if (self.QxZ[i] != self.QyZ[i]) :
                    self.axi = 0
                    print('Detected triaxial component: self.axi set to 0')
            ## Add all sorts of parameters which are useful for further derivation
            self._comp_Nparticles()

            ## Set default inclination to 90 degrees
            if 'Euler' in self.__dict__ :
                inclination = self.Euler[1]
            else :
                self.Euler = np.zeros((3,), floatMGE)
                self.Euler[1] = 90.0

            if self._findGauss3D == 0 & self._findGauss2D > 0 :
                self.deproject(inclin=self.Euler[1], verbose=False)
            if self._findGauss3D > 0 :
                if self._findGauss2D == 0 :
                    self.project(inclin=self.Euler[1], verbose=False)
                else :
                    print_msg("Both 3D and 2D Gaussians were found: ", 1)
                    print_msg("We thus used the 2D Gaussians as a prior for the deprojection at %5.2f degrees"%(self.Euler[1]), 1)
                    self.deproject(inclin=self.Euler[1], verbose=True)

            print("Found %d Spatial and %d projected Gaussians" %(self._findGauss3D, self._findGauss2D))
            print("With an Inclination of %5.2f (degrees)"%(self.Euler[1]))
            if self.nStarGauss != 0 :
                print("This includes %d STAR Gaussians" %(np.maximum(findStarGauss3D, findStarGauss2D)))
            if self.nGasGauss != 0 :
                print("This includes %d GAS Gaussians" %(np.maximum(findGasGauss3D, findGasGauss2D)))
            if self.nHaloGauss != 0 :
                print("This also includes %d HALO Gaussians" %(np.maximum(findHaloGauss3D,findHaloGauss2D)))
            print("Found %d Particle Groups" %(findGroup))
            print("Found %d Dynamical Components (each may include a set of Gaussians)" %(nDynComp))
            print("Distance set up to %6.2f Mpc"%(self.Dist))

        # no name was specified #
        else :
            print('You should specify an output file name')

    #====================== END OF READING / INIT THE MGE INPUT FILE =======================#
    ### INTEGRATED LUMINOSITY - ALL -------------------------------------------------
    ### Deriving the integrated Lum (Rcut, Zcut) for 1 gaussian, R and Z are in arcsec
    def rhointL_1G(self, Rcut, Zcut, ind) :
        """
        Integrated LUMINOSITY truncated within a cylindre defined by Rcut, Zcut (in arcsec)
           for 1 Gaussian only: ind is the indice of that gaussian
        """
        if self._findGauss3D == 0 : 
            print_msg("No Spatial Gaussians yet", 2)
            return 0.
        return self.MGEFlux[ind] * (1. - exp(- Rcut*Rcut/self._pParam.dSig3Darc2[ind])) * float(special.erf(Zcut/self._pParam.dqSig3Darc[ind]))

    ### Deriving the integrated Mass (Rcut, Zcut) for 1 gaussian, R and are in arcsec
    def rhointM_1G(self, Rcut, Zcut, ind) :
        """
        Integrated Mass truncated within a cylindre defined by Rcut, Zcut (in arcsec)
           for 1 Gaussian only: ind is the indice of that gaussian
        """
        if self._findGauss3D == 0 : 
            print_msg("No Spatial Gaussians yet", 2)
            return 0.
        return self.MGEMass[ind] * (1. - exp(- Rcut * Rcut / self._pParam.dSig3Darc2[ind])) \
            * float(special.erf(Zcut / self._pParam.dqSig3Darc[ind]))

    ### INTEGRATED MASS - SPHERE ALL --------------------------------------------------------
    ### Deriving the integrated Mass (mcut) for 1 gaussian, m in arcsec
    def rhoSphereintM_1G(self, mcut, ind) :
        """
        Integrated Mass truncated within a spheroid of m=mcut (in arcsec)
           for 1 Gaussian only: ind is the indice of that gaussian
        """
        if self._findGauss3D == 0 : 
            print_msg("No Spatial Gaussians yet", 2)
            return 0.
        return self.MGEMass[ind] * (float(special.erf(mcut / self._pParam.dSig3Darc[ind])) - mcut * np.sqrt(2. / np.pi) \
            * exp(- mcut*mcut/self._pParam.dSig3Darc2[ind])/ self.Sig3Darc[ind])

    ### Deriving the integrated Lum (mcut) for 1 gaussian, m in arcsec
    ################### A REVOIR
    def rhoSphereintL_1G(self, mcut, ind) :
        """
        Integrated LUMINOSITY truncated within a spheroid of m=mcut (in arcsec)
           for 1 Gaussian only: ind is the indice of that gaussian
        """
        if self._findGauss3D == 0 : 
            print_msg("No Spatial Gaussians yet", 2)
            return 0.
        return self.MGEFlux[ind] * (float(special.erf(mcut / self._pParam.dSig3Darc[ind])) - mcut * np.sqrt(2. / np.pi) \
             * exp(- mcut*mcut/self._pParam.dSig3Darc2[ind])/ self.Sig3Darc[ind])

    #####################################
    ## Adding more Gaussian parameters ##
    #####################################
    def _add_PhotometricParam(self) :
        """
        Add many more parameters using the basic I, Sig, q, PA parameters of the model
        These parameters are important for many (photometry/dynamics-related) routines
        """

        ## Only if axisymmetric
        if self.axi :

            ##################################################################
            ## Compute some useful parameters for the projected Gaussians
            ##################################################################
            if  (self._findGauss2D > 0) :
                # some useful numbers from the projected gaussians if they exist
                self.Sig2D =  self.Sig2Darc * self.pc_per_arcsec        # Sigma in pc
                self.Q2D2 = self.Q2D * self.Q2D
                self.Sig2Darc2 = self.Sig2Darc * self.Sig2Darc  # Projected Sigma in arcsecond
                self.dSig2Darc2 = 2. * self.Sig2Darc2
                self.Pp = self.Imax2D * self.ML      # Mass maximum in Mass/pc-2
                self.MGEFluxp = self.Imax2D*(self.Sig2D**2) * self.Q2D2 * np.pi

            ##################################################################
            ## Compute some useful parameters for the Spatial Gaussians
            ##################################################################
            if (self._findGauss3D > 0):
                # some more useful numbers
                self.Imax3Dpc3 = self.Imax3D / self.pc_per_arcsec  # I in Lum.pc-3
                self.Sig3D =  self.Sig3Darc * self.pc_per_arcsec # Sigma in pc
                self.Parc = self.Imax3D * self.ML  # Mass maximum in Mass/pc-2/arcsec-1
                self.QxZ2 = self.QxZ ** 2
                self.e2 = 1. - self.QxZ2
                self.Sig3Darc2 = self.Sig3Darc**2           # Sigma in arcsecond !
                self.qSig3Darc = self.QxZ * self.Sig3Darc

                ## Add photometric parameters
                self._pParam = photParamMGE(self)
                ## Add dynamics parameters
                self._dParam = dynParamMGE(self)

                ## Fluxes and Masses
                self.MGEFlux = self.Imax3Dpc3 * self.QxZ * (sqrt(2.*np.pi) * self.Sig3D)**3
                self.MGEMass = self.MGEFlux * self.ML

                ## Total Mass and Flux for Stars and Gas and Halo (not truncated)
                self.MGEStarMass = np.sum(self.MGEMass[:self.nStarGauss],axis=0)
                self.MGEStarFlux = np.sum(self.MGEFlux[:self.nStarGauss],axis=0)
                self.MGEGasMass = np.sum(self.MGEMass[self.nStarGauss:self.nStarGauss+self.nGasGauss],axis=0)
                self.MGEGasFlux = np.sum(self.MGEFlux[self.nStarGauss:self.nStarGauss+self.nGasGauss],axis=0)
                self.MGEHaloMass = np.sum(self.MGEMass[self.nStarGauss+self.nGasGauss:self.nStarGauss+self.nGasGauss+self.nHaloGauss],axis=0)
                self.MGEHaloFlux = np.sum(self.MGEFlux[self.nStarGauss+self.nGasGauss:self.nStarGauss+self.nGasGauss+self.nHaloGauss],axis=0)
                ## Total Mass and Flux for all
                self.TMGEFlux = np.sum(self.MGEFlux,axis=0)
                self.TMGEMass = np.sum(self.MGEMass,axis=0)

                self.facMbh = self.Mbh / (4. * np.pi * self.pc_per_arcsec * self.pc_per_arcsec)  # in M*pc-2*arcsec2

                ## TRUNCATED Mass and Flux for each Gaussian
                self.truncMass = np.zeros(self.nGauss, floatMGE)
                self.truncFlux = np.zeros(self.nGauss, floatMGE)
                if self.TruncationMethod == "Cylindre" :
                    for i in range(self.nGauss) :
                        self.truncFlux[i] = self.rhointL_1G(self.Rcutarc, self.Zcutarc, i)
                        self.truncMass[i] = self.rhointM_1G(self.Rcutarc, self.Zcutarc, i)
                elif  self.TruncationMethod == "Ellipsoid" :
                    for i in range(self.nGauss) :
                        self.truncFlux[i] = self.rhoSphereintL_1G(self.mcutarc, i)
                        self.truncMass[i] = self.rhoSphereintM_1G(self.mcutarc, i)
                ## Total TRUNCATED Flux and Mass
                self.TtruncFlux = np.sum(self.truncFlux,axis=0)
                self.TtruncMass = np.sum(self.truncMass,axis=0)

                # Listing the Gaussians in the Groups
                self._listGroups()
                self._listDynComps()

                ## Total Mass and Flux for Groups TRUNCATED!
                self.truncGroupMass = np.zeros(self.nGroup, floatMGE)
                self.truncGroupFlux = np.zeros(self.nGroup, floatMGE)
                for i in range(self.nGroup) :
                    self.truncGroupMass[i] = np.sum(self.truncMass[self.listGaussGroup[i]], axis=0)
                    self.truncGroupFlux[i] = np.sum(self.truncFlux[self.listGaussGroup[i]], axis=0)
                ## Total TRUNCATED Flux and Mass for STARS, GAS, HALO
                ## STARS
                self.truncStarFlux = np.sum(self.truncFlux[0: self.nStarGauss])
                self.truncStarMass = np.sum(self.truncMass[0: self.nStarGauss])
                ## GAS
                self.truncGasFlux = np.sum(self.truncFlux[self.nStarGauss:self.nStarGauss + self.nGasGauss])
                self.truncGasMass = np.sum(self.truncMass[self.nStarGauss:self.nStarGauss + self.nGasGauss])
                ## HALO
                self.truncHaloFlux = np.sum(self.truncFlux[self.nStarGauss + self.nGasGauss:self.nStarGauss + self.nGasGauss + self.nHaloGauss])
                self.truncHaloMass = np.sum(self.truncMass[self.nStarGauss + self.nGasGauss:self.nStarGauss + self.nGasGauss + self.nHaloGauss])

        else :
            print_msg("Triaxial model, cannot compute additional photometric parameters", 1)

    ## ===========================================================================================================

    ###################################################
    ### Set the list of Indices for Gaussians        ##
    ###################################################
    def _set_ilist(self, ilist=None) :
        if ilist is None : return list(range(self.nGauss))
        else : return ilist

    ###################################################
    ### Compute the fraction for each component      ##
    ##  for a list of indices                        ##
    ###################################################
    def _fraclistNbody(self, nbody, ilist) :
        """
        Compute the fraction of particles for each component
        corresponding to the truncated (Gaussian) Mass
        """
        ### Set the list of indices
        ilist = self._set_ilist(ilist)
        nind = len(ilist)

        fracNpGauss = np.zeros(nind, np.int32)
        totaln = np.zeros(nind+1, np.int32)
        TMass = np.sum(self.truncMass[ilist], axis=0)
        for i in range(nind) :
            fracNpGauss[i] = np.int(self.truncMass[ilist[i]] * nbody / TMass)
            totaln[i+1] = totaln[i] + fracNpGauss[i]
        fracNpGauss[nind-1] = nbody - totaln[nind-1]
        totaln[nind] = nbody

        return fracNpGauss, totaln
    ## ==================================================

    ###############################################################################################################
    ## To compute the number of particles and particle masses for each Gaussian/Groups ############################
    ###############################################################################################################
    def _comp_Nparticles(self) :
        """
        Add the respective numbers of particles for each Gaussian, Group
        Depending on the Mass of each component
        pmassGroup, pmassGauss: mass of the particles for each Gaussian, Group
        nPartGauss : number of particles for each Gaussian
        """
        self._add_PhotometricParam()
        if (self.axi == 1) & (self._findGauss3D > 0):
            # For this we use the list of Gaussians in the Groups
            # First step is to have the Mass of Each Group to get the particle mass
            mask = (self.nPartGroup !=0)
            self.pmassGroup = np.zeros_like(self.truncGroupMass)
            self.pmassGroup[mask] = self.truncGroupMass[mask] / self.nPartGroup[mask]  # Mass of the particles in Groups
            self.pmassGauss = self.pmassGroup[self.GaussGroupNumber - 1]   # Mass of the particles in Gaussians
            self.nPartGauss = np.zeros(self.nGauss, dtype=int)
            self.nRealisedPartGauss = np.zeros(self.nGauss, dtype=int)
            for i in range(self.nGroup) :
                fracNpGauss, totaln = self._fraclistNbody(self.nPartGroup[i], self.listGaussGroup[i])
                fracRealNpGauss, totaln = self._fraclistNbody(self.nRealisedPartGroup[i], self.listGaussGroup[i])
                self.nPartGauss[self.listGaussGroup[i]] = fracNpGauss  # TOTAL Number of particles in that Gaussian
                self.nRealisedPartGauss[self.listGaussGroup[i]] = fracRealNpGauss  # TOTAL Number of particles to be Realised in that Gaussian

            ## Cumulative sum for the total number of particles in the Model
            self.nPartCum = np.concatenate((np.array([0]),asarray(np.cumsum(self.nPartGauss),dtype=int)))

            ## Now we calculate the number of particles to be realised in each Gaussian taking into account the MaxPart
            ##
            ## Temporary sum for the following calculation
            self.nRealisedPartCum = np.concatenate((np.array([0]),asarray(np.cumsum(self.nRealisedPartGauss),dtype=int)))

            ## If we limit the number of particles, we use nMaxPart and FirstRealisedPart as guidelines
            if self.nMaxPart > 0 :
                firstPart = self.FirstRealisedPart  ## This is the first particle to be realised
                lastPart = firstPart + np.minimum(self.nMaxPart, np.sum(self.nRealisedPartGroup, axis=0) - firstPart) ## last particle to be realised
                imin = 0   # Counter
                for i in range(self.nGauss) :
                    n1 = np.maximum(imin, firstPart)
                    n2 = np.minimum(imin + self.nRealisedPartGauss[i], lastPart)
                    imin += self.nRealisedPartGauss[i]
                    self.nRealisedPartGauss[i] = np.maximum(0,n2 - n1)

            ## Derive the cumulative sum now
            self.nRealisedPartCum = np.concatenate((np.array([0]),asarray(np.cumsum(self.nRealisedPartGauss),dtype=int)))

            ## Allocation for particles positions ############################
            if self.Add_BHParticle & (self.Mbh > 0) :
                self.nRealisedPartBH = 1
            else :
                self.nRealisedPartBH = 0
            self.nPartStar = np.sum(self.nPartGauss[:self.nStarGauss], dtype=np.int)
            self.nPartGas = np.sum(self.nPartGauss[self.nStarGauss:self.nStarGauss+self.nGasGauss], dtype=np.int)
            self.nPartHalo = np.sum(self.nPartGauss[self.nStarGauss+self.nGasGauss:], dtype=np.int)
            self.nPart = self.nPartStar + self.nPartGas + self.nPartHalo 
            if self.Mbh > 0 :
                self.nPart += 1

            self.nRealisedPartStar = np.sum(self.nRealisedPartGauss[:self.nStarGauss], dtype=np.int)
            self.nRealisedPartGas = np.sum(self.nRealisedPartGauss[self.nStarGauss:self.nStarGauss+self.nGasGauss], dtype=np.int)
            self.nRealisedPartHalo = np.sum(self.nRealisedPartGauss[self.nStarGauss+self.nGasGauss:], dtype=np.int)
            self.nRealisedPart = self.nRealisedPartStar + self.nRealisedPartGas + self.nRealisedPartHalo + self.nRealisedPartBH
    ## =============================================================

    ################################################################
    ### Deprojection of the MGE model for an axiymmetric galaxy  ###
    ################################################################
    def deproject(self, inclin=None, printformat="E", particles=True, verbose=True) :
        """
        Deproject the Gaussians and provide the spatial parameters
        
        inclin: inclination in degrees
        printformat: "E" or "F" depending if you want Engineering or Float notation
                     default is "E"
        """
        if self.axi != 1 :
            print("ERROR: cannot deproject this model: not axisymmetric !\n")
            return

        if inclin is None : inclin = self.Euler[1]

        self.Euler = np.array([0., inclin, 0.])
        if inclin == 0. :
            print("Not yet supported\n")
            return
            for i in range(self.nGauss) :
                if self.Q2D[i] != 1 :
                    print("ERROR: cannot deproject this model as component %d does not have Q2D = 1!\n" %(i+1))
        elif inclin == 90. :
            if verbose :
                print("Edge-on deprojection\n")
            self.Sig3Darc = self.Sig2Darc
            self.QxZ = self.Q2D * 1.0
            self.QyZ = self.Q2D * 1.0
            self.Imax3D = self.Imax2D / (sqrt(2. * np.pi) * self.Sig2Darc)
            self._findGauss3D = self.QxZ.shape[0]
        else :
            inclin_rad = inclin * np.pi / 180.
            cosi2 = cos(inclin_rad) * cos(inclin_rad)
            sini2 = sin(inclin_rad) * sin(inclin_rad)
            for i in range(self.nGauss) :
                if cosi2 > (self.Q2D[i] * self.Q2D[i]) :
                    maxangle = np.arccos(self.Q2D[i])
                    print("ERROR: cannot deproject the component %d. Max angle is %f" %(i+1, maxangle*180./np.pi))
                    continue
                self.QxZ[i] = sqrt((self.Q2D[i] * self.Q2D[i] - cosi2) / sini2)
                self.QyZ[i] = self.QxZ[i] * 1.0
                self.Sig3Darc[i] = self.Sig2Darc[i] * 1.0
                self.Imax3D[i] = self.Imax2D[i] *  self.Q2D[i] / (sqrt(2. * np.pi) * self.QxZ[i] * self.Sig2Darc[i])
            self._findGauss3D = self.QxZ.shape[0]

        if verbose :
            print("Deprojected Model with inclination of %5.2f" %(inclin))
            print("      #       Imax              Sigma       Qx        Qy")        
            print("           Lsun/pc^2/arcsec     arcsec")
            if printformat == "F" : ff = "%13.5f"
            else : ff = "%13.8e"
            for i in range(self.nGauss) :
                print(("3D-G %2d    {0}        %10.5f %9.5f %9.5f" %(i+1, self.Sig3Darc[i], self.QxZ[i], self.QyZ[i])).format(ff%(self.Imax3D[i])))

        if particles :
            if 'kRZ' not in self.__dict__ :
                self.kRZ = np.ones(self.nGauss, floatMGE)
            self._init_BetaEps(verbose=False)
            self._comp_Nparticles()
        return
    ## ===========================================================================================

    ################################################################
    ### Projection of the MGE model for an axiymmetric galaxy  ###
    ################################################################
    def project(self, inclin=90, printformat="E", particles=True, verbose=True) :
        """
        Project the Gaussians and provide the 2D parameters
        
        inclin: inclination in degrees
        printformat: "E" or "F" depending if you want Engineering or Float notation
                      default is "E"
        """
        if self.axi != 1 :
            print("ERROR: cannot project this model: not axisymmetric !\n")
            return

        self.Euler = np.array([0., inclin, 0.])
        if inclin == 0. :
            if verbose :
                print("Face-on Projection\n")
            self.Sig2Darc = self.Sig3Darc
            self.Q2D = np.ones(self.nGauss, floatMGE)
            self.Imax2D = self.Imax3D * sqrt(2. * np.pi) * self.QxZ * self.Sig3Darc
        elif inclin == 90. :
            if verbose :
                print("Edge-on Projection\n")
            self.Sig2Darc = self.Sig3Darc * 1.0
            self.Q2D = self.QxZ * 1.0
            self.Imax2D = self.Imax3D * (sqrt(2. * np.pi) * self.Sig3Darc)
        else :
            inclin_rad = inclin * np.pi / 180.
            cosi2 = cos(inclin_rad) * cos(inclin_rad)
            sini2 = sin(inclin_rad) * sin(inclin_rad)
            for i in range(self.nGauss) :
                self.Q2D[i] = sqrt(self.QxZ[i] * self.QxZ[i] * sini2 + cosi2)
                self.Sig2Darc[i] = self.Sig3Darc[i] * 1.0
                self.Imax2D[i] = self.Imax3D[i] * sqrt(2. * np.pi) * self.QxZ[i] * self.Sig3Darc[i] / self.Q2D[i]

        self._findGauss2D = self.Q2D.shape[0]
        if verbose :
            print("Projected Model with inclination of %5.2f" %(inclin))
            print("      #       Imax       Sigma        Q2D")        
            print("            Lsun/pc^2   arcsec")
            if printformat == "F" : ff = "%13.5f"
            else : ff = "%13.8e"
            for i in range(self.nGauss) :
                print(("2D-G %2d {0} %9.5f %9.5f"%(i+1, self.Sig2Darc[i], self.Q2D[i])).format(ff%(self.Imax2D[i])))
        if particles :
            self._comp_Nparticles()
        return
    #===================================================================

    ##################################################################
    ### Write an ascii MGE file using an existing MGE class object ###
    ##################################################################
    def write_mge(self, outdir=None, outfilename=None, overwrite=False) :
        if (outfilename is None) :                       # testing if the name was set
            print('You should specify an output file name')
            return

        if outdir is not None :
            outfilename = outdir + outfilename

        ## Testing if the file exists
        if os.path.isfile(outfilename) :
            if not overwrite : # testing if the existing file should be overwritten
                print('WRITING ERROR: File %s already exists, use overwrite=True if you wish' %outfilename)
                return

        mgeout = open(outfilename, "w+")
        ## Starting to write the output file
        linecomment = "#######################################################\n"

        def set_txtcomment(text, name, value, valform="%f") :
            textout = "## %s \n"%(text)
            return textout + name + " " + valform%(value)+"\n"

        mgeout.write(linecomment + "## %s MGE model \n"%(outfilename) + linecomment)

        ## Basic Parameters
        mgeout.write(set_txtcomment("Distance [Mpc]", "DIST", self.Dist, "%5.2f"))
        mgeout.write(set_txtcomment("Black Hole Mass [Msun]", "MBH", self.Mbh, "%8.4e"))
        mgeout.write(set_txtcomment("Euler Angles [Degrees]", "EULER", tuple(self.Euler), "%8.5f %8.5f %8.5f"))
        mgeout.write(set_txtcomment("Center [Arcsec]", "CENTER", tuple(self.Center), "%8.5f %8.5f"))

        ## Number of Gaussians
        NGauss = (self.nStarGauss, self.nGasGauss, self.nHaloGauss)
        mgeout.write(set_txtcomment("Number of Gaussians (Stars, Gas, Dark Matter)", "NGAUSS", NGauss, "%d %d %d"))
        Gaussians3D = np.zeros((self.nGauss, 9), float)
        Gaussians2D = np.zeros((self.nGauss, 4), float)
        if self._findGauss3D  > 0:
            ## Projecting to get the 2D values
            self.project(inclin=self.Euler[1], particles=False)
        elif self._findGauss2D  > 0:
            ## Deprojecting to get the 3D values
            self.deproject(inclin=self.Euler[1], particles=False)
        else :
            print_msg("No Gaussians found in this model", 3)
        ## Deprojecting to get the 3D values
        Gaussians2D[:,0] = self.Imax2D
        Gaussians2D[:,1] = self.Sig2Darc
        Gaussians2D[:,2] = self.Q2D
        Gaussians2D[:,3] = self.PAp
        Gaussians3D[:,0] = self.Imax3D
        Gaussians3D[:,1] = self.Sig3Darc
        Gaussians3D[:,2] = self.QxZ 
        Gaussians3D[:,3] = self.QyZ 
        Gaussians3D[:,4] = self.ML
        Gaussians3D[:,5] = self.kRTheta
        Gaussians3D[:,6] = self.kRZ 
        Gaussians3D[:,7] = np.asarray(self.GaussGroupNumber, float)
        Gaussians3D[:,8] = np.asarray(self.GaussDynCompNumber, float) 
        self.axi = 1

        ###################
        ## 2D Gaussians
        ###################
        ## STARS First
        k = 0
        mgeout.write("## No                  Imax   Sigma      Q      PA\n")
        mgeout.write("## Stellar 2D Gaussians\n")
        for i in range(NGauss[0]) :
            mgeout.write("STARGAUSS2D%02d   "%(i+1) + "%8.5e %8.5f %8.5f %8.5f \n"%tuple(Gaussians2D[k]))
            k += 1
        ## then Gas
        mgeout.write("## Gas 2D Gaussians\n")
        for i in range(NGauss[1]) :
            mgeout.write("GASGAUSS2D%02d   "%(i+1) + "%8.5e %8.5f %8.5f %8.5f \n"%tuple(Gaussians2D[k]))
            k += 1
        ## Then Dark Matter
        mgeout.write("## Dark Matter 2D Gaussians\n")
        for i in range(NGauss[2]) :
            mgeout.write("HALOGAUSS2D%02d   "%(i+1) + "%8.5e %8.5f %8.5f %8.5f \n"%tuple(Gaussians2D[k]))
            k += 1
        ###################
        ## 3D Gaussians
        ###################
        ## STARS First
        k = 0
        mgeout.write("## ID                  Imax    Sigma       QxZ       QyZ      M/L     kRT     kRZ   Group DynComp\n")
        mgeout.write("## Stellar 3D Gaussians\n")
        for i in range(NGauss[0]) :
            if self.betaeps[k]:
                mgeout.write("STARGAUSS3D%02d   "%(i+1) + "%8.5e %8.5f %8.5f %8.5f %8.5f %8.5f"%tuple(Gaussians3d[k][:6]) \
                        + " BETAEPS " + "%d %d \n"%tuple(Gaussians3D[k][7:]))
            else:
                mgeout.write("STARGAUSS3D%02d   "%(i+1) + "%8.5e %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %d %d \n"%tuple(Gaussians3D[k]))
            k += 1
        ## then Gas
        mgeout.write("## Gas 3D Gaussians\n")
        for i in range(NGauss[1]) :
            if self.betaeps[k]:
                mgeout.write("GASGAUSS3D%02d   "%(i+1) + "%8.5e %8.5f %8.5f %8.5f %8.5f %8.5f"%tuple(Gaussians3d[k][:6]) \
                        + " BETAEPS " + "%d %d \n"%tuple(Gaussians3D[k][7:]))
            else:
                mgeout.write("GASGAUSS3D%02d   "%(i+1) + "%8.5e %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %d %d \n"%tuple(Gaussians3D[k]))
            k += 1
        ## Then Dark Matter
        mgeout.write("## Dark Matter 3D Gaussians\n")
        for i in range(NGauss[2]) :
            if self.betaeps[k]:
                mgeout.write("HALOGAUSS3D%02d   "%(i+1) + "%8.5e %8.5f %8.5f %8.5f %8.5f %8.5f"%tuple(Gaussians3d[k][:6]) \
                        + " BETAEPS " + "%d %d \n"%tuple(Gaussians3D[k][7:]))
            else:
                mgeout.write("HALOGAUSS3D%02d   "%(i+1) + "%8.5e %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %d %d \n"%tuple(Gaussians3D[k]))
            k += 1

        ## Number of Groups et al.
        mgeout.write(set_txtcomment("Number of Groups", "NGROUP", self.nGroup, "%d"))
        mgeout.write(set_txtcomment("Number of Dynamical Components", "NDYNCOMP", self.nDynComp, "%d"))
        mgeout.write("## PARTICLES for each DynComp: Total number and Number to be realised\n")
        for i in range(self.nGroup) :
             NPartGroup = (self.nPartGroup[i], self.nRealisedPartGroup[i])
             mgeout.write("NPARTGROUP%02d %d %d\n"%(i+1, self.nPartGroup[i], self.nRealisedPartGroup[i]))

        mgeout.close()
#===================================================================================================================================

def create_mge(outfilename=None, overwrite=False, outdir=None, **kwargs) :
    """Create an MGE ascii file corresponding to the input parameters
    """
    ## Setting a temporary MGE object
    saveMGE = kwargs.get('saveMGE', None)
    if saveMGE is None :
        tempMGE = paramMGE()
    else :
        tempMGE = paramMGE(saveMGE=saveMGE)

    ## Test if the structure was properly initialised
    if not hasattr(tempMGE, "mcut") :
        ## If not just return and stop as the message was already clear
        ## From the initialisation
        return

    ## Get the numbers from kwargs
    ## First the Gaussians
    NGauss = np.asarray(kwargs.get('NGauss', np.array([1,0,0])), int)
    if NGauss.size == 1 : NGauss = np.asarray(np.array([NGauss, 0, 0]), int)
    TNGauss = NGauss.sum()

    ## Inclination
    if "Inclination" in kwargs :
        if "Euler" in kwargs :
            print_msg("Both Euler and Inclination are defined here: will use Euler as a default", 1)
        else :
            kwargs["Euler"] =  np.array([0., float(kwargs.get("Inclination")), 0.])
    tempMGE.Euler = np.asarray(kwargs.get("Euler"))
    if tempMGE.Euler.size != 3 :
        print_msg("Problem with Euler angles, will set the default = 0, 90, 0 = edge-on", 1)
        tempMGE.Euler = np.array([0., 90., 0.])

    tempMGE._reset(nGauss=NGauss)

    temp2D = np.array([0., 1., 1., 0.])
    temp3D = np.array([0., 1., 1., 1., 1., 1., 1., 1, 1])
    temp3D_short = np.array([1., 1., 1., 1, 1])

    # Testing for betaeps
    if 'betaeps' in kwargs:
        betaeps = kwargs.pop('betaeps', np.ones(tempMGE.nGauss, dtype=int))
        if size(betaeps) == 1:
            betaeps = [betaeps] * tempMGE.nGauss
        elif size(betaeps) != tempMGE.nGauss:
            print_msg("Provided value(s) for betaeps has the wrong shape")
            print_msg("Setting betaeps to 0 (False) for all Gaussians")
            betaeps = np.zeros(tempMGE.nGauss, dtype=int)
        tempMGE.betaeps = np.asarray(betaeps, dtype=int)
    else:
        tempMGE.betaeps = np.zeros(tempMGE.nGauss, dtype=int)

    found2D = found3D = 0
    if 'Gauss3D' in kwargs :
        Gaussians3D = np.asarray(kwargs.get('Gauss3D'))
        if Gaussians3D.size == 9 : 
            Gaussians3D = np.tile(Gaussians3D, tempMGE.nGauss)
        elif Gaussians3D.size == 4 * tempMGE.nGauss :
            Gaussians3D = np.append(Gaussians3D.reshape(tempMGE.nGauss, 4), np.tile(temp3D_short, tempMGE.nGauss).reshape(tempMGE.nGauss, 5), 1)
        if Gaussians3D.size == 9 * tempMGE.nGauss :
            Gaussians3D = Gaussians3D.reshape(tempMGE.nGauss, 9) 
        else :
            print_msg("The provided 3D Gaussians have the wrong shape", 1)
            print_msg("We will set up a DUMMY set of 3D Gaussians", 1)
            Gaussians3D = np.tile(temp3D, tempMGE.nGauss).reshape(tempMGE.nGauss, 9) 

        found3D = 1
        if 'Gauss2D' in kwargs :
            print_msg("We will only use the 3D Gaussians here and will project them accordingly", 1)

    elif 'Gauss2D' in kwargs :
        Gaussians2D = np.asarray(kwargs.get('Gauss2D'))
        if Gaussians2D.size == 4 : 
            Gaussians2D = np.tile(Gaussians2D, tempMGE.nGauss).reshape(tempMGE.nGauss, 4) 
        elif Gaussians2D.size == 4 * tempMGE.nGauss :
            Gaussians2D = Gaussians2D.reshape(tempMGE.nGauss, 4) 
            found2D = 1
        elif Gaussians2D.size == 5 * tempMGE.nGauss:
            Gaussians2D = Gaussians2D.reshape(tempMGE.nGauss, 5) 
            found2D = 1
        elif Gaussians2D.size == 9 * tempMGE.nGauss:
            Gaussians2D = Gaussians2D.reshape(tempMGE.nGauss, 9) 
            found2D = 1
        else :
            print_msg("The provided 2D Gaussians have the wrong shape", 1)
            print_msg("We will instead set up a DUMMY set of 3D Gaussians ", 1)
            Gaussians3D = np.tile(temp3D, NGauss).reshape(NGauss, 9) 
            found3D = 1

    if found3D :
        tempMGE._findGauss3D = tempMGE.nGauss
        ## Projecting to get the 2D values
        tempMGE.Imax3D = Gaussians3D[:,0]
        tempMGE.Sig3Darc = Gaussians3D[:,1]
        tempMGE.QxZ = Gaussians3D[:,2]
        tempMGE.QyZ = Gaussians3D[:,3]
        tempMGE.ML = Gaussians3D[:,4]
        tempMGE.kRTheta = Gaussians3D[:,5]
        tempMGE.kRZ = Gaussians3D[:,6]
        tempMGE.GaussGroupNumber = np.asarray(Gaussians3D[:,7], int)
        tempMGE.GaussDynCompNumber = np.asarray(Gaussians3D[:,8], int)
        tempMGE.axi = 1
#        tempMGE.project(inclin=tempMGE.Euler[1], particles=False)
    else :
        tempMGE._findGauss2D = tempMGE.nGauss
        ## Deprojecting to get the 3D values
        tempMGE.Imax2D = Gaussians2D[:,0]
        tempMGE.Sig2Darc = Gaussians2D[:,1]
        tempMGE.Q2D = Gaussians2D[:,2]
        tempMGE.PAp = Gaussians2D[:,3]
        if Gaussians2D.shape[1] > 4:
            tempMGE.ML = np.asarray(Gaussians2D[:,4], float)
        if Gaussians2D.shape[1] > 5:
            tempMGE.kRTheta = np.asarray(Gaussians2D[:,5], float)
            tempMGE.kRZ = np.asarray(Gaussians2D[:,6], float)
            tempMGE.GaussGroupNumber = np.asarray(Gaussians2D[:,7], int)
            tempMGE.GaussDynCompNumber = np.asarray(Gaussians2D[:,8], int)
        tempMGE.axi = 1
#        tempMGE.deproject(inclin=tempMGE.Euler[1], particles=False)

    tempMGE.Center = np.asarray(kwargs.get('Center', np.zeros(2, float)))
    tempMGE.Dist = float(kwargs.get('Distance', 10.))
    tempMGE.Mbh = float(kwargs.get('MBH', 0.))
    tempMGE.mcut = float(kwargs.get('mcut', 50000.))

    tempMGE.nDynComp = kwargs.get("NDynComp", 1)
    tempMGE.nGroup = kwargs.get("NGroup", 1)
    tempMGE.nPartGroup = kwargs.get("NPartGroup", np.ones(tempMGE.nGroup, int))
    tempMGE.nRealisedPartGroup = kwargs.get("NRealisedPartGroup", np.ones(tempMGE.nGroup, int))

    tempMGE.write_mge(outdir=outdir, outfilename=outfilename, overwrite=overwrite)

    ###===============================================================
