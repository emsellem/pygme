#!/usr/bin/python
"""
This module includes some simple functions to transform
a simple input structure (nbody) into GADGET2, PMSPH, or RAMSES structures
for Initial conditions and snapshots
"""

"""
Importing the most import modules
This MGE module requires NUMARRAY and SCIPY
"""
import os
try:
    import numpy as np
except ImportError:
    raise Exception("numpy is required for pygme")

from numpy import cos, sin, sqrt

from .rwcfor import floatsG, floatG, intG

from pygme.plotsnap import PlotSnap

__version__ = '1.2.6 (25 December, 2012)'
# Version 1.2.6 : - i -> inclin
# Version 1.2.5 : - Clean version of pygme release
# Version 1.2.4 : - Transferred print_msg to rwcfor
# Version 1.2.3 : - Debug logspace
# Version 1.2.2 : - Added compute_vsigma in logR
# Version 1.2.1 : - Fixed order of ordermass
# Version 1.2.0 : - Fixed order of indices
# Version 1.1.5 : - Fixed parameters to include Halo particles
# Version 1.1.4 : - Added nform in ntot (init_indices)
# Version 1.1.3 : - Added print_msg function
# Version 1.1.2 : - Bug in the massd/massb calculation in init_mass
# Version 1.1.1 : - Debug compute_sigma
# Version 1.1.0 : - Added the function compute_sigma and init_var
# Version 1.0.2 : - Fixed tmd derived with StarMass and not GasMass
# Version 1.0.1 : - Fixed the conversion factors for mass and velocity in init_convsim

############################################
#       Physics Constants
############################################
GRAVITY   = 6.672e-8                              #PRINT 'GRAVITY=',GRAVITY,' (dyn cm^2 g^-2)'
BOLTZMANN = 1.3806e-16                            #PRINT 'BOLTZMANN=',BOLTZMANN,' (erg.K^-1)'
PROTONMASS = 1.6726e-24                           #PRINT 'MASSE DU PROTON=',PROTONMASS,' (g)'
SECONDPERYEAR = 3.155e+07       #31557600.        #PRINT 'NOMBRE DE SECONDES PAR AN=',SECONDPERYEAR,' s'
SOLARMASS = 1.989e33            #g                #PRINT 'MASSE DU SOLEIL=',SOLARMASS,' g'
PARSEC    = 3.085678e18         #cm               #PRINT 'PARSEC=',PARSEC,' cm'
GG=4.5e-15   #Msol/pc/an d'ou unit_gadget_dist=1000 !

#######################################################################################
#              NBODY Class = snapsim
#######################################################################################
##### Defining the snapsim class #################
""" Class snapshot: includes a struture with basic properties
   - pos = the positions of all particles
   - velo = velocities of these particles
   - self.ngas, ndisc, ntot, nhalo, nbulge, nform, ndust number of particles
   - umass, uvel, udist for units to be used in the writing of the binary files

The snapshot class is inheriting from the Plotting class for simulations
   UNITS for snapshots are:
     * kpc for distance
     * Km/s for velocities
     * Solar Masses for Masses
"""

class SnapShot(PlotSnap) :
    def __init__(self, name=None, verbose=1) :
        PlotSnap.__init__(self)
        self.typesim = "Standard"
        if (name is not None) :
            if os.path.isfile(name):
                self.read_ascii(name)
            else :
                print('Path %s' %(name), ' does not exists, sorry!')
                return
            self.indices_sim()
        else :
            self.umass = 1.0 ## 1 Solar Mass
            self.udist = 1.0 ## 1 kpc
            self.udistcgs = 1000.0 ## in pc (CGS)
            self.uvelo= 1.0 ## km/s
            self.indices_sim(mode=0)
            self.init_mass(mode=0) ## initialise the masses
            self.redshift = self.t = floatG(0.)
            self.sfr = 0
            self.fdback = 0

        self.init_var() ## Initialise the normal notations
        self.gui = 0
        self.PA = self.inclin = 0.
        self.units_sim(verbose=verbose)

    ############################################################
    ##    Units                                               ##
    ############################################################
    def units_sim(self, verbose=1):
        """
        Compute the time and density units and spit out some
        comments on the already available units
        """
        self.utime = sqrt((self.udistcgs**3)/(GG*self.umass))
        self.udens = float(self.umass/(self.udistcgs**3)) # -> 10^10 Msol/kpc^3
        self.uergg = float((self.udistcgs**2)/(self.utime**2)) # -> pc2/yr2
        # If we wish CGS, (as for GADGET) we need to do the following
        # but beware of convert_temperature which requires pc2/yr2
        # self.uergg = uergg*(PARSEC/SECONDPERYEAR)^2
        # energy per volume
        self.uergcc = float(self.umass/(self.udistcgs*self.utime**2))

        if verbose:
            print('----------------------------')
            print('Units in SIMULATION (%s):' %(self.typesim))
            print(' Mass (solar masses)=',self.umass)
            print(' Distance (kiloparcsec)=',self.udist)
            print(' Velocity (km/s)=',self.uvelo)
            print(' Constant of Gravitation (Msol/pc/yr)=',GG)
            print(' Density=',self.udens)
            print(' Time (years)=',self.utime)
            print(' Specific Internal Energy=',self.uergg)
            print(' Volume Internal Energy=',self.uergcc)
            print('----------------------------')

        return
    ###=============================================================================
    ############################################################
    ##    masses and other variables                          ##
    ############################################################
    def init_mass(self, mode=1):
        ## Mode = 0 means the arrays have not been initialised yet
        if mode == 0 :
            self.pos = np.zeros((self.ntot,3), floatsG)
            self.vel = np.zeros_like(self.pos)
            self.pmass = np.zeros(self.ntot, floatsG)

        self.massg = np.sum(self.pmass[:self.ngas],axis=None) * self.umass
        self.massb = np.sum(self.pmass[self.ngas:self.ngas+self.nbulge],axis=None) * self.umass
        self.massd = np.sum(self.pmass[self.ngas+self.nbulge:self.ngas+self.nbulge+self.ndisc],axis=None) * self.umass
        self.massn = np.sum(self.pmass[self.ngas+self.nbulge+self.ndisc:self.ngas+self.nbulge+self.ndisc+self.nform],axis=None) * self.umass
        self.massh = np.sum(self.pmass[self.ngas+self.nbulge+self.ndisc+self.nform:self.ntot],axis=None) * self.umass

        self.TGasMass = np.sum(self.pmass[:self.ngas], axis=None) * self.umass
        self.TStarMass = np.sum(self.pmass[self.ngas:self.ngas+self.nbulge+self.ndisc], axis=None) * self.umass
        self.TNewMass = np.sum(self.pmass[self.ngas+self.nbulge+self.ndisc:self.ngas+self.nbulge+self.ndisc+self.nform], axis=None) * self.umass
        self.THaloMass = np.sum(self.pmass[self.ngas+self.nbulge+self.ndisc+self.nform:self.ntot], axis=None) * self.umass

    ############################################################
    ##  Get the normal notations for positions and velocities ##
    ############################################################
    def init_var(self):
        self.x = self.pos[:,0]
        self.y = self.pos[:,1]
        self.z = self.pos[:,2]
        self.Vx = self.vel[:,0]
        self.Vy = self.vel[:,1]
        self.Vz = self.vel[:,2]
        self.R = sqrt(self.x**2 + self.y**2)
        self.theta = np.empty_like(self.x)
        mask = (self.x != 0)
        self.theta[mask] = np.arctan(self.y[mask] / self.x[mask]) + np.where(self.x[mask] > 0, 0., np.pi).astype(floatsG)
        self.theta[~mask] = np.where(self.y[~mask] >= 0, np.pi / 2., -np.pi / 2.).astype(floatsG)
        self.VR = self.Vx * cos(self.theta) + self.Vy * sin(self.theta)
        self.VTheta = - self.Vx * sin(self.theta) + self.Vy * cos(self.theta)

    ############################################################
    ##    Indices                                             ##
    ############################################################
    def indices_sim(self, mode=1):
        # Indices of particles. Using PMSPH rules

        if mode == 0 :
            self.ngas = 0
            self.nhalo = 0
            self.ndisc = 0
            self.nbulge = 0
            self.nform = 0
            self.ndust = 0
            self.ncold = 0

        # indices of particles
        if (self.typesim == "Standard") |  (self.typesim == "PMSPH") | (self.typesim == "GADGET2") | (self.typesim == "RAMSES"):
            ## Gas
            self.iming = 0
            self.imaxg = self.iming + self.ngas
            ## BULGE
            self.iminb = self.imaxg
            self.imaxb = self.iminb + self.nbulge
            ## DISC
            self.imind = self.imaxb
            self.imaxd = self.imind + self.ndisc
            ## NEW STARS
            self.iminn = self.imaxd
            self.imaxn = self.iminn + self.nform
            ## HALO
            self.iminh = self.imaxn
            self.imaxh = self.iminh + self.nhalo
##      elif (self.typesim == "GADGET2") :
##         ## Gas
##         self.iming = 0
##         self.imaxg = self.iming + self.ngas
##         ## HALO
##         self.iminh = self.imaxg
##         self.imaxh = self.iminh + self.nhalo
##         ## DISC
##         self.imind = self.imaxh
##         self.imaxd = self.imind + self.ndisc
##         ## BULGE
##         self.iminb = self.imaxd
##         self.imaxb = self.iminb + self.nbulge
##         ## NEW STARS
##         self.iminn = self.imaxb
##         self.imaxn = self.iminn + self.nform
        else :
            print("ERROR: type of simulation not recognised\n")
            return

        self.imins = (self.iming,self.iminb,self.imind,self.iminn,self.iminh)
        self.imaxs = (self.imaxg,self.imaxb,self.imaxd,self.imaxn,self.imaxh)
        self.ntot = self.ngas + self.nhalo + self.ndisc + self.nbulge + self.nform
        return
    ###=============================================================================
    ############################################################
    ##     READ ASCII                                         ##
    ############################################################
    def read_ascii(self, name=None) :
        # Check if name of file exists, and then open
        if os.path.isfile(name):
            ## Reading the file
            fsim = open(name)
            lines = fsim.readlines
            nlines = len(lines)

            ## Initialise some basic numbers
            self.nhalo = self.nform = self.ndust = self.ncold = 0
            self.t = self.redshift = floatG(0.)

            NbodyStar = NbodyGas = NbodyHalo = StarMass = GasMass = HaloMass = 0
            Gas = []
            Star = []
            Halo = []
            coordread = 0
            ## Loop on all lines
            for i in range(nlines) :
                ## Comment in file
                if lines[i][0] == "#" :
                    continue
                slines = lines[i].split()

                ## Mass of Stars: same for all except if coord has been read already
                if (lines[i][:7] == "STARMASS") & (not coordread) :
                    weightS = 1
                    StarMass = float(slines[1])
                ## Mass of Gas: same for all except if coord has been read already
                elif (lines[i][:6] == "GASMASS") & (not coordread) :
                    weightG = 1
                    GasMass = float(slines[1])
                ## Mass of Halo: same for all except if coord has been read already
                elif (lines[i][:6] == "HALOMASS") & (not coordread) :
                    weightH = 1
                    HaloMass = float(slines[1])

                ## Units
                elif (lines[i][:8] == "UNITMASS") :
                    self.umass = float(slines[1])
                elif (lines[i][:8] == "UNITDIST") :
                    self.udist = float(slines[1])
                elif (lines[i][:8] == "UNITVELO") :
                    self.uvelo = float(slines[1])
                elif (lines[i][:8] == "TIME") :
                    self.t = floatG(slines[1])
                elif (lines[i][:8] == "REDSHIFT") :
                    self.redshift = floatG(slines[1])
                elif (lines[i][:8] == "SFR") :
                    self.sfr = intG(slines[1])
                elif (lines[i][:8] == "FDBACK") :
                    self.fdback = intG(slines[1])

                ## If line has less than 7 items, continue
                if len(lines.split()) < 7 :
                    continue
                ## Otherwise read the coordinates, x, y, z, Vx, Vy, Vz, mass
                if lines[i][:7] == "GASCOORD" :
                    coordread = 1   ### coordinate will be read so do not force Star/Gas mass
                    if weightG :
                        Gas.append([float(slines[1]), float(slines[2]), float(slines[3]), float(slines[4]), float(slines[5]), float(slines[6]), GasMass])
                    else :
                        if len(slines) < 8 :
                            continue
                        Gas.append([float(slines[1]), float(slines[2]), float(slines[3]), float(slines[4]), float(slines[5]), float(slines[6]), float(slines[7])])
                    NbodyGas += 1
                elif lines[i][:7] == "STARCOORD" :
                    coordread = 1   ### coordinate will be read so do not force Star/Gas mass
                    if weightS :
                        Star.append([float(slines[1]), float(slines[2]), float(slines[3]), float(slines[4]), float(slines[5]), float(slines[6]), StarMass])
                    else :
                        if len(slines) < 8 :
                            continue
                        Star.append([float(slines[1]), float(slines[2]), float(slines[3]), float(slines[4]), float(slines[5]), float(slines[6]), float(slines[7])])
                    NbodyStar += 1
                elif lines[i][:7] == "HALOCOORD" :
                    coordread = 1   ### coordinate will be read so do not force Star/Gas mass
                    if weightH :
                        Halo.append([float(slines[1]), float(slines[2]), float(slines[3]), float(slines[4]), float(slines[5]), float(slines[6]), HaloMass])
                    else :
                        if len(slines) < 8 :
                            continue
                        Halo.append([float(slines[1]), float(slines[2]), float(slines[3]), float(slines[4]), float(slines[5]), float(slines[6]), float(slines[7])])
                    NbodyHalo += 1

            ## Closing the input file
            fsim.close()
            ## Now fill in the coordinates in the structure
            self.ngas = NbodyGas
            self.ndisc = NbodyStar / 3
            self.bulge = NbodyStar - self.ndisc
            self.nhalo = NbodyHalo
            self.ntot = NbodyStar + NbodyGas + NbodyHalo

            self.pos = np.zeros((self.ntot,3), floatsG)
            self.vel = np.zeros_like(self.pos)
            self.pmass = np.zeros(self.ntot, floatsG)
            k = 0
            for i in range(self.NbodyGas) :
                self.pos[k][0] = Gas[i][0] / self.udist
                self.pos[k][1] = Gas[i][1] / self.udist
                self.pos[k][2] = Gas[i][2] / self.udist
                self.vel[k][0] = Gas[i][3] / self.uvelo
                self.vel[k][1] = Gas[i][4] / self.uvelo
                self.vel[k][2] = Gas[i][5] / self.uvelo
                self.pmass[k] = Gas[i][6] / self.umass
                k += 1
            for i in range(self.NbodyStar) :
                self.pos[k][0] = Star[i][0] / self.udist
                self.pos[k][1] = Star[i][1] / self.udist
                self.pos[k][2] = Star[i][2] / self.udist
                self.vel[k][0] = Star[i][3] / self.uvelo
                self.vel[k][1] = Star[i][4] / self.uvelo
                self.vel[k][2] = Star[i][5] / self.uvelo
                self.pmass[k] = Star[i][6] / self.umass
                k += 1
            for i in range(self.NbodyHalo) :
                self.pos[k][0] = Halo[i][0] / self.udist
                self.pos[k][1] = Halo[i][1] / self.udist
                self.pos[k][2] = Halo[i][2] / self.udist
                self.vel[k][0] = Halo[i][3] / self.uvelo
                self.vel[k][1] = Halo[i][4] / self.uvelo
                self.vel[k][2] = Halo[i][5] / self.uvelo
                self.pmass[k] = Halo[i][6] / self.umass
                k += 1
            ## Total mass of Gas and Stars and Halo
            self.init_mass(mode=1)

    ############################################################
    ##  Compute the radial profile (Eq plane) of the dispersion #
    ############################################################
    def compute_vsigma(self, nbins=20, dz=np.array([-0.2,0.2]), Rmax=None):
        """
           Compute the dispersions in the three directions
           selecting paticles in the Equatorial plane (or close to it)
           nbins : number of bins (default = 20)
           dz : vertical range over which to select (default = 200 pc)
           Rmax : maximum radius for this profile (default = None)
        """
        sz = np.zeros(nbins, dtype=np.float32)
        sR = np.zeros_like(sz)
        sT = np.zeros_like(sz)
        vz = np.zeros(nbins, dtype=np.float32)
        vR = np.zeros_like(sz)
        vT = np.zeros_like(sz)
        # Selection in Z
        selectZ = np.where((dz[0] < self.z) & (self.z < dz[1]))

        x = self.x[selectZ]
        y = self.y[selectZ]
        R = sqrt(x**2+y**2)

        VR = self.VR[selectZ]
        VT = self.VTheta[selectZ]
        Vz = self.Vz[selectZ]

        if Rmax is None : Rmax = R.max()
        Rbin = np.logspace(-2.0,np.log10(Rmax),nbins)

        Rdigit = np.digitize(R, Rbin)
        for i in range(nbins) :
            Rdigit_temp = np.where(Rdigit == i)[0]
            sR[i] = VR[Rdigit_temp].std()
            sz[i] = Vz[Rdigit_temp].std()
            sT[i] = VT[Rdigit_temp].std()
            vR[i] = VR[Rdigit_temp].mean()
            vz[i] = Vz[Rdigit_temp].mean()
            vT[i] = VT[Rdigit_temp].mean()

        return Rbin, vR, vT, vz, sR, sT, sz
