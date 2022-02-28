#! /usr/bin/python
# -*- coding: iso-8859-15 -*-
"""
This module allows to read and write snapshots from Gadget2 simulations
and includes some basic plotting routines to view the particles
"""

"""
Importing the most important modules
This module requires :
 - numpy version>=1.0
 - os
 - string
"""
import numpy as np
import os
import string

from . import rwcfor
from .rwcfor import intG, floatG, floatsG
from .rwcfor import sizefloatsG, sizeintsG, sizefloatG, sizeintG

from .mge_miscfunctions import print_msg

from pygme.snapshot import SnapShot

__version__ = '2.3.2 (25 Dec 201)'
# Version 2.3.2 : - i -> inclin
# Version 2.3.1 : - Cleaning and redistribution for release of pygme
# Version 2.3.0 : - Change a bug for the order of components also for pmass and ordermass
# Version 2.2.0 : - Change a bug for the order of components also for pos. vel and id
# Version 2.1.0 : - Force the format of arrays in writing
# Version 2.0.7 : - Incompatibilities between intG and sizeint => sizelong
# Version 2.0.6 : - Do not import all rwcfor anymore, and added test for end of file and corrected bug with Nwithmass
# Version 2.0.5 : - Added print_msg lines
# Version 2.0.4 : - Made a tiny change in the init
# Version 2.0.3 : - Fixed the units conversion for the masses
# Version 2.0.2 : - Fixed the umass to 10^10 solar masses
# Version 2.0.1 : - Fixed a bug in the writing: floatG to floatsG for mass and thermo param
# Version 2.0.0 : - Completely rewritten to incldue snapshot class
# Version 1.0.0 : - First version adapted from the version 2.1.4 of pmsphsf.py

################################################################
#       CLASS SNAP_GADGET: main class of this module
################################################################

class snap_gadget(SnapShot):
    # ##################################################
    #  INITIALISATION ##################################
    # ##################################################
    def __init__(self,*args):
        """Opens snapshots wri self.uvelotten by GADGET2.

        @version: 1.0.0 (Nov 13, 2007)
        @author: Nicolas Champavert and Eric Emsellem
        """
        self.typesim = "GADGET2"
        self.umass = 1.0e10 ## Mass
        self.udist = 1. ## 1 kpc
        self.udistcgs = 1000. ## in pc
        self.uvelo= 1.0 ## km/s
        self.arch = 0
        self.gui = 0
        self.units_sim(verbose=1)
        self.PA = self.inclin = 0.
        if (len(args) !=0) :
            if os.path.isfile(args[0]):
                self.read(*args)
            else :
                print('Path %s' %(args[0]), ' does not exists, sorry!')
                return
        else :
            self.indices_sim(mode=0)

        return
    #===================================================

    # ##################################################
    # ## READ function #################################
    # ##################################################
    def read(self, file=None,verbose=0) :
        """


        @param file: name of the snapshot produced by GADGET.
        @type  file: string (Default = None)
        @param verbose:
        @type  verbose: 0 or 1
        """
        ##=============== Checking input names
        if file is None :
            print('No file name provided!')
            return
        if not os.path.isfile(file):
            print('Path %s' %(file), ' does not exist, sorry!')
            return

        try:
            tmp = os.path.split(file)
            if(tmp[0]!=''):
                dirfile = tmp[0] + "/"
                namefile = tmp[1]
            else :
                dirfile = "./"
                namefile = file
            simufile = open(dirfile+namefile,'rb')
        except IOError:
            print("Can\'t open file for reading.")
            return

        if verbose: print('Fichier a ouvrir= ',file)
        ##=============== Checking input binary format and architecture
        ##                as well as the possible presence of a live halo
        b1 = np.fromfile(simufile,dtype=np.int32,count=1)[0]
        if verbose : print("b1 = ", b1)
        if b1 == 256:
            print(' DIGITAL, LINUX and WINDOWS format...')
            self.arch = 0
        elif b1 == 65536:
            print(' NEC and IBM format...')
            self.arch = 1
        else:
            print(' Unsupported format...')
            return

        # No difference when opening with arch = 0 et arch = 1
        # Byte-swapping when arch = 1 (done in function read_for_fast)
        simufile.close()
        simufile = open(dirfile+namefile,'rb')
        ##=============== Init the number of dimensions and other input numbers
        self.dim = 3
        self.namerun = namefile

        # Reading the header information first
        bytesleft=256-6*4 - 6*8 - 8 - 8 - 2*4-6*4
        narr = np.int(bytesleft/4)
        dummyarray = np.zeros(narr,np.int32)
        numbersdata = [6,6,1,1,1,1,6,narr]
        typedata = [intG,floatG,floatG,floatG,intG,intG,intG,intG]
        status, [b1,self.npart,self.masspart,self.t,self.redshift,self.sfr,self.fdback,self.npart_tot,dummyarray,b2] = rwcfor.read_for_fast(simufile, numbers=numbersdata, type=typedata,arch=self.arch)
        print_msg("Reading header information", status)

        self.ngas = self.npart[0]
        self.nhalo = self.npart[1]
        self.ndisc = self.npart[2]
        self.nbulge = self.npart[3]
        self.nform = self.npart[4]
        self.ndust = self.npart[5]

        self.ngas0 = self.ngas
        self.nsta = self.nsta0 = self.ndisc + self.nbulge

        self.ntot = np.sum(self.npart,axis=None)
        self.indices_sim()

        print('t =',self.t)                            # Time of the snapshot
        if verbose :
            print(' TOTAL particles = ', self.ntot, '\n', \
                  ' GAS particles = ', self.ngas, '\n', \
                  ' HALO particles = ', self.nhalo, '\n', \
                  ' DISK particles = ', self.ndisc, '\n', \
                  ' BULGE particles = ', self.nbulge, '\n', \
                  ' NEW particles = ', self.nform, '\n', \
                  ' DUST particles = ', self.ndust, '\n', \
                  ' MassPart = ', self.masspart, '\n')

        # ####################################
        #               GAS
        # ####################################
        if self.ntot == 0:
            print('ERROR: ntot is 0, no way to go on ...')
            return
        if self.ngas == 0:
            if verbose: print('WARNING: ngas=0, no gas here ... ')

        # ####################################
        # POSITIONS
        # ####################################
        numbersdata = [self.ntot*self.dim]
        typedata = [floatsG]
        status, [b1,self.pos,b2] = rwcfor.read_for_fast(simufile, numbers=numbersdata, type=typedata, arch=self.arch)
        print_msg("Reading positions", status)

        self.pos = (self.pos).reshape((self.ntot,self.dim))
        if verbose: print('pos',self.pos.shape)      ### VERBOSE ###

        # ####################################
        # VELOCITIES
        # ####################################
        status, [b1,self.vel,b2] = rwcfor.read_for_fast(simufile, numbers=numbersdata, type=typedata, arch=self.arch)
        self.vel = (self.vel).reshape((self.ntot,self.dim))
        print_msg("Reading velocities", status)

        if verbose: print('vel',self.vel.shape)     ### VERBOSE ###

        # ####################################
        # IDs
        # ####################################
        numbersdata = [self.ntot]
        typedata = [intG]
        status, [b1,self.id,b2] = rwcfor.read_for_fast(simufile, numbers=numbersdata, type=typedata, arch=self.arch)
        print_msg("Reading IDs", status)

        self.id = np.array(self.id).reshape(self.ntot,)
        if verbose: print('id',self.id.shape)     ### VERBOSE ###

        # ####################################
        # MASSES
        # ####################################
        ## Writing the masses for variable mass particles
        indmass = np.where((self.npart > 0) & (self.masspart == 0))
        if len(np.ravel(indmass)) > 0 :
            Nwithmass = np.sum(self.npart[indmass], axis=None)
        else :
            Nwithmass = 0

        if verbose : print("Nwithmass == %d" %(Nwithmass))
        if Nwithmass != 0 :
            numbersdata = [Nwithmass]
            typedata = [floatsG]
            status, [b1,self.pmass,b2] = rwcfor.read_for_fast(simufile, numbers=numbersdata, type=typedata, arch=self.arch)
            print_msg("Reading masses", status)
            self.pmass = np.array(self.pmass).reshape(Nwithmass,)
        else :
            self.pmass = np.zeros(1,floatsG)

        ## We are switching masses to have: gas, disc, bulge, new, halo, dust
        nstart = 0
        self.tempmass = np.zeros(self.ntot, floatsG)
        self.temppos = np.zeros_like(self.pos)
        self.tempvel = np.zeros_like(self.vel)
        self.tempid = np.zeros_like(self.id)
        self.ordermass = [0,3,2,4,1,5]
        for i in range(6) :
            indcomp = self.ordermass.index(i)
            npstart = np.sum(self.npart[self.ordermass[:indcomp]])
            npend = npstart + self.npart[i]
            if ((self.npart[i] > 0) & (self.masspart[i] == 0)):
                nend = nstart + self.npart[i]
                self.tempmass[npstart:npend] = self.pmass[nstart:nend]
                self.temppos[npstart:npend] = self.pos[nstart:nend]
                self.tempvel[npstart:npend] = self.vel[nstart:nend]
                self.tempid[npstart:npend] = self.id[nstart:nend]
                nstart = nend
            else :
                self.pmass[npstart:npend] = [self.masspart[i]] * self.npart[i]


        self.pmass = self.tempmass
        self.pos = self.temppos
        self.vel = self.tempvel
        self.id = self.tempid
        self.massg = self.masspart[0]
        self.massh = self.masspart[1]
        self.massd = self.masspart[2]
        self.massb = self.masspart[3]
        self.massn = self.masspart[4]
        self.massu = self.masspart[5]

        # ####################################
        # THERMODYNAMICS
        # ####################################
        if self.ngas != 0:
            ## Reading the physical parameters
            numbersdata = [self.ngas]
            typedata = [floatsG]
            ## Reading u
            status, [b1,self.u,b2] = rwcfor.read_for_fast(simufile, numbers=numbersdata, type=typedata, arch=self.arch)
            print_msg("Reading u", status)
            if not rwcfor.end_of_file(simufile) :
                ## Reading rho
                status, [b1,self.rho,b2] = rwcfor.read_for_fast(simufile, numbers=numbersdata, type=typedata, arch=self.arch)
                print_msg("Reading rho", status)
                self.rho = self.rho * self.udens
            if not rwcfor.end_of_file(simufile) :
                ## Reading h
                status, [b1,self.h,b2] = rwcfor.read_for_fast(simufile, numbers=numbersdata, type=typedata, arch=self.arch)
                print_msg("Reading h", status)

        self.pos = self.pos * self.udist  # to have kpc
        self.vel = self.vel * self.uvelo  # to have km/s
        self.pmass = self.pmass * self.umass  # to have 2 10^11 Msun

        # Initialization of rotation parameters
        self.PA = self.inclin = self.PAb = self.inclinb = 0.

        simufile.close()
        return

    #============================================================================

    # ###########################################################################
    # ### WRITE function ########################################################
    # ###########################################################################
    def write(self, file=None, verbose=0, mode=None, arch=None) :
        """
        @param file: name of the snapshot produced by GADGET.
        @type  file: string (Default = None)
        @param verbose:
        @type  verbose: 0 or 1
        """
        ##=============== Checking input names and existence of file
        if file is None :
            print('No file name provided!')
            return

        if  os.path.isfile(file) :
            if mode == "O" or  mode == "o":
                print('WARNING: Path %s' %(file), ' already exists! Will be overwritten')
            else :
                print('ERROR: Path %s' %(file), " already exists and mode is not 'O' (for Overwrite)!")
                return

        try:
            simufile = open(file,'wb')
        except IOError:
            print("Can\'t open file for writing.")
            return

        ##=============== Checking input binary format and architecture
        architecture = [('LINUX', 'WINDOWS', 'DIGITAL'), ('NEC', 'IBM')]
        if arch is None:
            self.arch = architecture[self.arch][0]
        else :
            self.arch = arch.upper()

        if verbose: print('File to write = ',file)
        ##=============== Init the number of dimensions and other input numbers
        if self.dim != 3 :
            print("ERROR: can only support Ndimensions = 3 (see self.dim)")
            return

        ## Writing the first few numbers
        bytesleft=256-6*4 - 6*8 - 8 - 8 - 2*4-6*4
        narr = np.int(bytesleft/4)
        dummyarray = np.zeros(narr,intG)

        ## Initialisation of npart structure
        self.npart = np.zeros(6, intG)
        self.npart[0] = self.ngas
        self.npart[1] = self.nhalo
        self.npart[2] = self.ndisc
        self.npart[3] = self.nbulge
        self.npart[4] = self.nform
        self.npart[5] = self.ndust
        self.ntot = np.sum(self.npart,axis=None)

        ## Initialisation of masspart structure
        self.masspart = np.zeros(6, floatG)
        self.masspart[0] = self.massg
        self.masspart[1] = self.massh
        self.masspart[2] = self.massd
        self.masspart[3] = self.massb
        self.masspart[4] = 0.0
        self.masspart[5] = 0.0

        sizedata=[sizeintG*6, sizefloatG*6, sizefloatG, sizefloatG, sizeintG, sizeintG, sizeintG*6, sizeintG*narr]
        status = rwcfor.write_for_fast(simufile, data=[self.npart, self.masspart, self.t,self.redshift,self.sfr,self.fdback,self.npart_tot, dummyarray], size=sizedata, arch=self.arch)

        ## We are switching masses to have: gas, halo, disc, bulge, new, dust
        self.ordermass = [0,3,2,4,1,5]
        nstart = 0
        self.temppos = np.zeros_like(self.pos)
        self.tempvel = np.zeros_like(self.vel)
        self.tempid = np.zeros_like(self.id)
        nstart = 0
        for i in range(6) :
            indcomp = self.ordermass.index(i)
            npstart = np.sum(self.npart[self.ordermass[:indcomp]])
            npend = npstart + self.npart[i]
            nend = nstart + self.npart[i]
            self.temppos[nstart:nend] = self.pos[npstart:npend]
            self.tempvel[nstart:nend] = self.vel[npstart:npend]
            self.tempid[nstart:nend] = self.id[npstart:npend]
            nstart = nend

        # ####################################
        # POSITIONS
        # ####################################
        sizedata = [self.ntot*self.dim*sizefloatsG]
        if verbose: print('pos',self.temppos.shape)      ### VERBOSE ###
        datatemp = (np.ravel(self.temppos / self.udist)).astype(floatsG) # going to PMSPH units from kpc
        if len(datatemp) != (self.ntot*self.dim) :
            print("ERROR: pos does not have the right size (self.ntot*self.dim)")
            return

        status = rwcfor.write_for_fast(simufile, data=[datatemp], size=sizedata, arch=self.arch)

        # ####################################
        # VELOCITIES
        # ####################################
        if verbose: print('vel',self.tempvel.shape)     ### VERBOSE ###
        datatemp = (np.ravel(self.tempvel / self.uvelo)).astype(floatsG)  # going to PMSPH units from km/s
        if len(datatemp) != (self.ntot*self.dim) :
            print("ERROR: vel does not have the right size (self.ntot*self.dim)")

        status = rwcfor.write_for_fast(simufile, data=[datatemp], size=sizedata, arch=self.arch)

        # ####################################
        # IDs
        # ####################################
        sizedata = [self.ntot*sizeintG]
        if verbose: print('ID',self.tempid.shape)     ### VERBOSE ###
        datatemp = (np.ravel(self.tempid)).astype(intG)
        if len(datatemp) != (self.ntot) :
            print("ERROR: ID does not have the right size (self.ntot)")

        status = rwcfor.write_for_fast(simufile, data=[datatemp], size=sizedata, arch=self.arch)

        # ####################################
        # MASSES
        # ####################################
        indmass = np.where((self.npart > 0) & (self.masspart == 0))
        if len(np.ravel(indmass)) > 0 :
            Nwithmass = np.sum(self.npart[indmass], axis=None)
        else :
            Nwithmass = 0

        ## We are switching masses to have: gas, halo, disc, bulge, new, dust
        if Nwithmass != 0 :
            self.tempmass = np.zeros(Nwithmass, floatsG)
            nstart = 0
            for i in range(6) :
                indcomp = self.ordermass.index(i)
                npstart = np.sum(self.npart[self.ordermass[:indcomp]])
                npend = npstart + self.npart[i]
                if ((self.npart[i] > 0) & (self.masspart[i] == 0)):
                    nend = nstart + self.npart[i]
                    self.tempmass[nstart:nend] = self.pmass[npstart:npend]
                    nstart = nend

            sizedata = [Nwithmass*sizefloatsG]
            datatemp = (np.ravel(self.tempmass / self.umass)).astype(floatsG) # to go from Msun to Gadget units
            status = rwcfor.write_for_fast(simufile, data=[datatemp], size=sizedata, arch=self.arch)

        # ####################################
        # THERMODYNAMICS
        # ####################################
        if self.ngas != 0:
            ## Writing out the physical properties for the gas
            if len(self.h) != self.ngas or len(self.u) != self.ngas or len(self.rho) != self.ngas :
                print("ERROR: h/u/rho do not have the right size (ngas)")
                return

            sizedata = [self.ngas*sizefloatsG]
            datatemp = np.zeros(self.ngas,floatsG)
            datatemp = floatsG(self.u)
            status = rwcfor.write_for_fast(simufile, data=[datatemp], size=sizedata, arch=self.arch)
            datatemp = floatsG(self.rho / self.udens)
            status = rwcfor.write_for_fast(simufile, data=[datatemp], size=sizedata, arch=self.arch)
            datatemp = floatsG(self.h)
            status = rwcfor.write_for_fast(simufile, data=[datatemp], size=sizedata, arch=self.arch)
            if status != 0 : print("Warning, status is", status)
        simufile.close() # Closing the main file

        return
    #============================================================================
