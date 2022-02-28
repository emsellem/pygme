#! /usr/bin/python
# -*- coding: iso-8859-15 -*-
"""
This module allows to read and write snapshots from Ramses Simulations
"""

"""
Importing the most important modules
This module requires :
 - pylab
 - numpy version>=1.0
 - os
"""
import numpy as np
import os

from pygme.snapshot import SnapShot
from pygme.mge import MGE
from pygme.mge_miscfunctions import print_msg

__version__ = '1.0.2 (16 Aug 2013)'
# Version 1.0.2 : Checking the directory existence  in a better way (16 Aug 2013)
# Version 1.0.1 : - i -> inclin (25 Dec 2012)
# Version 1.0.0 : - First version

################################################################
#       CLASS SNAP_RAMSES: main class of this module
################################################################

class snap_ramses(SnapShot):
    # ##################################################
    #  INITIALISATION ##################################
    # ##################################################
    def __init__(self,*args):
        """Opens snapshots
        @version: 1.0.0 (Nov 23, 2010)
        @author: Eric Emsellem
        """
        self.typesim = "RAMSES"
        self.umass = 1.0 ## Mass
        self.udist = 1. ## 1 kpc
        self.udistcgs = 1000. ## in pc
        self.uvelo= 1.0 ## km/s
        self.arch = 0
        self.gui = 0
        self.units_sim(verbose=1)
        self.PA = self.inclin = self.PAb = self.inclinb = 0.
        self.gasdiscname = "exponential"
        self.RVcmaxpc = 100000.0   # 100 kpc
        self.nVc = 10000
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
        # For the moment does not do anything
        """
        @param file: name of the snapshot produced by RAMSES
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

        simufile.close()
        return

    #============================================================================

    # ###########################################################################
    # ### WRITE function ########################################################
    # ###########################################################################
    def write(self, Suffix=None, dirout=None, mgefile=None, mgemodel=None, verbose=0, mode=None) :
        """
        @param file: name of the snapshot produced by RAMSES
        @type  file: string (Default = None)
        @param verbose:
        @type  verbose: 0 or 1
        """
        ##=============== Checking input names and existence of file
        if Suffix is None :
            print('No Suffix provided! Will use default Suffix (RAMSES)')
            Suffix = "RAMSES"

        if dirout is None :
            print('No Input Directory (dirout) provided! Will use default (current directory)')
            dirout = "Ramses_temp/"
            if not os.path.isdir(dirout) :
                print("Creating RAMSES output directory Ramses_temp/")
                os.makedirs(dirout)
        if not os.path.isdir(dirout) :
            if mode == "O" :
                print("WARNING: directory does not exist, and is being created (%s)"%(dirout))
                os.makedirs(dirout)
            else :
                print("Directory %s doesn't exist, sorry!"%(dirout))
                print("Do you want it to be created now? :")
                import string
                yesOrNo = input("Do you want it to be created now? (y/N) :")
                if (string.lower(yesOrNo) in ["yes","y"])  :
                    os.makedirs(dirout)
                else :
                    print_msg("Ok, we stop here then")
                    return

        ## Check if mge file is given and if yes, read it
        if mgefile is None :
            if mgemodel is None :
                if 'MGE' not in self.__dict__ :
                    print("No MGE file provided, sorry!")
                    return
            ## Associating the given mgemodel to self.MGE
            else :
                self.MGE = mgemodel
        else :
            ## if mgefile provide, read it
            self.read_mge(mgefile=mgefile)

        ## Will now check the various files we need :
        ## First Vcirc_suff.dat
        ## Then  ic_part_suff.dat
        ## Then  info_suff.dat
        name_Vc = "Vcirc_%s.dat"%(Suffix)
        name_Info = "info_%s.txt"%(Suffix)
        name_IC = "ic_part_%s.dat"%(Suffix)
        listnames = [name_Vc, name_Info, name_IC]

        for name in listnames :
            filename = dirout + name
            if  os.path.isfile(filename) :
                if mode == "O" or  mode == "o":
                    print('WARNING: Path %s' %(filename), ' already exists! Will be overwritten')
                else :
                    print('ERROR: Path %s' %(filename), " already exists and mode is not 'O' (for Overwrite)!")
                    return

            try:
                tempfile = open(filename,'w')
            except IOError:
                print("Can\'t open file for writing.")
                return
            tempfile.close()

        ## Writing the circular velocity curve
        ## First generating the Vc curve in pc, km/s
        Rrangepc = np.linspace(0.,self.RVcmaxpc,self.nVc)
        Vc = self.MGE.Vcirc(R=Rrangepc / self.MGE.pc_per_arcsec)
        f_Vc = open(dirout + name_Vc, 'w')
        for i in range(self.nVc) :
            line = "%15.8e    %15.8e\n"%(Rrangepc[i],Vc[i])
            f_Vc.write(line)

        f_Vc.close()
        ## Writing the particles (Stars+DM)
        f_IC = open(dirout + name_IC, 'w')
        self.nsta = self.ndisc + self.nbulge
        self.rmax = np.int((np.int(np.max(np.sqrt(self.pos[:,0]**2 + self.pos[:,1]**2 + self.pos[:,2]**2))/10)+1)*10)
        for i in range(self.ngas + self.nsta, self.ngas + self.nsta + self.nhalo) :
            x = self.pos[i][0]
            y = self.pos[i][1]
            z = self.pos[i][2]
            Vx = self.vel[i][0]
            Vy = self.vel[i][1]
            Vz = self.vel[i][2]
            line = "%15.8e    %15.8e    %15.8e    %15.8e    %15.8e    %15.8e    %15.8e\n"%(x, y, z, Vx, Vy, Vz, self.pmass[i]/1.e9)
            f_IC.write(line)
        for i in range(self.ngas, self.ngas + self.nsta) :
            x = self.pos[i][0]
            y = self.pos[i][1]
            z = self.pos[i][2]
            Vx = self.vel[i][0]
            Vy = self.vel[i][1]
            Vz = self.vel[i][2] 
            line = "%15.8e    %15.8e    %15.8e    %15.8e    %15.8e    %15.8e    %15.8e\n"%(x, y, z, Vx, Vy, Vz, self.pmass[i]/1.e9)
            f_IC.write(line)


        f_IC.close()

        ## Writing the Info file
        self.init_mass()
        f_Info = open(dirout + name_Info, 'w')
        line = " h               =    1.00000000000000\n"
        line += " Lbox            =    %12.10f\n"%(self.rmax*2)
        line += " -Ndark_matter   =      %d\n"%(self.nhalo)
        line += " -Mass_dm        =    %10.4f\n"%(self.THaloMass/1.e9)
        line += " -Nstars_bulge   =            0\n"
        line += " -Mass_starsbulge=   0.000000000000000E+000\n"
        line += " -Nstars_disk    =      %d\n"%(self.nsta)
        line += " -Mass_starsdisk =    %10.4f\n"%(self.TStarMass/1.e9)
        line += " -Ngaz           =            0\n"
        line += " -Mass_gaz       =   0.000000000000000E+000\n"
        line += " -Nstar_form     =            0\n"
        line += " -Mass_sf        =   0.000000000000000E+000\n"
        line += " -Nblack_holes   =            0\n"
        line += " -Mass_bh        =   0.000000000000000E+000\n"
        line += " -Ntot           =      %d\n"%(self.nsta + self.nhalo)
        line += " -Mass_tot       =    %10.4f\n"%((self.TStarMass+self.THaloMass)/1.e9)
        f_Info.write(line)
        f_Info.close()

        return
    #============================================================================
    # ######################################################
    # ## READ MGE function #################################
    # ######################################################
    def read_mge(self, mgefile=None,verbose=0) :
        # For the moment does not do anything
        """
        @param file: name of the MGE model used for RAMSES input
        @type  file: string (Default = None)
        @param verbose:
        @type  verbose: 0 or 1
        """
        ##=============== Checking input names
        if mgefile is None :
            print('No file name provided!')
            return
        if not os.path.isfile(mgefile):
            print('Path %s' %(mgefile), ' does not exist, sorry!')
            return

        try:
            tmp = os.path.split(mgefile)
            if(tmp[0]!=''):
                dirfile = tmp[0] + "/"
                namefile = tmp[1]
            else :
                dirfile = "./"
                namefile = mgefile
        except IOError:
            print("Can\'t open file for reading.")
            return


        print("Opening the MGE file : %s"%(dirfile + namefile))
        self.mgefile = namefile
        self.mgedir = dirfile
        self.MGE = MGE(dirfile+namefile)
        return

    #============================================================================
