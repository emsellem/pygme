#!/usr/bin/python
"""
This module includes some transformation function to convert a snapshot into
another format (GADGET, RAMSES)
"""

"""
Importing the most import modules
This MGE module requires NUMARRAY and SCIPY
"""
try:
    import numpy as np
except ImportError:
    raise Exception("numpy is required for pygme")

from pygme.rwcfor import floatsG, floatG, intG

__version__ = '1.0.0 (10 January, 2012)'

from pygme.pygadget import snap_gadget
from pygme.pyramses import snap_ramses
from . import snapshot as sp
from pygme.rwcfor import floatMGE

##################################################################
# Switch Comp 1 and 2
##################################################################
def switchcomp(inarray, outarray, n1, n2) :
    """
    Specific function to switch two chunks of data within the array
    """
    outarray[:n2] = inarray[n1:n1+n2]
    outarray[n2:n2+n1] = inarray[:n1]
    outarray[n1+n2:] = inarray[n1+n2:]
    return outarray
#===================================================================

##################################################################
### Common rules for Transforming the N body into RAMSES/GADGET   #
##################################################################
def snapshot_to_other(snapshot, type="GADGET2") :
    snapshot.typesim = type
    if (type == "GADGET2") :
        snap = snap_gadget()
        snap.redshift = floatG(snapshot.redshift)
        snap.sfr = intG(snapshot.sfr)
        snap.fdback = intG(snapshot.fdback)
        snap.npart_tot = np.zeros(6, intG)
        # In snap, order is : Gas, Bulge, Disc, New, Halo
        # In gadget file, order is : Gas, Halo, Disc, Bulge, New
        snap.npart_tot[:] = np.array([snapshot.ngas, snapshot.nhalo,
            snapshot.ndisc, snapshot.nbulge, snapshot.nform, snapshot.ndust])
        snap.id = intG(range(snapshot.ntot))
        snap.ordermass = [0,3,2,4,1,5]
    elif (type == "RAMSES") :
        snap = snap_ramses()
    else :
        print("ERROR: type not recognised: %s - Should be GADGET2 or RAMSES!" %(type))
        return

    ## Initialisation of required parameters
    snap.dim = 3
    snap.t = floatG(snapshot.t)

    # Number of particles
    snap.ngas =  snapshot.ngas
    snap.nsta = snapshot.ndisc + snapshot.nbulge
    snap.nhalo = snapshot.nhalo
    snap.ndisc = snapshot.ndisc
    snap.nbulge = snapshot.nbulge
    snap.pmass = snapshot.pmass  # Msun
    snap.pos = snapshot.pos      # kpc
    snap.vel = snapshot.vel      # km/s

    snap.ngas0 = snapshot.ngas
    snap.nsta0 = snap.nsta

    # Some mass
    snap.massg = snapshot.massg
    snap.massh = snapshot.massh
    snap.massd = snapshot.massd
    snap.massb = snapshot.massb

    # Some thermo parameters
    snap.gamma = float(5. / 3.)  ## Ideal gas monoatomic
    snap.h = np.zeros(snapshot.ngas,floatsG) + 0.1
    snap.u = np.zeros(snapshot.ngas,floatsG) + 2.e-4
    snap.rho = np.zeros(snapshot.ngas,floatsG) + 0.01
    snap.dq = np.zeros(snapshot.ngas,floatsG)
    snap.pdv = np.zeros(snapshot.ngas,floatsG)

    # Total mass for Gas and Stars
    snap.tmg0 = snapshot.TGasMass / snap.umass
    snap.tmg = snap.tmg0
    snap.tms0 = snapshot.TStarMass / snap.umass
    snap.tms = snap.tms0
    snap.tmh0 = snapshot.THaloMass / snap.umass
    snap.tmh = snap.tmh0
    snap.tmb = np.sum(snap.pmass[snap.ngas:snap.ngas+snap.nbulge], axis=0)
    snap.tmd = np.sum(snap.pmass[snap.ngas+snap.nbulge:snap.ngas+snap.nbulge+snap.ndisc],axis=0)

    # Misc numbers
    snap.tmf = float(0.)
    snap.tkin = snap.tkins = snap.Epot = snap.tterm = snap.angtot = float(0.)
    snap.angtos = snap.angtog = float(0.)
    if snap.ngas != 0 :
        snap.gmg = snap.tmg / snap.ngas
    else :
        snap.gmg = 0.
    if snap.nbulge != 0 :
        snap.gmb = snap.tmb / snap.nbulge
    else :
        snap.gmb = 0.
    if snap.ndisc != 0 :
        snap.gmd = snap.tmd / snap.ndisc
    else :
        snap.gmd = 0.
    snap.erotg0 = snap.erots0 = floatsG(0.)

    # Chemistry
    snap.nchemo = 0
    snap.iche = 0
    snap.OC = np.zeros(1, floatsG)
    snap.YC = np.zeros(1, floatsG)
    snap.ZC = np.zeros(1, floatsG)

    snap.PA = snap.i = snap.PAb = snap.ib = float(0.)
    snap.ah = snap.rhmax = snap.crocst = float(0.)

    snap.isf = 0
    snap.posb = 0
    snap.vitb = 0
    snap.lphi = 0
    # ####################################
    # Updating non active parameters
    # ####################################
    snap.tmdu = 0.
    snap.tmc = 0.

    snap.indices_sim()

    return snap


##################################################################
### TRansform the N body into a SNAPSHOT FILE                    #
##################################################################
def mge_to_snapshot(MGEmodel, verbose=0) :
    snap = sp.SnapShot(verbose=0)

    UnitLength_MGE = floatMGE(1000.)  # L unit is in kpc for simulations and pc for MGE
    UnitVelo_MGE = floatMGE(1.)  # V unit is in km/s in both MGE and sim
    snap.udist = 1.  # kpc (MGE is in pc, but sim is in kpc)
    snap.uvelo = 1.  # km/s
    snap.units_sim(verbose=1)

    ## Initialisation of required parameters
    snap.dim = 3
    snap.t = floatMGE(0.)
    snap.nhalo = MGEmodel.nRealisedPartHalo + MGEmodel.nRealisedPartBH
    snap.ngas = MGEmodel.nRealisedPartGas
    snap.ntot = MGEmodel.nRealisedPart
    snap.ngas0 = MGEmodel.nRealisedPartGas
    snap.nsta = MGEmodel.nRealisedPartStar
    snap.nbulge = MGEmodel.nRealisedPartStar / 3
    snap.ndisc = MGEmodel.nRealisedPartStar - snap.nbulge
    snap.nsta0 = MGEmodel.nRealisedPartStar

    # Positions and velocities
    snap.pos = np.zeros((snap.ntot,snap.dim),floatMGE)
    snap.vel = np.zeros((snap.ntot,snap.dim),floatMGE)

    snap.pos[:,0] = switchcomp(MGEmodel.x / UnitLength_MGE, snap.pos[:,0], MGEmodel.nRealisedPartStar, MGEmodel.nRealisedPartGas)
    snap.pos[:,1] = switchcomp(MGEmodel.y / UnitLength_MGE, snap.pos[:,1], MGEmodel.nRealisedPartStar, MGEmodel.nRealisedPartGas)
    snap.pos[:,2] = switchcomp(MGEmodel.z / UnitLength_MGE, snap.pos[:,2], MGEmodel.nRealisedPartStar, MGEmodel.nRealisedPartGas)
    snap.vel[:,0] = switchcomp(MGEmodel.Vx / UnitVelo_MGE, snap.vel[:,0], MGEmodel.nRealisedPartStar, MGEmodel.nRealisedPartGas)
    snap.vel[:,1] = switchcomp(MGEmodel.Vy / UnitVelo_MGE, snap.vel[:,1], MGEmodel.nRealisedPartStar, MGEmodel.nRealisedPartGas)
    snap.vel[:,2] = switchcomp(MGEmodel.Vz / UnitVelo_MGE, snap.vel[:,2], MGEmodel.nRealisedPartStar, MGEmodel.nRealisedPartGas)

    # Some thermo parameters
    snap.gamma = 5. / 3. ## Ideal gas monoatomic
    snap.h = np.zeros(MGEmodel.nRealisedPartGas,floatMGE) + 0.1
    snap.u = np.zeros(MGEmodel.nRealisedPartGas,floatMGE) + 2.e-4
    snap.rho = np.zeros(MGEmodel.nRealisedPartGas,floatMGE) + 0.001
    snap.dq = np.zeros(MGEmodel.nRealisedPartGas,floatMGE)
    snap.pdv = np.zeros(MGEmodel.nRealisedPartGas,floatMGE)

    # Mass of each particle
    snap.pmass = np.zeros(snap.ntot, floatMGE)
    snap.pmass = switchcomp(MGEmodel.BodMass / snap.umass, snap.pmass, MGEmodel.nRealisedPartStar, MGEmodel.nRealisedPartGas)

    # Total mass for Gas and Stars (truncated)
    snap.tms0 = MGEmodel.truncStarMass / snap.umass
    snap.tms = snap.tms0
    snap.tmg0 = MGEmodel.truncGasMass / snap.umass
    snap.tmg = snap.tmg0
    snap.tmh0 = (MGEmodel.truncHaloMass + MGEmodel.Mbh) / snap.umass
    snap.tmh = snap.tmh0
    snap.tmb = np.sum(snap.pmass[MGEmodel.nRealisedPartGas:MGEmodel.nRealisedPartGas+snap.nbulge], axis=0)
    snap.tmd = np.sum(snap.pmass[MGEmodel.nRealisedPartGas+snap.nbulge:MGEmodel.nRealisedPartGas+MGEmodel.nRealisedPartStar],axis=0)

    ## Total mass (in Nbody: as truncated by Rcut and Zcut)
    snap.truncStarMass = MGEmodel.truncStarMass
    snap.truncGasMass = MGEmodel.truncGasMass
    snap.truncHaloMass = (MGEmodel.truncHaloMass)
    snap.Mbh = MGEmodel.Mbh
    snap.indices_sim()
    snap.init_var()

    return snap

##################################################################
### TRansform the N body into a GADGET2 FILE                     #
##################################################################
def mge_to_gadget(MGEmodel, filename=None,  mode="O", arch="Linux") :
    """
       Convert the Mge Nbody realisation and write a gadget file
       mode : default is O for Overwrite (the output file will be overwritten)
       arch : default is Linux
    """
    ## Convert into a snapshot
    snap = mge_to_snapshot(MGEmodel)
    ## Convert into a gadget snapshot
    snap_gadget = snapshot_to_other(snap, type="GADGET2")
    ## Write the Gadget dat file
    print("Writing the Gadget file : %s"%(filename))
    snap_gadget.write(file=filename, mode=mode, arch=arch)

##################################################################
### TRansform the N body into RAMSES FILEs                       #
##################################################################
def mge_to_ramses(MGEmodel, dirout=None, Suffix=None, mgefile=None, mode="O", verbose=0) :
    """
       Convert the Mge Nbody realisation and write a gadget file
       mode : default is O for Overwrite (the output file will be overwritten)
       arch : default is Linux
    """
    ## Convert into a snapshot
    snap = mge_to_snapshot(MGEmodel)
    ## Convert into a RAMSES snapshot
    snap_ramses = snapshot_to_other(snap, type="RAMSES")
    ## Write the RAMSES dat file
    print("Writing the RAMSES files with Suffix %s"%(Suffix))
    snap_ramses.write(Suffix=Suffix, dirout=dirout, mgefile=mgefile, verbose=verbose, mode=mode)
