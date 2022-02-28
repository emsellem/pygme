#!/usr/bin/python
"""
This module contains an envelope class (MGE) which drives the rest of the package classes.

"""

# Importing the most important modules
# This MGE module requires numpy

from pygme.init_partMGE import nbodyMGE

__version__ = '4.0.4 (03/07/2012)'

DEBUG = 1
# Version 4.0.4: Small change replacing savedir by saveMGE
# Version 4.0.3: Cleaning and redistribution in modules for release pygme
# Version 4.0.2: Small bug with an expression on sigthetagas
# Version 4.0.1: Small bug when only reading projected quantities
# Version 4.0.0: Cleaning - using kwargs, removing all unnecessary parameters
# Version 3.9.1: Add softening and savefile for each model
# Version 3.9.0: Add the mge_to_ramses to simplify things
#                Homegeneisation of nBody to nPart
#                Adding the possibility not to realise ALL bodies
#                Adding the Dynamic Groups
#                Adding the possibility to only realise part of the particles
#                Debug factor on Mbh calculation
# Version 3.8.0: Add the mge_to_gadget to simplify things
# Version 3.7.9: Put the right sigmaR in QToomre
# Version 3.7.8: Included an external function with Dij for anisotropy variables
# Version 3.7.7: Change of float to floatMGE for exp
# Version 3.7.6: Small bug with FirstHalo stars - Lablanche
# Version 3.7.5: Small bug with TGroupMass
# Version 3.7.4: Add the Black Hole as a Dark Matter particle (last particle)
#                and change opNumber into comp_particles within addparam
# Version 3.7.3: Add opNumber option to get just the numbers
# Version 3.7.2: Major change to have single Gaussians with their own sigma
# Version 3.7.1: Add some None to imax and imin in the function calls
# Version 3.7.0: Add a Spin to be able to make counter-rotating components
# Version 3.6.1: Major debug when multiple components
# Version 3.6.0: Major rewriting of truncation of gaussians
# Version 3.5.5: Changed method to sample the position (Sphere / Cube)
# Version 3.5.4: Fixed distrib in mgetosnap
# Version 3.5.3: Fixed parameters to add imin/imax in rhop
# Version 3.5.2: Fixed parameters to include Halo particles
# Version 3.5.1: Fixed bug in derivation of theta (init_nbody)
# Version 3.5.0: Included Groups for the Gaussians, and cleaning a little
# Version 3.4.4: Added import floatMGE
# Version 3.4.3: Added specific MGE float to solve pb when Mbh is not 0 with exp
# Version 3.4.2: BUG! Already corrected??? Gas MUST BE BEFORE STARS in pmass
# Version 3.4.1: Added function set_minmax
# Version 3.4.0: Major change: using systematically imin, imax for functions
# Version 3.3.4: Debug the betaeps option
# Version 3.3.3: Debug the Gas mass when there is a Halo and some gas options
# Version 3.3.2: Added the option betaeps in the init_nbody to force beta(eps)
# Version 3.3.1: Added the option betaeps in the init_nbody to force beta(eps)
# Version 3.3.0: Added the epicycle approximation for the init of nbody
# Version 3.2.0: Changed anisotropy kRZ, KRTheta and added options in init_body
# Version 3.1.0: Added the derivation of kappa, Omega, and QToomre
# Version 3.0.1: Small bug in face-on projection
# Version 3.0.0: Reshuffling of all modules, with new snapshot
# Version 2.6.0: BUG. Mass of Gas was after Stars = INCONSISTENT!!
# Version 2.5.1: Changed default sigma for the gas
# Version 2.5.0: Added kR, kZ, kTheta
# Version 2.4.5: Debug: in weightGas, only take the right gaussians
# Version 2.4.4: Added the option of different kSatoh for the components
# Version 2.4.3: Changed rho to non 0 value and gamma to Ideal gas value
# Version 2.4.2: Added some more initialisation for snapshots
# Version 2.4.1: Minor changes to initialise self.axi
# Version 2.4.0: Introduced asarray instead of nfloat to convert arrays
# Version 2.3.0: Introduced the Halo Gaussians
# Version 2.2.2: Changed a few float32
# Version 2.2.1: Changed the name of pmsphpy a pmsphsf v1.6.0
# Version 2.2.0: Solved many bugs due to memory allocation
# Version 2.1.0: Added comp_Ep and comp_Ec modules
# Version 2.0.2: Added projection option
# Version 2.0.1: Changed value of G very slightly from Remco
# Version 2.0: Adding the gas and Black Holes!
# Version 1.9: Add a cut-off with Vescape
# Version 1.8: Debug rhoint which was wrong
# Version 1.7: Added the transformation to pmsph
# Version 1.6: Adding some comments
# Version 1.5: Adding of photometry deprojection
# Version 1.4: Adding of projected Jeans
# Version 1.1: Bug of biased distribution solved : R ==> x,y
# Version 1.0: first draft of the module

################################################################################
#                   MGE Functions    - Specific Class and associated functions
################################################################################

class MGE(nbodyMGE):
    """ Class MGE: includes a rather large structure describing the MGE
        Multi-Gaussian Expansion model for a specific galaxy. Basic variables are:
          - the gaussian parameters (Intensity, Sigma, Axis ratio, PA)
          - the M/L for each gaussian
          - the total mass and flux for each gaussian
          - parameters for the galaxy: distance, inclination...
          - and of course the number of particles per Group

        The MGE class is not inheriting from any pref-defined class.
        It is however, initialising the upper nbodyMGE class, which itself inherit
        from dynMGE (dynamics-related functions/methods), and then from photMGE
        (photometric-related functions/methods), and then from paramMGE (basic
        MGE parameters).
    """
    def __init__(self, infilename=None, indir=None, saveMGE=None, **kwargs) :
        """Wrapper around the top dynMGE class including the
        photometric and dynamic modules.

        :param name: str. infilename
        """
        nbodyMGE.__init__(self, infilename=infilename, indir=indir, saveMGE=saveMGE, **kwargs)
