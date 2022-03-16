#! /usr/bin/env python
"""
This module provides simple tools to derive a number of profiles, functions etc,
which can be useful in Astronomy

This includes:
    Sersic profiles
    Exponential (1D, 2D)
    Einasto, Navarro/Frenk/White profiles
"""

# Importing the required stuff from different modules
import matplotlib as mpl

# Numpy
import numpy as np
from numpy import pi, log

# Scipy and functions
import scipy
from scipy import interpolate, special, integrate
from scipy.special import gammainc, gamma, kv, k0
from numpy import float64 as nfloat

__version__ = '1.1.0 (March 3, 2012)'

###################################################
## Profiles in 1D which could be useful
###################################################
## Navarro Frenk White profile
def nfw(r, Rs=None, rho0=None, c=None, Mvir= None, alpha=-0.12, kc=12.0) :
    """
    NFW profile with Rs in kpc and rho0 in Msun/kpc-3
    r: input radial values (in kpc)

    We need as input:

    Rs: scale radius (in kpc)
    rho0: normalised density

    or
       c : concentration parameter
    or
       Mvir: virial mass (total) - c or Mvir should be given
    alpha: 
    kc : relation between Mvir and c, defaults are -0.12, andd 12.0
    """
    rho = np.zeros_like(r)
    H0 = 72 # km/s/Mpc
    # G is in (km/s)2. Msun-1 . pc .
    G = 0.0043225524
    rhocrit = 3. * H0**2 / (8. * pi * G * 1.0e3) # in Msun kpc**-3

    if Rs != None and rho0 != None :
        print("WARNING: We use Rs and rho0 for the profile")
    else :
        if c == None :
            if Mvir == None :
               print("ERROR: either c or Mvir should be provided")
               return [0.]
            c = kc * (Mvir / 1.e12)**(alpha)
        else :
            if Mvir != None :
                print("ERROR: either c or Mvir should be provided, not BOTH")
                return [0.]
            Mvir = 1.0e12 * (c / kc)**(1./alpha) 

        Rvir = (Mvir * 3.0 / (4. * pi * 360. * rhocrit * 0.27))**(1./3.)
        Rs = Rvir / c

    ### profile
    print(c, Rs, Mvir, Rvir)
    sel_radius = (r < c * Rs)
    rho0 = Mvir / (4. * pi * Rs**3 * (log(1. + c) - c/ (1. + c)))
    rho[sel_radius] = rho0 / ((r[sel_radius] / Rs) * (1. + r[sel_radius] / Rs)**2)

    return rho

def einasto_rho2_r2(r2=None, rho2=None):
    """
    Relation from
    Li, P., Lelli, F., McGaugh, S., & Schombert, J. 2018a, A&A, 615, A3
    log(ρ−2) = (−1.32 ± 0.15) × log(r−2) − (1.27 ± 0.18).
    and
    log(n) = (+0.5 ± 0.15) × log(r−2) − (0.16 ± 0.18).

    Input
    -----
    r2: in kpc
    rho2: in Mpc-3
    """
    import numpy as np
    if r2 is None:
        if rho2 is None:
            print("Error: you need to set up at least one of the two")
            return 0., 0.
        else:
            print("rho2 is set up and will be used as a priority")
            r2 =  10**(- (np.log10(rho2) + 1.27) / 1.32)
            r2m =  10**(- (np.log10(rho2) + 1.09) / 1.47)
            r2p =  10**(- (np.log10(rho2) + 1.45) / 1.17)
            rho2m = rho2p = rho2
            print(f"r2 = {r2} [{r2m} -- {r2p}]")
    else:
        print("r2 is set up and will be used as a priority")
        rho2 =  10**(-1.32 * np.log10(r2) - 1.27)
        rho2p =  10**(-1.17 * np.log10(r2) - 1.09)
        rho2m =  10**(-1.47 * np.log10(r2) - 1.45)
        r2p = r2m = r2
        print(f"rho2 = {rho2} [{rho2m} -- {rho2p}]")

    ns = 10**(np.log10(r2) * 0.5 - 0.16)
    nsp = 10**(np.log10(r2) * 0.65 + 0.02)
    nsm = 10**(np.log10(r2) * 0.35 - 0.34)
    return [r2, r2m, r2p], [rho2, rho2m, rho2p], [ns, nsm, nsp]
    
def comp_einasto(r2=1000., rho2=0.001, n=1, rsample=None,
                 rmax=30000.0, nr=201):
    """Einasto Profile

    Input
    -----
    r2 (float): radius in parsec
    rho2
    n (int): index
    rsample (float array): default is None

    Returns
    -------
    rsample (in pc)
    bkt (in Msun/pc3)
    Mr (in Msun)
    Vc (in km/s)    
    """
    if rsample is None:
        rsample = linspace(0, 30000.0,201)  # in pc

    rhoE = rho2 * np.exp(-2. * n * ((rsample/r2)**(1./n) - 1.))

    Mtot = 4. * np.pi * rho2 * r2**3 * n * (2*n)**(-3*n) * gamma(3*n) * np.exp(2*n)
    Mr = Mtot * gammainc(3* n, 2 * n * (rsample/r2)**(1./n))
    G = 0.0043225821 # in (km/s)2. Msun-1 . pc
    Vc = np.sqrt(G * Mr / rsample) # in km/s
    return rsample, rhoE, Mr, Vc

############################################################
# BURKERT rho(r) = rhos / [ (1+r/rs) * (1+(r/rs)^2) ],
# Donato 2009 -> log(rhos * rs) = 2.15 +/-0.2
#====================================================
def comp_burkert(rs=1000., deltarho=0, rsample=None,
                 rmax=30000.0, nr=201):
    """Burkert profiles

    Input
    -----
    rs (float): radius in parsec
    rsample (float array): default is None

    Returns
    -------
    rsample (in pc)
    bkt (in Msun/pc3)
    Mr (in Msun)
    Vc (in km/s)
    """
    if rsample is None:
        rsample = linspace(0, rmax, nr)  # in pc
    rhobkt = 10**(2.15 + deltarho) / rs # in Msun pc-3
    x = rsample / rs
    bkt = rhobkt / ((1. + x) * (1. + x**2))
    M0 = np.pi * 2. * rhobkt * rs**3 # in Msun
    Mr = M0 * (np.log(1. + x) - np.arctan(x) + 0.5 * np.log(1. + x**2)) # in Msun
    G = 0.0043225821 # in (km/s)2. Msun-1 . pc
    Vc = np.sqrt(G * Mr / rsample) # in km/s
    return rsample, bkt, Mr, Vc

def m200_to_mstar(M200, beta=1.057, gamma=0.556, mM0=0.02820, logM1=11.884):
    M1 = 10**(logM1)
    Mstar = 2. * M200 * mM0 / ((M200 / M1)**(-beta) + (M200 / M1)**gamma)
    return Mstar

def m200_to_mstar_moster(M200, beta=1.376, gamma=0.608, mM0=0.0351, logM1=11.59):
    M1 = 10**(logM1)
    Mstar = 2. * M200 * mM0 / ((M200 / M1)**(-beta) + (M200 / M1)**gamma)
    return Mstar

def m200_to_c200_maccio(M200, a=0.83, b=-0.098):
    return 10**(a - b * np.log10(M200 / 1e12))

def m200_to_c200_dutton14(M200, a=0.997, b=-0.13):
    return 10**(a - b * np.log10(M200 / 1e12))

def c200_to_v200(c200=1., rs=1.0, H0=73):
    """get C200
    H0 in km/s/Mpc
    rs in Mpc 
    c200 is dimensionless

    Return V in km/s
    """
    return 10. * c200 * rs * H0

def mstar_to_halo_mm(Mstar):
    from scipy.interpolate import interp1d
    M200range = np.logspace(9, 16, 201)
    Msrange = m200_to_mstar_moster(M200range)
    fM = interp1d(Msrange, M200range)
    M200 = fM(Mstar)
    c200 = m200_to_c200_maccio(M200)
    return M200, c200
