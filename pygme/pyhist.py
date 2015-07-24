"""
This module allows to compute histograms in an efficient way
"""

"""
Importing the most important modules
This module requires :
 - numpy version>=1.0
"""
import numpy as np

__version__ = '1.0.4 (16 July 2014)'
# Version 1.0.4 : - Changed comp_moments output (now return X,Y too)
# Version 1.0.3 : - Changed comp_moments and comp_losvd
# Version 1.0.2 : - First cleaning for release of pygme
# Version 1.0.1 : - Added a couple of functions for moments
# Version 1.0.0 : - First version adapted from the version 2.1.4 of pmsphsf.py
###################
def hist1d_bincount(x,nbin,weights=np.array([0.])):
    """
    Compute the 1D histogram, using array of weights
    This uses the rather inefficient bincount method
    It may be more efficient to use the numpy histogram functions
    """

    delta = (x.max()-x.min())/nbin
    indices = np.array((x[:]-x.min())/delta, dtype=np.int32)
    mask = (indices == nbin)
    indices[mask] = nbin-1
    # the last bin is inclusive, [ ],
    # and the other ones are [ [
    bornes = x.min() + np.arange(nbin)*delta

    if((weights==0.).all()):
        hist = np.bincount(indices)
    else:
        hist = np.bincount(indices,weights)

    return (hist,bornes)

############################
def hist2d_bincount(x,y,n,lim,weights):
    """
    Make a 2D weighted histogram

    @param x: x position of selected particles
    @type  x: 1D array
    @param y: y position of selected particles
    @type  y: 1D array
    @param n: histogram size. The size of hist is n*n.
    @type  n: integer
    @param lim: limits for the selection of particles
    @type  lim: tuple (xmin,xmax,ymin,ymax)
    @param weights: weights
    @type  weights: 1D array (same length as x and y)

    @return: hist
    @rtype: weighted histogram (2D array)
    This uses the rather inefficient bincount method
    It may be more efficient to use the numpy histogram functions
    """
    indices = np.array(n*(x-lim[0])/(lim[1]-lim[0]),dtype=np.int32) + n*np.array(n*(y-lim[2])/(lim[3]-lim[2]),dtype=np.int32)
    #we add one "ghost particle" with weight=0 at the end
    indices = np.concatenate((indices,[n*n-1])) #
    weights = np.concatenate((weights,[0.]))
    hist = np.bincount(indices,weights)
    hist = hist.reshape((n,n))
    return hist

###########################################################
def select_xy(x,y, lim,ilist=None,indxy=None):
    """
    Calculate the selection of particles in x,y given lim

    :param lim: limits for the selection of particles
    :type  lim: tuple (xmin,xmax,ymin,ymax,zmin,zmax)
    :param ilist: indices of the particles for the selected population
    :type  ilist: list of integer
    :return: arrays of positions of selected particles and arrays of indices of selected particles in xy, zy and xz
    :rtype: arrays
    """
    if indxy is None:
        indxy = np.array([],dtype=np.int32)

    if imax is None :
        imax = x.shape[0]

    ## WARNING: np.nonzero now returns a tuple with only one element (an array with the indices...)
    return np.concatenate((indxy,ind[np.nonzero((lim[0]<x[ilist])& (x[ilist]<lim[1]) &\
                                                   (lim[2]<y[ilist])& (y[ilist]<lim[3]))[0]]))

#################################################
def comp_moments_bincount(x,y,val,w,lim,nXY):
    """

    WARNING : DEPRECATED!! Use comp_moments instead!

    Calculate the first 2 velocity moments
    x : x coordinate
    y : y coordinate
    val : velocity
    w : weight (e.g., mass)
    lim : (xmin, xmax, ymin, ymax)
    nXY : number of bins in X and Y

    Return M, M*V, V, M*(V*V+S*S), S
    Uses the inefficient bincount method: look at the comp_moments
    using histogram functions from numpy (also available in pygme)
    """
    indxy = select_xy(x,y,lim)
    selectx = x[indxy]
    selecty = y[indxy]
    selectmass = w[indxy]
    selectmassvel = w[indxy]*val[indxy]
    selectmassvelvel = w[indxy]*val[indxy]*val[indxy]

    mat_mass = hist2d_bincount(selectx,selecty,nXY,lim,selectmass)
    mat_massvel = hist2d_bincount(selectx,selecty,nXY,lim,selectmassvel)
    mat_massvsquare = hist2d_bincount(selectx,selecty,nXY,lim,selectmassvelvel)

    mask = (mat_mass != 0)
    mat_vel = np.zeros_like(mat_mass)
    mat_sig = np.zeros_like(mat_mass)
    mat_vel[mask] = mat_massvel[mask] / mat_mass[mask]
    mat_sig[mask] = np.sqrt(mat_massvsquare[mask] / mat_mass[mask] - mat_vel[mask] * mat_vel[mask])

    # Return : M, M*V, V, M*(V*V+S*S), S
    return (mat_mass,mat_massvel,mat_vel,mat_massvsquare,mat_sig)
############################################################
def comp_moments(x, y, val, weights=None,lim=[-1,1,-1,1],nXY=10):
    """
    Calculate the first 2 velocity moments
    x : x coordinate
    y : y coordinate
    val : velocity
    w : weight (e.g., mass) if None -> 1

    lim : tuple or array (xmin, xmax, ymin, ymax)
    nXY : number of bins in X and Y. If nXY is a single integer
          nX and nY will be set both to nXY
          Otherwise, nXY can be a set of 2 integers (tuple/array)
          which will then translate into nX=nXY[0] and nY=nXY[1]

    Return X, Y, M, V, S
    """
    if np.size(nXY) == 1 :
        nXY = np.zeros(2, dtype=np.int) + nXY
    elif np.size(nXY) != 2 :
        print "ERROR: dimension of n should be 1 or 2"
        return 0,0,0,0,0

    if weights is None : weights = np.ones_like(x)
    binX = np.linspace(lim[0], lim[1], nXY[0]+1)
    binY = np.linspace(lim[2], lim[3], nXY[1]+1)
    mat_mass = np.histogram2d(x, y, [binX, binY], weights=weights)[0]
    mat_massvel = np.histogram2d(x, y, [binX, binY], weights=weights * val)[0]
    mat_massvsquare = np.histogram2d(x, y, [binX, binY], weights=weights * val**2)[0]

    mask = (mat_mass != 0)
    mat_vel = np.zeros_like(mat_mass)
    mat_sig = np.zeros_like(mat_mass)
    mat_vel[mask] = mat_massvel[mask] / mat_mass[mask]
    mat_sig[mask] = np.sqrt(mat_massvsquare[mask] / mat_mass[mask] - mat_vel[mask] * mat_vel[mask])

    # Return : M, M*V, V, M*(V*V+S*S), S
    gridX, gridY = np.meshgrid(np.linspace(lim[0], lim[1], nXY[0]), np.linspace(lim[2], lim[3], nXY[1]))
    return (gridX, gridY, mat_mass, mat_vel, mat_sig)
############################################################
def comp_losvd(x, y, v, weights=None, limXY=[-1,1,-1,1], nXY=10, limV=[-1000,1000], nV=10):
    """
    Calculate the first 2 velocity moments
    x : x coordinate
    y : y coordinate
    v : velocity coordinate
    weights weight (e.g., mass)

    limXY : (xmin, xmax, ymin, ymax)
    nXY : number of bins in X and Y. If nXY is a single integer
          nX and nY will be set both to nXY
          Otherwise, nXY can be a set of 2 integers (tuple/array)
          which will then translate into nX=nXY[0] and nY=nXY[1]
    limV : (vmin, vmax)

    nV : number of bins for V

    Return a 3D array with grid of X, Y and V
    """
    if np.size(nXY) == 1 :
        nXY = np.zeros(2, dtype=np.int) + nXY
    elif np.size(nXY) != 2 :
        print "ERROR: dimension of n should be 1 or 2"
        return 0,0,0,0,0

    binX = np.linspace(limXY[0], limXY[1], nXY[0]+1)
    binY = np.linspace(limXY[2], limXY[3], nXY[1]+1)
    binV = np.linspace(limV[0], limV[1], nV+1)
    
    sizeX = np.size(x)
    sample = np.hstack((y.reshape(sizeX,1), x.reshape(sizeX,1), v.reshape(sizeX,1)))

    # Return LOSVD and edges
    return np.histogramdd(sample, bins=(binY,binX,binV), weights=weights)[0]
############################################################
