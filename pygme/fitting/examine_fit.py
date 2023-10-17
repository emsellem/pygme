######################################################################################
# examine_fit.py
# V.1.0.0 - 12 Mai 2014 - Emsellem 
#           Tuning the plots a bit, turning off the legend option
# V.0.1.0 - 2010 - Emsellem
#           Reshuffling and packaging from old python MGE routines
######################################################################################
"""
This module allows to examine the results from a MGE fit for 1D and 2D functions
"""

"""
Importing the most important modules
This module requires :
 - matplotlib version>=0.99.0
 - pylab
 - numpy
 - pyhist from pygme
"""

import os
import numpy as np
from numpy import log10

import matplotlib
from matplotlib import pyplot as plt
from pygme.mge_miscfunctions import convert_xy_to_polar
from pygme.binning.voronoibinning import derive_unbinned_field
from pygme.colormaps.mycolormap import get_SauronCmap

__version__ = '1.0.1 (21 August 2013)'
#__version__ = '1.0.0 (27 June 2012)'
# Version 1.0.1 : - Adding legend
# Version 1.0.0 : - Creation

def _plot_1dfit(ax1, ax2, x, data, fit, signx=1, xmin=None, xmax=None, labely=True, labelx=False, legend=False) :
    """ Hidden function (_) which is used to plot residuals of an MGE fit

    :param ax1, ax2: the two axes which correspond to the panels (Data versus fit and Data-fit in %)
    :param x : positions
    :param data : data points
    :param fit : fit of the data points
    :param signx : if only one side (positive) is present (default to 1)
    :param xmin, xmax : min and max for the plot in x
    :param labelx, labely : labelling in X and Y

    """
    from matplotlib.ticker import FormatStrFormatter
    sel_nonzero = (x > 0.)
    x_nz = x[sel_nonzero]
    d_nz = data[sel_nonzero]
    f_nz = fit[sel_nonzero]

    ## Sorting the arrays according to x
    arg_S = np.argsort(x_nz)
    if signx < 0 :
        x_nz = -x_nz[arg_S]
    else :
        x_nz = x_nz[arg_S]
    d_nz = d_nz[arg_S]
    f_nz = f_nz[arg_S]

    if xmin is None : xmin = np.min(x_nz)
    if xmax is None : xmax = np.max(x_nz)
    if signx < 0. :
        tmp = xmin
        xmin = xmax
        xmax = tmp

    ## Plotting the results using matplotlib
    ax1.scatter(x_nz, log10(d_nz), c='k', marker='o', s=3, label="Data")
    ax1.plot(x_nz, log10(f_nz), 'r-', label="Fit")
    ax2.plot(x_nz, 100.0 * (d_nz - f_nz) / d_nz, 'b-')
    ax2.hlines([0], xmin, xmax, linestyle='dashed')
    if labelx :
        ax1.set_xlabel("log10(x)")
        ax2.set_xlabel("log10(x)")
    else :
        xticklabels = ax1.get_xticklabels() + ax2.get_xticklabels()
        plt.setp(xticklabels, visible=False)
    if labely :
        ax1.set_ylabel("Data, fit")
        ax2.set_ylabel("Data-Fit [\%]")

    formatter = FormatStrFormatter('%3.1f')
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax1.set_xlim(xmin, xmax)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(-20,20)
    ax1.xaxis.set_major_formatter(formatter) 
    ax2.xaxis.set_major_formatter(formatter) 
    if legend :
        ax1.legend()

def plot_1dfit_residuals(x, data, fit, nfig=None, legend=False) :
    """ Create a figure with the residuals and fit

    :param x: input xaxis coordinates - array
    :param data: input data points (should have the same dim as x)
    :param fit: fitted points (should have the same dim as x)

    """

    if nfig is None:
        fig = plt.gcf()
    else :
        fig = plt.figure(nfig)
    fig.clf()

    ## Making the arrays as 1D
    x_rav = x.ravel()
    d_rav = data.ravel()
    f_rav = fit.ravel()

    lx = len(x_rav)
    ld = len(d_rav)
    lf = len(f_rav)

    ## checking that the dimensions are correct
    if (lx != ld) or (lx != lf) :
        print("ERROR: dimensions for x, data, and fit are not the same")
        print(" (respectively: %d %d and %d)"%(lx, ld, lf))
        return

    xmin = np.min(np.abs(x_rav))
    xmax = np.max(np.abs(x_rav))
    ## Plotting the results using matplotlib
    ## if all points are positive (or negative), just one side
    if np.alltrue(x_rav >= 0.) or np.alltrue(x_rav <= 0.) :
#        f, (ax1, ax2) = plt.subplots(1,2, sharex=True)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, sharex=ax1)
        _plot_1dfit(ax1, ax2, x_rav, d_rav, f_rav, labelx=True, legend=legend)
    else :
    ## Otherwise with 2 panels
#        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, sharex=True)
        selP = (x_rav >= 0.)
        selN = (x_rav < 0.)
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3, sharex=ax1)
        ax4 = fig.add_subplot(2, 2, 4, sharex=ax2)
        _plot_1dfit(ax1, ax2, x_rav[selP], d_rav[selP], f_rav[selP], xmin=xmin, xmax=xmax, labelx=False, legend=legend)
        _plot_1dfit(ax3, ax4, x_rav[selN], d_rav[selN], f_rav[selN], xmin=xmin, xmax=xmax, labelx=True, signx=-1, legend=legend)
    plt.subplots_adjust(hspace=0, bottom=0.08, right=0.98, left=0.1, top=0.98, wspace=0.3)

def show_2dfit_residuals(x, y, data, fit, xfield=None, yfield=None, nfig=None, cmap=None) :
    """ Create a figure with a map of the residuals in %

    :param x: input xaxis coordinates - array
    :param y: input yaxis coordinates - array (should have the same dim as x)
    :param data: input data points (should have the same dim as x)
    :param fit: fitted points (should have the same dim as x)
    :param xfield: 2d rectangular array for the field to show (xaxis)
    :param yfield: 2d rectangular array (yaxis)
    :param nfig: if None, will use the existing figure, otherwise will use that number

    :returns: Nothing

    """

    if nfig is None:
        fig = plt.gcf()
    else :
        fig = plt.figure(nfig)
    fig.clf()

    ## Making the arrays as 1D
    x_rav = x.ravel()
    y_rav = y.ravel()
    d_rav = data.ravel()
    f_rav = fit.ravel()

    lx = len(x_rav)
    ly = len(y_rav)
    ld = len(d_rav)
    lf = len(f_rav)

    ## checking that the dimensions are correct
    if (lx != ld) or (lx != lf) or (lx != ly) :
        print("ERROR: dimensions for x, y, data, and fit are not the same")
        print(" (respectively: %d %d %d and %d)"%(lx, ly, ld, lf))
        return

    unbinned_residuals = derive_unbinned_field(x, y, 100.0 * (data-fit) / fit, xfield, yfield)
    Sauron_Cmap = get_SauronCmap()
    plt.imshow(unbinned_residuals, vmin=-20., vmax=20., cmap=Sauron_Cmap)
    plt.colorbar()

def plot_2dfit_residuals(x, y, data, fit, PAmin=0., PAmax=360., nSectors=8, WedgeFactor=1., nfig=None,
        legend=False) :
    """ Create a figure with the residuals and fit

    :param x: input xaxis coordinates - array
    :param y: input yaxis coordinates - array (should have the same dim as x)
    :param data: input data points (should have the same dim as x)
    :param fit: fitted points (should have the same dim as x)

    :returns: Nothing

    """

    if nfig is None:
        fig = plt.gcf()
    else :
        fig = plt.figure(nfig)
    fig.clf()

    ## Making the arrays as 1D
    x_rav = x.ravel()
    y_rav = y.ravel()
    d_rav = data.ravel()
    f_rav = fit.ravel()

    lx = len(x_rav)
    ly = len(y_rav)
    ld = len(d_rav)
    lf = len(f_rav)

    ## checking that the dimensions are correct
    if (lx != ld) or (lx != lf) or (lx != ly) :
        print("ERROR: dimensions for x, y, data, and fit are not the same")
        print(" (respectively: %d %d %d and %d)"%(lx, ly, ld, lf))
        return

    ## Polar coordinates
    r, theta = convert_xy_to_polar(x_rav, y_rav)
    theta = np.rad2deg(theta)

    ## Selecting the points with respect to their sectors
    ## And sorting them out
    Sample_Theta = np.linspace(PAmin, PAmax, nSectors+1)
    Step_Theta = Sample_Theta[1] - Sample_Theta[0]
    Min_Theta = Sample_Theta[:-1] - Step_Theta / WedgeFactor
    Max_Theta = Sample_Theta[:-1] + Step_Theta / WedgeFactor

    Sel_Theta = []
    for i in range(nSectors) :
        newsel = np.argwhere((theta >= Min_Theta[i]) & (theta < Max_Theta[i]))
        Sel_Theta.append(newsel)

    xmin = np.min(np.abs(r[r > 0.]))
    xmax = np.max(np.abs(r))
    plt.ioff()
    ## Plotting the results using matplotlib
    ax01 = fig.add_subplot(nSectors, 2, 1)
    ax02 = fig.add_subplot(nSectors, 2, 2)
    _plot_1dfit(ax01, ax02, r[Sel_Theta[0]], d_rav[Sel_Theta[0]], f_rav[Sel_Theta[0]], xmin=xmin, xmax=xmax, labelx=False, legend=legend)
    for i in range(1, nSectors-1) :
        ax1 = fig.add_subplot(nSectors, 2, 2*i+1, sharex=ax01)
        ax2 = fig.add_subplot(nSectors, 2, 2*i+2, sharex=ax02)
        _plot_1dfit(ax1, ax2, r[Sel_Theta[i]], d_rav[Sel_Theta[i]], f_rav[Sel_Theta[i]], xmin=xmin, xmax=xmax, labelx=False, legend=legend)
    ax1 = fig.add_subplot(nSectors, 2, 2*nSectors-1, sharex=ax01)
    ax2 = fig.add_subplot(nSectors, 2, 2*nSectors, sharex=ax02)
    _plot_1dfit(ax1, ax2, r[Sel_Theta[i]], d_rav[Sel_Theta[i]], f_rav[Sel_Theta[i]], xmin=xmin, xmax=xmax, labelx=True, legend=legend)
    plt.ion()
    plt.subplots_adjust(hspace=0, bottom=0.08, right=0.98, left=0.1, top=0.98, wspace=0.3)

def contour_2dfit(x, y, data, fit=None, par=None, nfig=None, interp=True) :
    """ Create a figure with the fit shown as contours

    :param x: input xaxis coordinates - array
    :param y: input yaxis coordinates - array (should have the same dim as x)
    :param data: input data points (should have the same dim as x)
    :param interp: interpolation used or not (Default is True)
    :param par: parameters of the fit - Default is None. Only used if interpolation (interp=True) is used
    :param fit: fitted points (should have the same dim as x)

    :returns: Nothing

    """
    from pygme.binning.voronoibinning import derive_unbinned_field, guess_regular_grid
    from matplotlib.mlab import griddata
    from pygme.mgefunctions import convert_xy_to_polar
    from pygme.fitting.fitn2dgauss_mpfit import n_centred_twodgaussian_I

    if nfig is None:
        fig = plt.gcf()
    else :
        fig = plt.figure(nfig)
    fig.clf()

    ## If interpolation is requested 
    if interp:
        if fit is None : 
            print("ERROR: you did not provide 'fit' data")
            return
        xu, yu = guess_regular_grid(x, y)
        du = griddata(x, y, data, xu, yu, interp='nn')
        ## if par is provided, then we compute the real MGE function
        if par is None :
            fu = griddata(x, y, fit, xu, yu, interp='nn')
        else :
            r, t = convert_xy_to_polar(xu, yu)
            fu = n_centred_twodgaussian_I(pars=par)(r, t)
            fu = fu.reshape(xu.shape)
    ## Otherwise we just use the nearest neighbour
    else :
        xu, yu, du = derive_unbinned_field(x, y, data)
        xu, yu, fu = derive_unbinned_field(x, y, fit)

    CS = plt.contour(xu, yu, np.log10(du), colors='k', label='Data')
    CSfit = plt.contour(xu, yu, np.log10(fu), levels=CS.levels, colors='r', label='MGE Fit')
    plt.axes().set_aspect('equal')
    plt.legend()
