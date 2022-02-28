#! /usr/bin/python
# -*- coding: iso-8859-15 -*-
"""
This module allows to basic routines ot plot snapshots from simulations
"""

"""
Importing the most important modules
This module requires :
 - os
 - matplotlib version>=0.99.0
 - pylab
 - numpy
 - pyhist from pygme
"""

import os

import numpy as np

import matplotlib
import matplotlib.pyplot as mpl
from matplotlib.figure import Figure
from matplotlib.colors import Normalize

from .pyhist import hist2d_bincount

__version__ = '1.0.2 (25 December 2012)'
# Version 1.0.2 : - i -> inclin
# Version 1.0.0 : - Imported from the previous PMSPHSF version

############################################################
### rotation of positions and velocities according to 3 Euler angles
############################################################
def rotmatXYZ(pos, vel=None, alpha=0.0, beta=0.0, gamma=0.0, rotorder=[0,1,2], direct=1):   # Angles in degrees
    """
    Performs a rotation around the three axes in order given by rotorder
    for the position and velocities of particles.

    @param alpha: around X
    @param beta: around Y
    @param gamma: around Z
    @param direct: type of rotation
    @type  direct: 0 (return to initial configuration) or 1 (do the rotation)
    """
    alpha = alpha * np.pi / 180.
    beta = beta * np.pi / 180.
    gamma = gamma * np.pi / 180.

    calpha, salpha = np.cos(alpha), np.sin(alpha)
    cbeta, sbeta = np.cos(beta), np.sin(beta)
    cgamma, sgamma = np.cos(gamma), np.sin(gamma)

    if direct :
        matX=np.array([[1, 0, 0],[0, calpha, salpha],[0, -salpha, calpha]],dtype=np.float32)
        matY=np.array([[cbeta, 0, -sbeta],[0, 1, 0],[sbeta, 0, cbeta]],dtype=np.float32)
        matZ=np.array([[cgamma, sgamma, 0],[-sgamma, cgamma, 0],[0, 0, 1]],dtype=np.float32)
    else :
        matX=np.transpose(np.array([[1, 0, 0],[0, calpha, salpha],[0, -salpha, calpha]],dtype=np.float32))
        matY=np.transpose(np.array([[cbeta, 0, -sbeta],[0, 1, 0],[sbeta, 0, cbeta]],dtype=np.float32))
        matZ=np.transpose(np.array([[cgamma, sgamma, 0],[-sgamma, cgamma, 0],[0, 0, 1]],dtype=np.float32))

    list_mat = [matX, matY, matZ]
    mat = np.dot(np.dot(list_mat[rotorder[0]], list_mat[rotorder[1]]), list_mat[rotorder[2]])

    pos = np.transpose(np.dot(mat,np.transpose(pos)))
    if vel is not None:
        vel = np.transpose(np.dot(mat,np.transpose(vel)))

    return pos, vel

############################################################
### rotation of positions and velocities according to PA, i
############################################################
def rotmat(pos, vel=None, PA=0.0, inclin=0.0, direct=1):   # Angles in degrees
    """
    Performs a rotation of PA and i with angles given in degrees
    for the position and velocities of particles.

    Rotation with PA rotates around z
    And rotation with inclin rotates around X

    @param PA: Position Angle (in degrees)
    @type  PA: float
    @param inclin : Inclination (in degrees)
    @param direct: type of rotation
    @type  direct: 0 (return to initial configuration) or 1 (do the rotation)
    """
    PA = PA*np.pi/180.
    inclin = inclin*np.pi/180.
    cPA = np.cos(PA)
    sPA = np.sin(PA)
    ci = np.cos(inclin)
    si = np.sin(inclin)

    if direct :
        mat=np.array([[cPA,-sPA,0],[ci*sPA,ci*cPA,-si],[si*sPA,cPA*si,ci]],dtype=np.float32)
    else :
        mat=np.transpose(np.array([[cPA,-sPA,0],[ci*sPA,ci*cPA,-si],[si*sPA,cPA*si,ci]],dtype=np.float32))

    pos = np.transpose(np.dot(mat,np.transpose(pos)))
    if vel is not None:
        vel = np.transpose(np.dot(mat,np.transpose(vel)))

    return pos, vel

################################################################
#       CLASS PlotSnap: main class of this module
################################################################
class PlotSnap(object):
    # ##################################################
    #  INITIALISATION ##################################
    # ##################################################
    def __init__(self,*args):
        """define the snapshot to be plotted.
        @version: 1.0.0 (Nov 19, 2007)
        @author: Nicolas Champavert and Eric Emsellem
        """

        # gui: put gui=1 if you use the graphical interface pyberenice, else put gui=0
        self.gui = 0

###########################################################
###########################################################
###########################################################
    def init_figure(self,view,type='xyz'):
        """
        Make the initialisation of the plot

        @param view: select the view
        @type  view: string ('particles', 'density', 'contour', 'contourf' or 'gasprop')

        @param type: select the type (for 'xyz' view, we plot 3 different views -- xy, zy, xz -- and for 'kinematics' view, we plot 3 different physical quantities -- mass, velocity, velocity dispersion -- for the xy view)
        @type  type: string ('xyz', 'kinematics')

        """
        if(self.gui==1):
            self.f = Figure(figsize=(10,10),dpi=80)
        else:
            self.f = mpl.figure(figsize=(10,10),dpi=80)

        if(view=='particles'):
            self.p1 = self.f.add_axes((0.07,0.55,0.410,0.410),axisbg='k')
            self.p2 = self.f.add_axes((0.57,0.55,0.410,0.410),axisbg='k')
            self.p3 = self.f.add_axes((0.07,0.05,0.410,0.410),axisbg='k')
        else:
            self.p1 = self.f.add_axes((0.07,0.55,0.410,0.410),axisbg='w')
            self.p2 = self.f.add_axes((0.57,0.55,0.410,0.410),axisbg='w')
            self.p3 = self.f.add_axes((0.07,0.05,0.410,0.410),axisbg='w')

        self.p1.set_xlabel('x (kpc)')
        self.p1.set_ylabel('y (kpc)')
        if (type=='xyz'):
            self.p2.set_xlabel('z (kpc)')
            self.p2.set_ylabel('y (kpc)')
            self.p3.set_xlabel('x (kpc)')
            self.p3.set_ylabel('z (kpc)')
        elif(type=='kinematics'):
            self.p2.set_xlabel('x (kpc)')
            self.p2.set_ylabel('y (kpc)')
            self.p3.set_xlabel('x (kpc)')
            self.p3.set_ylabel('y (kpc)')

############################################################
############################################################
############################################################
    def rotastro(self, PA=0., inclin=90., type='n'):
        """
        Performs a counter-clockwised rotation of PA and i with angles given in degrees
        for the positions and velocities of particles.
        The rotation is always done starting with the initial configuration
        (galaxy is seen face-on). So rotastro(0,0) returns to the initial configuration.

        @param PA: Position Angle (in degrees)
        @type  PA: float
        @param inclin: Inclination (in degrees)
        @type  inclin: float
        @param type: select the type
        @type  type: 'n' (normal mode) or 'b' (position and velocities at birth)
        """
        if(type=='n') :
            if (PA != self.PA) or (inclin != self.inclin) :
                self.pos,self.vel = rotmat(self.pos, self.vel, self.PA, self.inclin, direct=0)
                self.pos,self.vel = rotmat(self.pos, self.vel, PA, inclin, direct=1)
                self.PA = PA
                self.inclin = inclin
        elif(type=='b') :
            if (PA != self.PAb) or (inclin != self.inclinb) :
                self.posb,self.velb = rotmat(self.posb, self.velb, self.PAb, self.inclinb, direct=0)
                self.posb,self.velb = rotmat(self.posb, self.velb, PA, inclin, direct=1)
                self.PAb = PA
                self.inclinb = inclin
        else:
            print('Wrong type for rotastro!')
            return

        print('rotation done!')
        return

###########################################################
###########################################################
###########################################################
    def calc_mats(self,lim,ind,n,weights='mass'):
        """
        Calculate the square matrices for plotting densities or contours.

        @param lim: limits for the selection of particles
        @type  lim: tuple (xmin,xmax,ymin,ymax,zmin,zmax)

        @param ind: indices of
        @type  ind:

        @param select: tuple of arrays of positions for selected particles
        @type  select: tuple of arrays

        @param w:
        @type  w:

        @param weights:
        @type  weights: string ('mass' or 'temp'; 'mass' is the default)

        @param n: size of the square matrix for density, contour or contourf view
        @type  n: integer

        @return: array of matrices of densities of particles in xy,zy,xz (mat0,mat1,mat2)
        @rtype: array
        """
        selectx0 = self.pos[ind[0],0]
        selecty0 = self.pos[ind[0],1]
        selectz1 = self.pos[ind[1],2]
        selecty1 = self.pos[ind[1],1]
        selectx2 = self.pos[ind[2],0]
        selectz2 = self.pos[ind[2],2]

        if(weights=='mass'):
            mat0 = hist2d_bincount(selectx0,selecty0,n,(lim[0],lim[1],lim[2],lim[3]),self.pmass[ind[0]])
            mat1 = hist2d_bincount(selectz1,selecty1,n,(lim[4],lim[5],lim[2],lim[3]),self.pmass[ind[1]])
            mat2 = hist2d_bincount(selectx2,selectz2,n,(lim[0],lim[1],lim[4],lim[5]),self.pmass[ind[2]])

            mat0 = mat0 * self.umass
            mat1 = mat1 * self.umass
            mat2 = mat2 * self.umass
        elif(weights=='temp'):
            # A modifier pour plotter la temperature au lieu de l'energie interne
            mat0 = hist2d_bincount(selectx0,selecty0,n,(lim[0],lim[1],lim[2],lim[3]),self.pmass[ind[0]]*self.u[ind[0]])
            mat1 = hist2d_bincount(selectz1,selecty1,n,(lim[4],lim[5],lim[2],lim[3]),self.pmass[ind[1]]*self.u[ind[1]])
            mat2 = hist2d_bincount(selectx2,selectz2,n,(lim[0],lim[1],lim[4],lim[5]),self.pmass[ind[2]]*self.u[ind[2]])

        return [mat0,mat1,mat2]
############################################################
############################################################
############################################################
    def calculate_select(self,lim,imin=0,imax=None):
        """
        Calculate the selection of particles for plotting

        @param lim: limits for the selection of particles
        @type  lim: tuple (xmin,xmax,ymin,ymax,zmin,zmax)

        @param imin: indice of the first particle for the selected population
        @type  imin: integer

        @param imax: indice of the first particle for the selected population
        @type  imax: integer

        @return: arrays of positions for selected particles and arrays of indices of selected particles in xy, zy and xz
        @rtype: tuple of 2 tuples (selects and inds which are tuple of arrays)
        """
        if imax == None :
            imax = self.pos.shape[0]

        ind = np.arange(imin,imax,dtype=np.int32)

          ## WARNING: np.nonzero() now returns a tuple with only one element (an array with the indices...)

        indxy = np.nonzero((lim[0]<=self.pos[imin:imax,0])&(self.pos[imin:imax,0]<lim[1]) & (lim[2]<=self.pos[imin:imax,1])&(self.pos[imin:imax,1]<lim[3]))[0]
        indzy = np.nonzero((lim[4]<=self.pos[imin:imax,2])&(self.pos[imin:imax,2]<lim[5]) & (lim[2]<=self.pos[imin:imax,1])&(self.pos[imin:imax,1]<lim[3]))[0]
        indxz = np.nonzero((lim[0]<=self.pos[imin:imax,0])&(self.pos[imin:imax,0]<lim[1]) & (lim[4]<=self.pos[imin:imax,2])&(self.pos[imin:imax,2]<lim[5]))[0]

        # we return the absolute indices of particles
        return (ind[indxy],ind[indzy],ind[indxz])

############################################################
############################################################
############################################################
    def plot_one_population(self,lim,p,ind,couleur):
        """
        Make the plot for one population

        @param lim: limits for the selection of particles
        @type  lim: tuple (xmin,xmax,ymin,ymax,zmin,zmax)
 calulate_select_kinematics
        @param p:
        @type  p: matplotlib.axes.Axes

        @param select: tuple of arrays of positions for selected particles
        @type  select: tuple of arrays

        @param couleur: color for plotting the particles
        @type  couleur: tuple

        """
        p[0].plot(self.pos[ind[0],0],self.pos[ind[0],1],',',mec=couleur,mfc=couleur)
        p[1].plot(self.pos[ind[1],2],self.pos[ind[1],1],',',mec=couleur,mfc=couleur)
        p[2].plot(self.pos[ind[2],0],self.pos[ind[2],2],',',mec=couleur,mfc=couleur)
        # mec = marker edge color
        # mfc = marker face color

        return

###########################################################
###########################################################
###########################################################
    def plot_scatter(self,p,ind,c,cmap,minc,maxc,size,l):
        """
        Make the scatter plot for one population

        @param p:
        @type  p: matplotlib.axes.Axes

        @param select: tuple of arrays of positions for selected particles
        @type  select: tuple of arrays

        @param c: tuple of arrays for plotting colors
        @type  c: tuple of arrays

        @param cmap: colormap used
        @type  cmap:

        @param couleur: color for plotting the particles
        @type  couleur: tuple

        @param size: size for scatter plot
        @type  size: float

        @param l: log of mass
        @type  l: 0 or 1
        """

        if(l==1):
            # we avoid c < 0. with adding 1
            c[0]=np.log10(c[0]+1)
            c[1]=np.log10(c[1]+1)
            c[2]=np.log10(c[2]+1)

        diff1 = c[0].max()-c[0].min()
        diff2 = c[1].max()-c[1].min()
        diff3 = c[2].max()-c[2].min()
        vmin1 = c[0].min()+minc*diff1
        vmax1 = c[0].min()+maxc*diff1
        vmin2 = c[1].min()+minc*diff2
        vmax2 = c[1].min()+maxc*diff2
        vmin3 = c[2].min()+minc*diff3
        vmax3 = c[2].min()+maxc*diff3


        p[0].scatter(self.pos[ind[0],0],self.pos[ind[0],1],c=c[0],s=size,cmap=cmap,faceted=False,vmin=vmin1,vmax=vmax1)
        p[1].scatter(self.pos[ind[1],2],self.pos[ind[1],1],c=c[1],s=size,cmap=cmap,faceted=False,vmin=vmin2,vmax=vmax2)
        p[2].scatter(self.pos[ind[2],0],self.pos[ind[2],2],c=c[2],s=size,cmap=cmap,faceted=False,vmin=vmin3,vmax=vmax3)

        return

###########################################################
###########################################################
###########################################################
    def plot(self,all=10,lim=(0.,0.,0.,0.,0.,0.),view='particles',l=1,gas=0,weights='mass',halo=0,bulge=0,disc=0,old=0,new=0,n=64,n_cont=10,min=0.,max=1.,cm=0,size=10., cmap=mpl.cm.jet):
        """
        Plot the particles...

        @param all: limits for the selection of particles in kpc. Equivalent to lim=(-all,all,-all,all,-all,all)
        @type  all: float (default is 10 kpc)

        @param lim: limits for the selection of particles in kpc (only active if all=0)
        @type  lim: tuple (xmin,xmax,ymin,ymax,zmin,zmax)

        @param view: select the view
        @type  view: string ('particles', 'density', 'contour', 'contourf' or 'scatter')

        @param l: log of density (only useful when view='density', 'contour' or 'contourf')
        @type  l: 0 or 1

        @param gas: select or not the gas particles
        @type  gas: 0 or 1

        @param weights: weights for the gas with 'density', 'contour' or 'contourf'
        @type  weights: string ('mass' or 'temp'; 'mass' is the default)

        @param halo: select or not the halo particles
        @type  halo: 0 or 1

        @param bulge: select or not the bulge particles
        @type  bulge: 0 or 1

        @param disc: select or not the disc particles
        @type  disc: 0 or 1

        @param old: select or not the old particles (equivalent to disc+bulge)
        @type  old: 0 or 1

        @param new: select or not the new particles
        @type  new: 0 or 1

        @param n: size of the square matrix for density or contour view
        @type  n: integer (default is 64)

        @param n_cont: number of contours (only used for contour view)
        @type  n_cont: integer (default is 10)

        @param min: set the minimum value for plotting values a to a.min()+min*(a.max()-a.min()) (only used if view!= particles)
        @type  min: float in [0,1] (default is 0.)

        @param max: set the maximum value for plotting values a to a.min()+max*(a.max()-a.min()) (only used if view!= particles)
        @type  max: float in [0,1] (default is 1.)

        @param cm: contour mass
        @type  cm: 0 or 1

        @param size: size for scatter plot
        @type  size: float (default is 10.)

        @return: figure to plot only if self.gui=1
        @rtype: matplotlib.figure.Figure
        """
        if (all!=0.):
            lim=(-all,all,-all,all,-all,all)

        self.init_figure(view)

        if(old==1):
            bulge=1
            disc=1

        ns = [0.,0.,0.,0.,0.]
        pops=(gas,disc,bulge,new,halo)
        couleur = ((0,1,0),(0,1,1),(1,1,0),(1,0,0),(1,0,1))

        if(view=='density' or view=='contour' or view=='contourf'):
            w = np.array([float(lim[1]-lim[0]),float(lim[3]-lim[2]),float(lim[5]-lim[4])])
            mat = np.array([np.zeros((n,n),np.float32),np.zeros((n,n),np.float32),np.zeros((n,n),np.float32)])
            indm = [np.array([],dtype=np.int32),np.array([],dtype=np.int32),np.array([],dtype=np.int32)]

        for i,popi in enumerate(pops):
            if(popi==1):
                ind = self.calculate_select(lim,self.imins[i],self.imaxs[i])
                ns[i] = len(ind[0])
                if(view=='particles'):
                    self.plot_one_population(lim,(self.p1,self.p2,self.p3),ind,couleur[i])
                elif(view=='scatter'):
                    self.plot_scatter((self.p1,self.p2,self.p3),ind,[self.pmass[ind[0]],self.pmass[ind[1]],self.pmass[ind[2]]],cmap,min,max,size=size,l=l)
                else:
                    indm[0] = np.concatenate((indm[0],ind[0]))
                    indm[1] = np.concatenate((indm[1],ind[1]))
                    indm[2] = np.concatenate((indm[2],ind[2]))

        if(view=='density' or view=='contour' or view=='contourf'):
            mat = self.calc_mats(lim,indm,n,weights)
            m0 = np.where(mat[0]==0,1,0)
            m1 = np.where(mat[1]==0,1,0)
            m2 = np.where(mat[2]==0,1,0)

            if(l==1):
                mat[0]=np.log10(mat[0]+1)
                mat[1]=np.log10(mat[1]+1)
                mat[2]=np.log10(mat[2]+1)

            if(cm==1):
                d=w/float(n)
                x = np.arange(lim[0]+d[0]/2,lim[1],d[0])
                y = np.arange(lim[2]+d[1]/2,lim[3],d[1])
                mappable=self.p1.contour(x,y,mat[0],n_cont,interpolation='nearest',colors='k')
                self.p2.contour(x,y,mat[1],n_cont,interpolation='nearest',colors='k')
                self.p3.contour(x,y,mat[2],n_cont,interpolation='nearest',colors='k')

            temparray = np.compress((1-m0).ravel(),mat[0].ravel())
            if len(temparray.ravel()) > 0 :
                min1 = temparray.min()
            else :
                min1 = 0.
            temparray = np.compress((1-m1).ravel(),mat[0].ravel())
            if len(temparray.ravel()) > 0 :
                min2 = temparray.min()
            else :
                min2 = 0.
            temparray = np.compress((1-m2).ravel(),mat[0].ravel())
            if len(temparray.ravel()) > 0 :
                min3 = temparray.min()
            else :
                min3 = 0.
            diff1 = mat[0].max()-min1
            diff2 = mat[1].max()-min2
            diff3 = mat[2].max()-min3
            vmin1 = min1+min*diff1
            vmax1 = min1+max*diff1
            vmin2 = min2+min*diff2
            vmax2 = min2+max*diff2
            vmin3 = min3+min*diff3
            vmax3 = min3+max*diff3

            if(view=='density'):
                mappable=self.p1.imshow(mat[0],interpolation='nearest',extent=(lim[0],lim[1],lim[2],lim[3]),origin='lower',norm=Normalize(vmin=vmin1,vmax=vmax1),cmap=cmap)
                self.p2.imshow(mat[1],interpolation='nearest',extent=(lim[4],lim[5],lim[2],lim[3]),origin='lower',norm=Normalize(vmin=vmin2,vmax=vmax2),cmap=cmap)
                self.p3.imshow(mat[2],interpolation='nearest',extent=(lim[0],lim[1],lim[4],lim[5]),origin='lower',norm=Normalize(vmin=vmin3,vmax=vmax3),cmap=cmap)
            else:
                d = w/float(n)
                x = np.arange(lim[0]+d[0]/2,lim[1],d[0])
                y = np.arange(lim[2]+d[1]/2,lim[3],d[1])
                z = np.arange(lim[4]+d[2]/2,lim[5],d[2])
                if(view=='contour'):
                    mappable=self.p1.contour(x,y,mat[0],n_cont,interpolation='nearest')#,norm=mpl.normalize(vmin=vmin1,vmax=vmax1))
                    self.p2.contour(z,y,mat[1],n_cont,interpolation='nearest')#,norm=mpl.normalize(vmin=vmin2,vmax=vmax2))
                    self.p3.contour(x,z,mat[2],n_cont,interpolation='nearest')#,norm=mpl.normalize(vmin=vmin3,vmax=vmax3))
                elif(view=='contourf'):
                    mappable=self.p1.contourf(x,y,mat[0],n_cont,interpolation='nearest')#,norm=mpl.normalize(vmin=vmin1,vmax=vmax1))
                    self.p2.contourf(z,y,mat[1],n_cont,interpolation='nearest')#,norm=mpl.normalize(vmin=vmin2,vmax=vmax2))
                    self.p3.contourf(x,z,mat[2],n_cont,interpolation='nearest')#,norm=mpl.normalize(vmin=vmin3,vmax=vmax3))
            cax=self.f.add_axes([0.50, 0.05, 0.05, 0.410])
            self.f.colorbar(mappable=mappable,cax=cax)


        self.f.text(0.6,0.1,' Time (Myr): '+str(self.t*self.utime/1.e+6)+\
               '\n\n PA = '+str(self.PA)+'  i = '+str(self.inclin)+\
               '\n\n Nb part. total = '+str(self.ntot)+ \
               '\n\n Nb part. gas = '+str(self.ngas)+ \
               '\n\n Nb part. stellaires (t=0) = '+str(self.nsta0)+ \
               '\n\n Nb part. bulbe (t=0) = '+str(self.nbulge)+ \
               '\n\n Nb part. disque (t=0) = '+str(self.ndisc)+ \
               '\n\n Nb part. total stellaire = '+str(self.nsta)+ \
               '\n\n Nb part. stellaires new = '+str(self.nform)+ \
               '\n\n Nb part. halo = '+str(self.nhalo)+ \
               '\n\n SELECTION (XY) :'+ \
               '\n\n Nb part. total = '+str( int(sum(ns)) )+ \
               '\n\n Nb part. gas (t=0) = '+str(int(ns[0]) )+ \
               '\n\n Nb part. bulbe (t=0) = '+str(int(ns[2]))+ \
               '\n\n Nb part. disque (t=0) = '+str(int(ns[1]))+ \
               '\n\n Nb part. stellaires new = '+str(int(ns[3]))+ \
               '\n\n Nb part. halo = '+str(int(ns[4]))+ \
               '\n\n'+ \
               '\n\n Namerun = '+str(self.namerun))

        self.p1.axis([lim[0],lim[1],lim[2],lim[3]])
        self.p2.axis([lim[4],lim[5],lim[2],lim[3]])
        self.p3.axis([lim[0],lim[1],lim[4],lim[5]])

        if(self.gui==1):
            return self.f
        else:
            mpl.show()
            mpl.draw()
            if(view=='density' or view=='contour' or view=='contourf'):
                return  mat

###########################################################
###########################################################
###########################################################
    def plot_gas(self,all=10.,lim=(0.,0.,0.,0.,0.,0.),size='',color='u',cmap=mpl.cm.jet,ls=0,lc=1,minc=0.,maxc=1.,k=20.):
        """
        Plot the mass density, the velocity and the velocity dispersion for the gas

        @param all: limits for the selection of particles in kpc. Equivalent to lim=(-all,all,-all,all,-all,all)
        @type  all: float (default is 10 kpc)

        @param lim: limits for the selection of particles in kpc (only active if all=0)
        @type  lim: tuple (xmin,xmax) ((ymin,ymax) = (xmin,xmax))

        @param size: variable to use for the size of symbols ('' means constant size, 'h' means we use sph size)
        @type  size: string ('' or 'h')

        @param color: variable to use for the color of symbols
        @type  color: string ('', 'u', 'P', 'rho' or 'm')

        @param cmap: colormap to use (only used if color!='')
        @type  cmap: matplotlib colormap

        @param ls: log of values used for the size of symbols
        @type  ls: 0 or 1

        @param lc: log of values used for the color of symbols
        @type  lc: 0 or 1

        @param minc: set the minimum value for plotting values c to c.min()+min*(c.max()-c.min()) (only used if color!='')
        @type  minc: float in [0,1] (default is 0.)

        @param maxc: set the maximum value for plotting values c to c.min()+max*(c.max()-c.min()) (only used if color!='')
        @type  maxc: float in [0,1] (default is 1.)

        @param k: size for plotting gas particles
        @type  k: float (default is 20.)

        @return: figure to plot only if gui=1
        @rtype: matplotlib.figure.Figure
        """
        if(all!=0.):
            lim=(-all,all,-all,all,-all,all)

        ind = self.calculate_select(lim,self.iming,self.imaxg)
#       (select,ind) = self.calculate_select(lim,self.iming,self.imaxg)
        self.init_figure('gasprop')
        (f,p1,p2,p3) = (self.f,self.p1,self.p2,self.p3)

        indgas = np.arange(self.iming,self.imaxg,dtype=np.int32)

        if(color==''):
            c1='g'
            c2='g'
            c3='g'
            vmin1 = 0.
            vmax1 = 1.
            vmin2 = 0.
            vmax2 = 1.
            vmin3 = 0.
            vmax3 = 1.
        else:
            if(color=='u'):
                c1=self.u[indgas[ind[0]]]
                c2=self.u[indgas[ind[1]]]
                c3=self.u[indgas[ind[2]]]
            elif(color=='P'):
                c1=self.pression[indgas[ind[0]]]
                c2=self.pression[indgas[ind[1]]]
                c3=self.pression[indgas[ind[2]]]
            elif(color=='rho'):
                c1=self.rho[ind[indgas[0]]]
                c2=self.rho[ind[indgas[1]]]
                c3=self.rho[ind[indgas[2]]]
            elif(color=='m'):
                c1=self.pmass[indgas[ind[0]]]
                c2=self.pmass[indgas[ind[1]]]
                c3=self.pmass[indgas[ind[2]]]
            if(lc==1):
                c1=np.log10(c1)
                c2=np.log10(c2)
                c3=np.log10(c3)

        if(size==''):
            s1=1.
            s2=1.
            s3=1.
        elif(size=='h'):
            s1=self.h[indgas[ind[0]]]
            s2=self.h[indgas[ind[1]]]
            s3=self.h[indgas[ind[2]]]


        if(ls==1):
            # we add 1 in order to have sizes > 0. or else some particles will not be plotted (particles with np.log10(s) < 0.)
            s1=np.log10(s1+1)
            s2=np.log10(s2+1)
            s3=np.log10(s3+1)

        s1=k*s1
        s2=k*s2
        s3=k*s3

        if(color!=''):
            diff1 = c1.max()-c1.min()
            diff2 = c2.max()-c2.min()
            diff3 = c3.max()-c3.min()
            vmin1 = c1.min()+minc*diff1
            vmax1 = c1.min()+maxc*diff1
            vmin2 = c2.min()+minc*diff2
            vmax2 = c2.min()+maxc*diff2
            vmin3 = c3.min()+minc*diff3
            vmax3 = c3.min()+maxc*diff3

        mappable=p1.scatter(self.pos[ind[0],0],self.pos[ind[0],1],s=s1,c=c1,cmap=cmap,faceted=False,vmin=vmin1,vmax=vmax1)
        p2.scatter(self.pos[ind[1],2],self.pos[ind[1],1],s=s2,c=c2,cmap=cmap,faceted=False,vmin=vmin2,vmax=vmax2)
        p3.scatter(self.pos[ind[2],0],self.pos[ind[2],2],s=s3,c=c3,cmap=cmap,faceted=False,vmin=vmin3,vmax=vmax3)

        f.text(0.6,0.1,' Time (Myr): '+str(self.t*self.utime/1.e+6)+\
               '\n\n PA = '+str(self.PA)+'  i = '+str(self.inclin)+\
               '\n\n Nb part. total = '+str(self.ntot)+ \
               '\n\n Nb part. gas = '+str(self.ngas)+ \
               '\n\n Nb part. stellaires (t=0) = '+str(self.nsta0)+ \
               '\n\n Nb part. bulbe (t=0) = '+str(self.nbulge)+ \
               '\n\n Nb part. disque (t=0) = '+str(self.ndisc)+ \
               '\n\n Nb part. total stellaire = '+str(self.nsta)+ \
               '\n\n Nb part. stellaires new = '+str(self.nform)+ \
               '\n\n Nb part. halo = '+str(self.nhalo)+ \
               '\n\n SELECTION (XY) :'+ \
               '\n\n Nb part. gas (t=0) = '+str(len(ind[0]) )+ \
               '\n\n'+ \
               '\n\n Namerun = '+str(self.namerun))

        p1.axis([lim[0],lim[1],lim[2],lim[3]])
        p2.axis([lim[4],lim[5],lim[2],lim[3]])
        p3.axis([lim[0],lim[1],lim[4],lim[5]])

        if(color!=''):
            cax=f.add_axes([0.50, 0.05, 0.05, 0.410])
            f.colorbar(mappable=mappable,cax=cax)

        if(self.gui==1):
            return f
        else:
#           if(color!=''):
#               cax=mpl.axes([0.55, 0.05, 0.05, 0.410])
#               f.colorbar(mappable=mappable,cax=cax)
            mpl.show()
            mpl.draw()
            return

###########################################################
###########################################################
###########################################################
    def calculate_select_kinematics(self,lim,imin=0,imax=None,indxy=None):
        """
        Calculate the selection of particles for plotting

        @param lim: limits for the selection of particles
        @type  lim: tuple (xmin,xmax,ymin,ymax,zmin,zmax)

        @param imin: indice of the first particle for the selected population
        @type  imin: integer

        @param imax: indice of the last particle for the selected population
        @type  imax: integer

        @return: arrays of positions of selected particles and arrays of indices of selected particles in xy, zy and xz
        @rtype: arrays
        """
        if indxy == None:
            indxy = np.array([],dtype=np.int32)

        if imax == None :
            imax = self.pos.shape[0]

        ind = np.arange(imin,imax,dtype=np.int32)

        ## WARNING: np.nonzero now returns a tuple with only one element (an array with the indices...)
        indxy = np.concatenate((indxy,ind[np.nonzero((lim[0]<self.pos[imin:imax,0])&\
                                                       (self.pos[imin:imax,0]<lim[1]) &\
                                                       (lim[2]<self.pos[imin:imax,1])&\
                                                       (self.pos[imin:imax,1]<lim[3]))[0]]))
        return indxy

###########################################################
###########################################################
###########################################################
    def calc_mats_kinematics(self,lim,indxy,n):
        """
        Calculate the square matrices for plotting densities or contours.

        @param lim: limits for the selection of particles
        @type  lim: tuple (xmin,xmax,ymin,ymax,zmin,zmax)

        @param indxy: indices of selected particles
        @type  indxy: array

        @param n: size of the square matrix for density or contour view
        @type  n: integer

        @return: matrices of densities of particles in xy,zy,xz (mat0,mat1,mat2)
        @rtype: tuple
        """
#       t1=os.times()[4]
        selectx = self.pos[indxy,0]
        selecty = self.pos[indxy,1]
        selectmass = self.pmass[indxy]
        selectmassvel = self.pmass[indxy]*self.vel[indxy,2]
        selectmassvelvel = self.pmass[indxy]*self.vel[indxy,2]*self.vel[indxy,2]

#       t2=os.times()[4]
#       print 'time calc_mats_kinematics selection',t2-t1

        mat_mass = hist2d_bincount(selectx,selecty,n,lim,selectmass)
        mat_massvel = hist2d_bincount(selectx,selecty,n,lim,selectmassvel)
        mat_massvsquare = hist2d_bincount(selectx,selecty,n,lim,selectmassvelvel)

#       print 'time calc_mats_kinematics',os.times()[4]-t2

        return (mat_mass,mat_massvel,mat_massvsquare)

###########################################################
###########################################################
###########################################################
    def plot_kinematics(self,all=10,lim=(0.,0.,0.,0.),type='gas',fft=1,nbinh=3,halo=0,bulge=0,disc=0,old=1,new=0,view='density',l=1,n=64,n_cont=10,min=0.,max=1.,cm=0,sort=4, cmap=mpl.cm.jet):
        """
        Plot the mass density, the velocity and the velocity dispersion for the gas or the stars
        For the gas, we take into account the sph kernel if fft=1.
        (problem on the left side of the velocity and velocity dispersion field)

        @param all: limits for the selection of particles in kpc. Equivalent to lim=(-all,all,-all,all)
        @type  all: float (default is 10 kpc)

        @param lim: limits for the selection of particles in kpc (only active if all=0)
        @type  lim: tuple (xmin,xmax,ymin,ymax) ((ymax-ymin) = (xmax-xmin))

        @param type: select the population to plot (gas or stars)
        @type  type: string ('gas' or 'stellar')

        @param fft: if selected, we take into account the sph size of gas particles with fft
        @type  fft: 0 or 1 (default is 1)

        @param nbinh: number of bins in h (sph-size of gas particles)
        @type  nbinh: integer (default is 3)

        @param halo: select or not the halo particles
        @type  halo: 0 or 1

        @param bulge: select or not the bulge particles
        @type  bulge: 0 or 1

        @param disc: select or not the disc particles
        @type  disc: 0 or 1

        @param old: select or not the old particles (equivalent to disc+bulge)
        @type  old: 0 or 1 (default is 1)

        @param new: select or not the new particles
        @type  new: 0 or 1

        @param view: select the view
        @type  view: string ('density', 'contour' or 'contourf')

        @param l: log of density (only useful when view='density', 'contour' or 'contourf')
        @type  l: 0 or 1

        @param n: size of the square matrix for density or contour view
        @type  n: integer (default is 64)

        @param n_cont: number of contours (only used for contour view)
        @type  n_cont: integer (default is 10)

        @param min: set the minimum value for plotting values c to c.min()+min*(c.max()-c.min())
        @type  min: float in [0,1] (default is 0.)

        @param max: set the maximum value for plotting values c to c.min()+max*(c.max()-c.min())
        @type  max: float in [0,1] (default is 1.)

        @param cm: contour mass
        @type  cm: 0 or 1

        @return: figure to plot only if gui=1
        @rtype: matplotlib.figure.Figure
        """
        ns=[0.,0.,0.,0.,0.]
        if (all!=0.):
            lim=(-all,all,-all,all)

        w = np.array([float(lim[1]-lim[0]),float(lim[3]-lim[2])],dtype=np.float32)
        indxy = np.array([],dtype=np.int32)

        if(type=='gas'):
            indxy = self.calculate_select_kinematics(lim,self.iming,self.imaxg,indxy)
            ns[0]=len(indxy)

            if (fft==0):
                (matm,matv,matvs)=self.calc_mats_kinematics(lim,indxy,n)
            elif (fft==1):
                if(w[0]!=w[1]):
                    print('(lim[1]-lim[0]) must be equal to (lim[3]-lim[2]) for the gas if fft=1')

                dbinsize = float(n)/w[0]

                h2 = np.log10(self.h)
                delta_h2 = ((h2[indxy]).max()-(h2[indxy]).min())/float(nbinh)

                hsize = np.zeros((nbinh))
                image = np.zeros((2*n,2*n))
                imagev = np.zeros((2*n,2*n))
                imagevsquare = np.zeros((2*n,2*n))
                matm = np.zeros((n,n),np.float32)
                matv = np.zeros((n,n),np.float32)
                matvs = np.zeros((n,n),np.float32)
                densitybis = np.zeros((n,n),np.float32)

                ###########################
                ## Boucle sur les bins en h
                ###########################
                for k in range(nbinh):
                    h_min = (h2[indxy]).min() + k*delta_h2
                    h_max = h_min + delta_h2
                    #indxy contient les indices des particules selectionnees selon x et y
                    select = indxy[np.nonzero((h2[indxy]>=h_min) & (h2[indxy]<h_max))[0]]
                    if(len(select)>0): # we continue only if there are particles in the selection in h

                        (density,densityv,densityvsquare)=self.calc_mats_kinematics(lim,select,n)

                        image[0:n,0:n] = density
                        imagev[0:n,0:n] = densityv
                        imagevsquare[0:n,0:n] = densityvsquare

                        hsize[k] = np.mean(self.h[select]) * dbinsize

                        #################
                        ## Green function
                        #################
                        (i,j) = np.indices((n,n),dtype=np.int32)
#                       i=np.reshape(np.repeat(np.arange(n),n),(n,n))
#                       j=np.transpose(np.reshape(np.repeat(np.arange(n),n),(n,n)))
                        v=np.sqrt(i*i+j*j)/hsize[k]
                        mask1 = np.where(v<=1,1,0)
                        mask2 = np.where((v>1)&(v<=2),1,0)
                        green = np.zeros((2*n,2*n))
                        green[0:n,0:n] = ( mask1*(1.-1.5*(v*v)+0.75*(v*v*v)) + mask2*(0.25*(2.-v)*(2.-v)*(2.-v)) )/(np.pi*(hsize[k])*(hsize[k])*(hsize[k]))
##                       green[0:n,0:n] = ( mask1*(1.-1.5*(v**2.)+0.75*(v**3.)) + mask2*(0.25*(2.-v)**3) )/(np.pi*(hsize[k])**3.)
                        ngg = green.sum()
                        green[0:n,n:2*n]=green[0:n,n:0:-1]
                        green[n:2*n,0:2*n]=green[n:0:-1,0:2*n]

                        ######
                        ## fft
                        ######
                        fftmass = np.fft.fft2(image)
                        fftmassv = np.fft.fft2(imagev)
                        fftmassvsquare = np.fft.fft2(imagevsquare)
                        fftgreen = np.fft.fft2(green)

                        fftmass = fftmass*fftgreen
                        fftmassv = fftmassv*fftgreen
                        fftmassvsquare = fftmassvsquare*fftgreen

                        density2 = np.fft.ifft2(fftmass)[0:n,0:n]
                        densityv2 = np.fft.ifft2(fftmassv)[0:n,0:n]
                        densityvsquare2 = np.fft.ifft2(fftmassvsquare)[0:n,0:n]

                        res = sum(sum(density.real))/ngg
                        resv = sum(sum(densityv.real))/ngg
                        resvsquare = sum(sum(densityvsquare.real))/ngg

                        matm = matm + density2.real/ngg
                        matv = matv + densityv2.real/ngg
                        matvs = matvs + densityvsquare2.real/ngg
                    ##fin if (len(select)>0)
                ##fin boucle k
            ##fin if fft

        elif(type=='stellar'):

            if (halo==1):
                indxy=self.calculate_select_kinematics(lim,self.iminh,self.imaxh,indxy)
                ns[4]=len(indxy)

            if (bulge==1 or old==1):
                indxy=self.calculate_select_kinematics(lim,self.iminb,self.imaxb,indxy)
                ns[2]=len(indxy)


            if (disc==1 or old==1):
                indxy=self.calculate_select_kinematics(lim,self.imind,self.imaxd,indxy)
                ns[1]=len(indxy)

            if (new==1):
                indxy=self.calculate_select_kinematics(lim,self.iminn,self.imaxn,indxy)
                ns[3]=len(indxy)

            (matm,matv,matvs)=self.calc_mats_kinematics(lim,indxy,n)

        ################
        ## Make the plot
        ################
        mask = np.where(matm>0.,1,0)
        # matv contains m*v and matvs contains m*v^2
        np.putmask(matv, mask, matv/matm)
        np.putmask(matvs, mask, np.sqrt(matvs/matm - (matv*matv))) # velocity dispersion sigma = sqrt(<v^2> - <v>^2)
        matm = matm * self.umass #to have physical unit (solar masses)

        self.init_figure(view,type='kinematics')
        (f,p1,p2,p3) = (self.f,self.p1,self.p2,self.p3)

        m0 = np.where(matm==0,1,0)
        # if there is no mass, we put zero for velocity and velocity dispersion
        # because there are nan from self.calc_mats_kinematics for velocity and
        # velocity dispersion if there are no particles
        np.putmask(matv,m0,0.)
        np.putmask(matvs,m0,0.)

        m2 = np.where(matvs==0,1,0)
        if(l==1):
            matm=np.log10(matm+1)
            matvs=np.log10(matvs+1)

        if(cm==1):
            d=w/float(n)
            x = np.arange(lim[0]+d[0]/2,lim[1],d[0])
            y = np.arange(lim[2]+d[1]/2,lim[3],d[1])
            p1.contour(x,y,matm,n_cont,interpolation='nearest',colors='k')
            p2.contour(x,y,matm,n_cont,interpolation='nearest',colors='k')
            p3.contour(x,y,matm,n_cont,interpolation='nearest',colors='k')

        if(view=='density'):
            if(min==0 and max==0):
                mappable=p1.imshow(matm,interpolation='nearest',extent=(lim[0],lim[1],lim[0],lim[1]),origin='lower',vmin=np.compress((1-m0).ravel(),matm.ravel()).min(), cmap=cmap)
                p2.imshow(matv,interpolation='nearest',extent=(lim[0],lim[1],lim[0],lim[1]),origin='lower', cmap=cmap) #,vmin=-absmax,vmax=absmax)
                p3.imshow(matvs,interpolation='nearest',extent=(lim[0],lim[1],lim[0],lim[1]),origin='lower',vmin=np.compress((1-m2).ravel(),matvs.ravel()).min(), cmap=cmap)
            else:
                if(min==0.):
                    min=1.
                if(max==0.):
                    max=1.
                mappable=p1.imshow(matm,interpolation='nearest',extent=(lim[0],lim[1],lim[0],lim[1]),origin='lower',vmin=min*np.compress((1-m0).ravel(),matm.ravel()).min(),vmax=max*(matm.ravel()).max(), cmap=cmap)
                p2.imshow(matv,interpolation='nearest',extent=(lim[0],lim[1],lim[0],lim[1]),origin='lower', cmap=cmap) #,vmin=-absmax,vmax=absmax)
                p3.imshow(matvs,interpolation='nearest',extent=(lim[0],lim[1],lim[0],lim[1]),origin='lower',vmin=min*np.compress((1-m2).ravel(),matvs.ravel()).min(),vmax=max*(matvs.ravel()).max(), cmap=cmap)

        elif(view=='contour' or view=='contourf'):
            d = w/float(n)
            x = np.arange(lim[0]+d[0]/2,lim[1],d[0])
            y = np.arange(lim[2]+d[1]/2,lim[3],d[1])

            if(view=='contour'):
                mappable=p1.contour(x,y,matm,n_cont,interpolation='nearest') #,colors='r')
                p2.contour(x,y,matv,n_cont,interpolation='nearest')
                p3.contour(x,y,matvs,n_cont,interpolation='nearest')
            elif(view=='contourf'):
                minp1 = min*np.compress((1-m0).ravel(),matm.ravel()).min()
                maxp1 = max*(matm.ravel()).max()
                mappable=p1.contourf(x,y,matm,n_cont,interpolation='nearest') #,colors='r')
                p2.contourf(x,y,matv,n_cont,interpolation='nearest')
                p3.contourf(x,y,matvs,n_cont,interpolation='nearest')

        f.text(0.6,0.1,' Time (Myr): '+str(self.t*self.utime/1.e+6)+\
               '\n PA = '+str(self.PA)+'  i = '+str(self.inclin)+\
               '\n Nb part. total = '+str(self.ntot)+ \
               '\n Nb part. gas = '+str(self.ngas)+ \
               '\n Nb part. stellaires (t=0) = '+str(self.nsta0)+ \
               '\n Nb part. bulbe (t=0) = '+str(self.nbulge)+ \
               '\n Nb part. disque (t=0) = '+str(self.ndisc)+ \
               '\n Nb part. total stellaire = '+str(self.nsta)+ \
               '\n Nb part. stellaires new = '+str(self.nform)+ \
               '\n Nb part. halo = '+str(self.nhalo)+ \
               '\n SELECTION (XY) :'+ \
               '\n Nb part. total = '+str( int(sum(ns)) )+ \
               '\n Nb part. gas (t=0) = '+str(int(ns[0]) )+ \
               '\n Nb part. bulbe (t=0) = '+str(int(ns[2]))+ \
               '\n Nb part. disque (t=0) = '+str(int(ns[1]))+ \
               '\n Nb part. stellaires new = '+str(int(ns[3]))+ \
               '\n Nb part. halo = '+str(int(ns[4]))+ \
               '\n'+ \
               '\n Namerun = '+str(self.namerun))

        p1.axis([lim[0],lim[1],lim[2],lim[3]])
        p2.axis([lim[0],lim[1],lim[2],lim[3]])
        p3.axis([lim[0],lim[1],lim[2],lim[3]])

        cax=f.add_axes([0.50, 0.05, 0.05, 0.410])
        f.colorbar(mappable=mappable,cax=cax)

        if(self.gui==1):
            return f
        else:
            mpl.show()
            mpl.draw()
            return matm, matv, matvs

############################################################
############################################################
############################################################
    def plot_abundances(self,type='gas',lim=(0.,10.),ref='H',n=20,gui=0):
        """
        plot radial abundances for gas or stars

        """

        mpl.interactive(False)
        XHsol = np.array([1.00000000e+00, 3.88083113e-01, 4.32674920e-03,\
                           1.55929284e-03, 1.35106235e-02, 9.16807548e-04,\
                           9.88718013e-04,   5.15973436e-04,   2.59159331e-03])

        if(type=='gas'):
            imin = self.iming
            imax = self.imaxg
        elif(type=='ns'):
            imin = self.iminn
            imax = self.imaxn

        r = np.sqrt(self.pos[imin:imax,0]**2+self.pos[imin:imax,1]**2)
        ind = np.arange(imin,imax,dtype=np.int32)
        #        ## WARNING: np.nonzero() now returns a tuple with only one element (an array with the indices...)
        indr = np.nonzero((lim[0]<=r[:]) & (r[:]<=lim[1]))[0]
        selectr = r[indr]

        if(ref=='H'):
            k=0
        elif(ref=='Fe'):
            k=8
        else:
            print("ref must be set to 'H' or 'Fe'")
            return

#       for i in np.arange(self.nb_elts):
#           yields_norm[:,i] = self.yields[:,i]/self.yields[:,k]

        if(gui==1):
            f = Figure(figsize=(10,10),dpi=80)
        else:
            f = mpl.figure(figsize=(15,15),dpi=80)


        delta = (lim[1]-lim[0])/n
        indices = np.array((selectr-lim[0])/delta,dtype=np.int32)
        mask = np.where(indices == n)[0]
        indices[mask]=n-1 # dernier bin ferme a droite [ ]
        # we add one "ghost particle" with weight=0 at the end
        indices = np.concatenate((indices,[n-1])) #

        norm = self.umass*self.pmass[indr[:]+imin]*self.yields[indr[:]+imin,k].ravel()
        norm = np.concatenate((norm,[0.]))
        print('np.bincount(indices)',np.bincount(indices))
        hist_norm = np.bincount(indices,norm)
#       print 'hist_norm',hist_norm[:]

        titles=('[H/H]','[He/H]','[C/H]','[N/H]','[O/H]','[Mg/H]','[Si/H]','[S/H]','[Fe/H]')
        #        print 'lim[0]+np.arange(n)*delta',lim[0]+np.arange(n)*delta

        f.text(0.45,0.95,str(self.namerun.rstrip())+' '+'t='+str(np.int(self.t*self.utime/1.e6))+' Myr')



        for i in np.arange(self.nb_elts):
            print('i',i)
            weights = self.umass*self.pmass[indr[:]+imin]*self.yields[indr[:]+imin,i].ravel()
            weights = np.concatenate((weights,[0.]))
            histind = np.bincount(indices,weights)
            mpl.subplot(3,3,i+1)
#           print 'hist',hist[:]
#           print 'log10(hist/(hist_norm*XHsol))',np.log10(hist[:]/(hist_norm[:]*XHsol[i]))
#           print '(hist/(hist_norm*XHsol))',(hist[:]/(hist_norm[:]*XHsol[i]))
#           print 'len(hist)',len(hist)
#           histo = np.log10(hist[:-1]/hist_norm[:-1])
            histo = np.log10(histind[:]/(hist_norm[:]*XHsol[i]))
            #            histo = (histind[:]/(hist_norm[:]*XHsol[i]))
#           histo[np.isinf(histo)==True]=-8
#           print 'len(histo)',len(histo)
#           print 'len(lim[0]+np.arange(n)*delta)',len(lim[0]+np.arange(n)*delta)
            #            mpl.bar(np.arange(n),histo[:],(lim[1]-lim[0])/n,color='w',edgecolor='k')
            mpl.plot((lim[0]+np.arange(n)*delta),histo[:],linestyle='steps')
            #            mpl.axis([lim[0],lim[1],histo.min(),histo.max()])
            if(i>=2):
                mpl.axis([lim[0],lim[1],-4.,0.])
            mpl.title(titles[i])
            mpl.xlabel('r (kpc)')

        # for tests
        #        mpl.figure()
        #        mpl.plot(self.pos[indr[:]+imin,0],self.pos[indr[:]+imin,1],',')

        if(gui==1):
            return f
        else:
            mpl.show()
            mpl.draw()
            return
###########################################################
###########################################################
    def plot_abundances_maps(self,type='gas',lim=(-10.,10.,-10.,10.),n=10,l=1,gui=0, cmap=mpl.cm.jet):
        """
        plot maps of abundances
        """
        mpl.interactive(False)

        # H He C N O Mg Si S Fe
        XHsol = np.array([1.00000000e+00, 3.88083113e-01, 4.32674920e-03,\
                           1.55929284e-03, 1.35106235e-02, 9.16807548e-04,\
                           9.88718013e-04,   5.15973436e-04,   2.59159331e-03])

        if(type=='gas'):
            imin = self.iming
            imax = self.imaxg
        elif(type=='ns'):
            imin = self.iminn
            imax = self.imaxn


        if(gui==1):
            f = Figure(figsize=(10,10),dpi=80)
        else:
            f = mpl.figure(figsize=(15,15),dpi=80)

        ind = np.arange(imin,imax,dtype=np.int32)
        ## WARNING: np.nonzero() now returns a tuple with only one element (an array with the indices...)
        indxy = np.nonzero((lim[0]<=self.pos[imin:imax,0])&(self.pos[imin:imax,0]<lim[1]) & (lim[2]<=self.pos[imin:imax,1])&(self.pos[imin:imax,1]<lim[3]))[0]
        selected_ind = ind[indxy]
#       print 'selected_ind',selected_ind

        selectx = self.pos[selected_ind,0]
        selecty = self.pos[selected_ind,1]

        mats = list(range(self.nb_elts))
        mats[0] = hist2d_bincount(selectx,selecty,n,(lim[0],lim[1],lim[2],lim[3]),self.pmass[selected_ind]*self.yields[selected_ind,0])
#       mats[0] = hist2d_bincount(selectx,selecty,n,(lim[0],lim[1],lim[2],lim[3]),self.pmass[selected_ind])
        print('before log min max mat[0]',mats[0].min(),mats[0].max())
#       print 'mats[0]',mats[0]*self.umass


        for i in range(1,self.nb_elts):
            mats[i] = hist2d_bincount(selectx,selecty,n,(lim[0],lim[1],lim[2],lim[3]),self.pmass[selected_ind]*self.yields[selected_ind,i])
            print('before log min max mat[',i,']',mats[i].min(),mats[i].max())

            if(l==1):
                mats[i] = mats[i]/(mats[0]*XHsol[i])
                mats[i] = np.log10(mats[i])
            if(l==2):
                mats[i] = mats[i]/(mats[0])
                mats[i] = 12+np.log10(mats[i])
            print('min max mat[',i,']',mats[i].min(),mats[i].max())

        m0 = np.where(mats[0]==0,1,0)
#       m0 = np.where(mats[0]!=0)[0]
        if(l==1):
            mats[0] = np.log10(mats[0])

        min0 = np.compress((1-m0).ravel(),mats[0].ravel()).min()
        diff0 = mats[0].max()-min0
        vmin0 = min0+0.*diff0
        vmax0 = min0+1.*diff0
        #        vmin0 = min0+min*diff0
        #        vmax0 = min0+max*diff0
        if(l==1):
            print('after log min max mat[0]',min0,mats[0].max())


        p1 = mpl.subplot(331)
        mappable = p1.imshow(mats[0],interpolation='nearest',extent=(lim[0],lim[1],lim[0],lim[1]),origin='lower',norm=Normalize(vmin=vmin0,vmax=vmax0), cmap=cmap)
        f.colorbar(mappable=mappable)


        p2 = mpl.subplot(332)
        mappable = p2.imshow(mats[1],interpolation='nearest',extent=(lim[0],lim[1],lim[0],lim[1]),origin='lower', cmap=cmap)
        f.colorbar(mappable=mappable)

        p3 = mpl.subplot(333)
        mappable = p3.imshow(mats[2],interpolation='nearest',extent=(lim[0],lim[1],lim[0],lim[1]),origin='lower', cmap=cmap)
        f.colorbar(mappable=mappable)

        p4 = mpl.subplot(334)
        mappable = p4.imshow(mats[3],interpolation='nearest',extent=(lim[0],lim[1],lim[0],lim[1]),origin='lower', cmap=cmap)
        f.colorbar(mappable=mappable)

        p5 = mpl.subplot(335)
        mappable = p5.imshow(mats[4],interpolation='nearest',extent=(lim[0],lim[1],lim[0],lim[1]),origin='lower', cmap=cmap)
        f.colorbar(mappable=mappable)

        p6 = mpl.subplot(336)
        mappable = p6.imshow(mats[5],interpolation='nearest',extent=(lim[0],lim[1],lim[0],lim[1]),origin='lower', cmap=cmap)
        f.colorbar(mappable=mappable)

        p7 = mpl.subplot(337)
        mappable = p7.imshow(mats[6],interpolation='nearest',extent=(lim[0],lim[1],lim[0],lim[1]),origin='lower', cmap=cmap)
        f.colorbar(mappable=mappable)

        p8 = mpl.subplot(338)
        mappable = p8.imshow(mats[7],interpolation='nearest',extent=(lim[0],lim[1],lim[0],lim[1]),origin='lower', cmap=cmap)
        f.colorbar(mappable=mappable)

        p9 = mpl.subplot(339)
        mappable = p9.imshow(mats[8],interpolation='nearest',extent=(lim[0],lim[1],lim[0],lim[1]),origin='lower', cmap=cmap)
        f.colorbar(mappable=mappable)


        if(gui==1):
            return f
        else:
            mpl.show()
            mpl.draw()
            return

###########################################################
###########################################################
###########################################################
    def evoldyn(self,tlim=9999.,old=0, plot=0):
        """
        from evoldyn.pro
        """
        #            ; old = relecture des anciens formats de fichier ascii unite 500
        #            PRO evoldyn,tlim=tlim,old=old,plot=plot           ;,rlim=rlim
        #        COMMON unites,umass, udist, udens, utime, uergg, uergcc, uvitess
        #        @commons_chemo

        #        units_pmsphsf
        udist=1000. #(cf. sph de pmsphsf)
        self.arch = 0
        #if not keyword_set(tlim) then tlim=9999.
        #;if not keyword_set(rlim) then rlim=40.

        if old:
            simufile = open('fort.500','r')
        else:
            simufile = open('fort.499','rb')

        i=int(0)
        t=0. ; q=0. ; tdyn=0. ; tdyncloud=0. ; tmjeans=0. ; tmjeans_cloud=0. ; mass=0.
        pmasstot=0. ; hma=0. ; rayon=0. ; z=0. ; volume=0. ; rhocloud=0. ; surface=0. ; tmucloud=0. ; ucloud=0. ; ilen=0.

        if old:
            #                datatemp = np.fromfile(file,dtype=np.float32,count=8,sep=' ')
            i = np.fromfile(file,dtype=np.int32,count=1)
            [t,q,tdyn,tdynckoud,tmjeans,tmjeans_cold,tmjeans_cloud] = np.fromfile(file,dtype=np.float32,count=7,sep=' ')
            #                i = datatemp[0]
            #                t = datatemp[1]
            #                q = datatemp[2]
            #                tdyn = datatemp[3]
            #               tdyncloud = datatemp[4]
            #                tmjeans = datatemp[5]
            #                tmjeans_cold = datatemp[6]
            #                tmjeans_cloud = datatemp[7]
        else:
            print('one')
            numbersdata = [1,17]
            typedata = [np.int32,np.float32]
            status, [b1, i,datatemp, b2] = read_for_fast(simufile, numbers=numbersdata, type=typedata, arch=self.arch)

            print('datatemp',datatemp)

            t = datatemp[0]
            q = datatemp[1]
            tdyn = datatemp[2]
            tdyncloud = datatemp[3]
            tmjeans = datatemp[4]
            tmjeans_cold = datatemp[5]
            tmjeans_cloud = datatemp[6]
            pmasstot = datatemp[7]
            hma = datatemp[8]
            rayon = datatemp[9]
            z = datatemp[10]
            volume = datatemp[11]
            rhocloud = datatemp[12]
            surface = datatemp[13]
            tmucloud = datatemp[14]
            ucloud = datatemp[15]
            ilen = datatemp[16]
            status, [b1, [i,t,q,tdyn,tdynckoud,tmjeans,tmjeans_cold,tmjeans_cloud,pmasstot,hma,rayon,z,volume,rhocloud,surface,tmucloud,ucloud,ilen], b2] = read_for_fast(simufile, numbers=numbersdata, type=typedata, arch=self.arch)


        tpart = [t]
        qpart = [q]
        tdynpart = [tdyn]
        tdyncloudpart = [tdyncloud]
        mjeanspart = [tmjeans]
        mjeanscold = [tmjeans_cold]
        mjeanscloud = [tmjeans_cloud]

        if old:
            while(t <= tlim):
                i = np.fromfile(file,dtype=np.int32,count=1,sep=' ')
                datatemp = np.fromfile(file,dtype=np.float32,count=7,sep=' ')
                if(datatemp==[]):
                    break
                tpart.append(datatemp[0])
                qpart.append(datatemp[1])
                tdynpart.append(datatemp[2])
                tdyncloudpart.append(datatemp[3])
                mjeanspart.append(datatemp[4])
                mjeanscold.append(datatemp[5])
                mjeanscloud.append(datatemp[6])
        else:
            while(t <= tlim):
                try:
                    numbersdata = [1,17]
                    typedata = [np.int32,np.float32]
                    status, [b1, i,datatemp, b2] = read_for_fast(simufile, numbers=numbersdata, type=typedata, arch=self.arch)
                    #                print 'status',status

##                   i = datatemp[0]
##                   t = datatemp[1]
##                   q = datatemp[2]
##                   tdyn = datatemp[3]
##                   tdyncloud = datatemp[4]
##                   tmjeans = datatemp[5]
##                   tmjeans_cold = datatemp[6]
##                   tmjeans_cloud = datatemp[7]
##                   pmasstot = datatemp[8]
##                   hma = datatemp[9]
##                   rayon = datatemp[10]
##                   z = datatemp[11]
##                   volume = datatemp[12]
##                   rhocloud = datatemp[13]
##                   surface = datatemp[14]
##                   tmucloud = datatemp[15]
##                   ucloud = datatemp[16]
##                   ilen = datatemp[17]
                    #                status, [b1, [i,t,q,tdyn,tdynckoud,tmjeans,tmjeans_cold,tmjeans_cloud,pmasstot,hma,rayon,z,volume,rhocloud,surface,tmucloud,ucloud,ilen], b2] = read_for_fast(simufile, numbers=numbersdata, type=typedata, arch=self.arch)
                    t = datatemp[0]
                    tpart.append(t)
                    qpart.append(datatemp[1])
                    tdynpart.append(datatemp[2])
                    tdyncloudpart.append(datatemp[3])
                    mjeanspart.append(datatemp[4])
                    mjeanscold.append(datatemp[5])
                    mjeanscloud.append(datatemp[6])

                except ValueError:
                    break

        print('after reading file...')

        tlim=tpart[-1]

        tpart = np.array(tpart,dtype=np.float32)
        qpart = np.array(qpart,dtype=np.float32)
        tdynpart = np.array(tdynpart,dtype=np.float32)
        tdyncloudpart = np.array(tdyncloudpart,dtype=np.float32)
        mjeanspart = np.array(mjeanspart,dtype=np.float32)
        mjeanscold = np.array(mjeanscold,dtype=np.float32)
        mjeanscloud = np.array(mjeanscloud,dtype=np.float32)


        #
        # proprietes de la particule au moment de son eligibilite
        #

        tmp = os.path.abspath('.')
        vtitle = 'fort.499'+tmp[tmp.rfind('/'):]

        #            device,decomposed=0
        #            if !d.window ne id_dyn[0] then window,id_dyn[0],retain=2,xsize=1200,ysize=800,title=wtitle


        #            !p.multi=[0,3,2]
        #            !p.charsize=2.5
        #            ;!x.charsize=1.5 & !y.charsize=1.5
        #            tek_color

        print('np.log10(tlim)',np.log10(tlim))
        print('np.log10(2000.)',np.log10(2000.))

        print('min max tpart)',tpart.min(),tpart.max())

        print('tlim',tlim)

        if(np.log10(tlim) >= np.log10(2000.)):
            xtitle = 'T (Gyr)'
            tpart = tpart*self.utime/1.e9
        else:
            xtitle = 'T (Myr)'
            tpart=tpart*self.utime/1.e6

        print('min max tpart)',tpart.min(),tpart.max())

        #        matplotlib.rcParams['axes.facecolor']='w'
        mpl.interactive(False)

        ########
        # colors,....
        ########
        mec = (0,1,0) #green
        dotcolor = 'w'
        axisbg = 'k'
        s=40
        marker='^'
        linewidth = 1.

        mpl.figure(figsize=(15,10))


        p1 = mpl.subplot(231,axisbg=axisbg)
        mpl.semilogy(tpart,qpart,',',color=dotcolor)
        # ',' : pixels
        # '.' : points
        xlim = p1.get_xlim()
        mpl.title('Q')
        mpl.xlabel(xtitle)

        idx = np.where(qpart > 1.4)[0]
        iidx = 0
        if(idx.size != 0):
            iidx = 1

        # ###############
        iidx=0.
        # ###############
#      print 'min(qpart),max(qpart)',min(qpart),max(qpart)
        print('au dessus de 1.4=', idx.size,'sur',len(qpart))


        if(iidx):
            #                semilogy(tpart[idx],qpart[idx],'^','g')
#          mpl.plot(tpart[idx],qpart[idx],'^',mfc=mfc,mec=mec)
            mpl.scatter(tpart[idx],qpart[idx],marker=marker,s=s,alpha=0.,linewidth=linewidth,edgecolor=mec)
            #            semilogy([tpart.min(),tpart.max()],[1.4,1.4],'r')
        mpl.plot([tpart.min(),tpart.max()],[1.4,1.4],'r')

        n = 100
        delta = (tpart.max()-tpart.min())/(n-1.)
        indices = np.array((tpart-tpart.min())/delta, dtype=np.int32)

        qdum = np.bincount(indices)
        qhist = np.bincount(indices,qpart)
        mask = np.where(qdum != 0)[0]
        qhist[mask] = qhist[mask]/qdum[mask]

        tdum = tpart.min() + np.arange(n)*delta #(tpart.max()-tpart.min())/(n-1.)


##       # ##################
##       print 'before...'
##       tdumbis = [tdum[0]]
##       delta = tdum[1]-tdum[0]
##       for i in (tdum[1:]):
##           tdumbis.append(i)
##           tdumbis.append(i)
##       tdumbis.append(tdum[-1]+delta)
##       tdumbis = np.array(tdumbis,dtype=np.float32)
##       #        tdumbis[1:] = tdumbis[1:] - delta/2.

##       qhistbis=[]
##       for i in qhist[:]:
##           qhistbis.append(i)
##           qhistbis.append(i)
##       qhistbis = np.array(qhistbis,dtype=np.float32)

##       mpl.semilogy(tdumbis,qhistbis,color=(0,1,1),linewidth=2)# histogram mode in IDL...
##       # ###########################


        mpl.semilogy(tdum,qhist,linestyle='steps',color=(0,1,1),linewidth=2)
        p1.set_xlim(xlim)

        print('plot 2...')
        p2 = mpl.subplot(232,axisbg=axisbg)
        mpl.semilogy(tpart,self.utime*tdynpart/1.e6,',',color=dotcolor)
        mpl.title('Tdyn (Myr) particles')
        mpl.xlabel(xtitle)
        xlim = p2.get_xlim()
        ylim = p2.get_ylim()
        if(iidx != 0):
#          mpl.plot(tpart[idx],self.utime*tdynpart[idx]/1.e6,'^',mfc=mfc,mec=mec)
            mpl.scatter(tpart[idx],self.utime*tdynpart[idx]/1.e6,marker=marker,s=s,alpha=0.,linewidth=linewidth,edgecolor=mec)
        p2.set_xlim(xlim)
        p2.set_ylim(ylim)


        print('plot 3...')
        p3 = mpl.subplot(233,axisbg=axisbg)
        mpl.semilogy(tpart,self.utime*tdyncloudpart/1.e6,',',color=dotcolor)
        mpl.title('Tdyn (Myr) cloud')
        mpl.xlabel(xtitle)
        xlim = p3.get_xlim()
        ylim = p3.get_ylim()
        if(iidx != 0):
#          mpl.plot(tpart[idx],self.utime*tdyncloudpart[idx]/1.e6,'^',mfc=mfc,mec=mec)
            mpl.scatter(tpart[idx],self.utime*tdyncloudpart[idx]/1.e6,marker=marker,s=s,alpha=0.,linewidth=linewidth,edgecolor=mec)
        p3.set_xlim(xlim)
        p3.set_ylim(ylim)


        print('plot 4...')
        p4 = mpl.subplot(234,axisbg=axisbg)
        mpl.plot(tpart,self.umass*mjeanspart,',',color=dotcolor)
        mpl.title('Mass Jeans tot')
        mpl.xlabel(xtitle)
        xlim = p4.get_xlim()
        ylim = p4.get_ylim()
        if(iidx != 0):
#          mpl.plot(tpart[idx],self.umass*mjeanspart[idx],'^',mfc=mfc,mec=mec)
            mpl.scatter(tpart[idx],self.umass*mjeanspart[idx],marker=marker,s=s,alpha=0.,linewidth=linewidth,edgecolor=mec)
        p4.set_xlim(xlim)
        p4.set_ylim(ylim)


        print('plot 5...')
        p5 = mpl.subplot(235,axisbg=axisbg)
        mpl.plot(tpart,self.umass*mjeanscold,',',color=dotcolor)
        mpl.title('Mass Jeans cold')
        mpl.xlabel(xtitle)
        xlim = p5.get_xlim()
        if(iidx != 0):
#          mpl.plot(tpart[idx],self.umass*mjeanscold[idx],'^',mfc=mfc,mec=mec)
            mpl.scatter(tpart[idx],self.umass*mjeanscold[idx],marker=marker,s=s,alpha=0.,linewidth=linewidth,edgecolor=mec)
        p5.set_ylim(ylim)
        p5.set_xlim(xlim)


        print('plot 6...')
        p6 = mpl.subplot(236,axisbg=axisbg)
        mpl.plot(tpart,self.umass*mjeanscloud,',',color=dotcolor)
        mpl.title('Mass Jeans GMC')
        mpl.xlabel(xtitle)
        xlim = p6.get_xlim()
        ylim = p6.get_ylim()
        if(iidx != 0):
#          mpl.plot(tpart[idx],self.umass*mjeanscloud[idx],'^',mfc=mfc,mec=mec)
            mpl.scatter(tpart[idx],self.umass*mjeanscloud[idx],marker=marker,s=s,alpha=0.,linewidth=linewidth,edgecolor=mec)
        p6.set_xlim(xlim)
        p6.set_ylim(ylim)

        if(plot==1):
            mpl.savefig('evoldyn.png',dpi=80)

        mpl.show()
        mpl.draw()
        return

#########################################
    def plot_sfr(self,all=20,lim=(0.,0.,0.,0.,0.,0.),nbin=10):
        """
        plot sfr
        """

        if(all!=0):
            lim=(-all,all,-all,all,-all,all)

        print('toto')
        ind=self.calculate_select(lim,self.iminn,self.imaxn)[0]
        print('ind',ind)
        tborn_select = self.tborn[ind]
        pmass_select = self.pmass[ind]
        print('before delta')


##        delta = (tborn_select.max()-tborn_select.min())/nbin
##        indices = np.array((tborn_select[:]-tborn_select.min())/delta, dtype=np.int32)
##        print 'after indices...'
##        mask = np.where(indices == nbin)
##        indices[mask] = nbin-1
##        # le dernier bin est ferme a gauche et a droite [ ],
##        # alors que les autres sont ferme a gauche et ouvert a droite [ [
##        bornes = tborn_select.min() + np.arange(nbin)*delta
##        sfr = np.bincount(indices,pmass_select)

        (sfr,bornes) = hist1d(tborn_select[:],nbin,pmass_select[:])
        delta = bornes[1]-bornes[0]
        print('bornes[0]',bornes[0])

        bornes = bornes * self.utime/1.e6 # en Myr
        print('after bornes...')

        sfr = sfr * self.umass/(delta*self.utime) # en Msol/an

        mpl.figure()
        #        matplotlib.rcParams['text.usetex'] = True
        #        mpl.ylabel('SFR (M$_{\odot}$/yr)')
        mpl.ylabel('SFR (Msol/yr)')
        mpl.xlabel('Time (Myr)')
        mpl.plot(bornes,sfr,linestyle='steps')

        mpl.show()
        mpl.draw()

        return
