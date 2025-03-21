# Modules:
import woma
import swiftsimio as sw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm_mpl
import matplotlib
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D

# Use LaTeX fonts in plots:
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex = True)

import seaborn as sns
import os
import sys
import unyt
import h5py
import copy
import imageio
import re

import requests
from IPython.display import IFrame
import ipywidgets as widgets

import colormaps as local_cmaps
from copy import deepcopy

from scipy.signal import find_peaks
from scipy.interpolate import griddata
from scipy.optimize import minimize
from scipy.optimize import fsolve

# Constants:
R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg m^-3 
G = 6.67408e-11  # m^3 kg^-1 s^-2

woma.load_eos_tables(A1_mat_input = ["ANEOS_iron",
                                     "ANEOS_forsterite",
                                     "SS08_water",
                                     "HM80_HHe",
                                     "HM80_HHe_extended",
                                     "CD21_HHe",
                                     "custom_0"]) # Loads the eos table for the calculation of entropy at later stage

##############################################################################
# FUNC_PLOT_GREAT_CIRCLE:
def func_plot_great_circle(phitest,xc,yc,zc,Rt,Rp,offset,zsign,flag_plot_rot=1):

    b=(Rt**2-Rp**2-offset**2)/(2*offset)
    #find the new tangent
    test=calc_tangent(phitest,0.0,Rp,offset,b,zsign)

    if test[0]>0:
        theta_testy=zsign*-1*(np.pi-np.arctan(np.abs(test[2]/test[0])))
    else:
        theta_testy=zsign*-1*np.arctan(np.abs(test[2]/test[0]))
    # if test[0]>0:
    #     theta_testy=-(np.pi-np.arctan(np.abs(test[2]/test[0])))
    # else:
    #     theta_testy=-np.arctan(np.abs(test[2]/test[0]))
    # print(theta_testy)

    # print('angles',theta_testy)
    
    test_rot=rotatey(test[0],test[1],test[2],theta_testy)
    test_rot1=rotatey(test[0],test[1],test[2],theta_testy)

    if test_rot[1]>0:
        theta_testz=-np.arctan(np.abs(test_rot[0]/test_rot[1]))
    else:
        theta_testz=-np.pi/2-np.arctan(np.abs(test_rot[1]/test_rot[0]))
    
    test_rot=rotatez(test_rot[0],test_rot[1],test_rot[2],theta_testz)
    # print(test_rot)

    # print('angles',theta_testy,theta_testz)
    
    #find the new tangent
    t=np.linspace(-5*zsign,0,2)
    temp=calc_tangent(phitest,t,Rp,offset,b,zsign)
    temp=rotatey(temp[0],temp[1],temp[2],theta_testy)
    temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
    
    # if ((temp[2][0]>0)&(temp[0][0]<0))|((temp[2][0]<0)&(temp[0][0]>0)):
    #     theta_testy2=-np.arctan(np.abs(temp[2][0]/temp[0][0]))
    # else:
    #     theta_testy2=np.arctan(np.abs(temp[2][0]/temp[0][0]))
    if ((temp[2][0]>0)&(temp[0][0]<0))|((temp[2][0]<0)&(temp[0][0]>0)):
        theta_testy2=-np.arctan(np.abs(temp[2][0]/temp[0][0]))
    else:
        if zsign==-1:
            theta_testy2=-(np.pi-np.arctan(np.abs(temp[2][0]/temp[0][0])))
        else:
            theta_testy2=np.arctan(np.abs(temp[2][0]/temp[0][0]))
    # print(theta_testy2)

    if flag_plot_rot==1:
        test_rot=rotatey(test_rot[0],test_rot[1],test_rot[2],theta_testy2)
    else:
        test_rot=calc_tangent(phitest,0.0,Rp,offset,b,zsign)
    # print('test_rot',test_rot)

    # print('angles',theta_testy,theta_testz,theta_testy2)
    
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(111, projection='3d')
    
    # Outline the intersecting sphere and cylinder in light grey circles.
    if flag_plot_rot==1:
        ng = Nplot//2 + 1
        # We only need to go from 0 to 2π for circles!
        t = np.linspace(0, 4*np.pi, Nplot)
        theta = t[:ng]
        for phi in np.linspace(-np.pi,np.pi,32):
            # Circles on the sphere in two perpendicular planes.
            temp=rotatey(Rt*np.sin(theta)*np.cos(phi), Rt*np.cos (theta)*np.cos(phi),\
                Rt*np.sin(phi)*np.ones(ng),theta_testy)
            temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
            temp=rotatey(temp[0],temp[1],temp[2],theta_testy2)
            ax.plot(temp[0],temp[1],temp[2], 'gray', alpha=0.2, lw=1)
            temp=rotatey(Rt*np.cos(theta)*np.cos(phi), Rt*np.sin(phi)*np.ones(ng),
                Rt*np.sin(theta)*np.cos(phi), theta_testy)
            temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
            temp=rotatey(temp[0],temp[1],temp[2],theta_testy2)
            ax.plot(temp[0],temp[1],temp[2], 'gray', alpha=0.2,lw=1)
            
        for phi in np.linspace(-np.pi,np.pi,120):
            # The circles of the cylinder.
            temp=rotatey(Rp*np.sin(theta)-offset,Rp*np.cos(theta),2*Rt*phi*np.ones(np.size(theta))/np.pi, theta_testy)
            temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
            temp=rotatey(temp[0],temp[1],temp[2],theta_testy2)
            ax.plot( temp[0],temp[1],temp[2], 
                    'm', alpha=0.2, lw=1)

    else:

        ng = Nplot//2 + 1
        # We only need to go from 0 to 2π for circles!
        t = np.linspace(0, 4*np.pi, Nplot)
        theta = t[:ng]
        for phi in np.linspace(-np.pi,np.pi,32):
            # Circles on the sphere in two perpendicular planes.
            temp=[Rt*np.sin(theta)*np.cos(phi), Rt*np.cos (theta)*np.cos(phi),\
                Rt*np.sin(phi)*np.ones(ng)]
            ax.plot(temp[0],temp[1],temp[2], 'gray', alpha=0.2, lw=1)
            temp=[Rt*np.cos(theta)*np.cos(phi), Rt*np.sin(phi)*np.ones(ng),
                Rt*np.sin(theta)*np.cos(phi)]
            ax.plot(temp[0],temp[1],temp[2], 'gray', alpha=0.2,lw=1)
            
        for phi in np.linspace(-np.pi,np.pi,120):
            # The circles of the cylinder.
            temp=[Rp*np.sin(theta)-offset,Rp*np.cos(theta),2*Rt*phi*np.ones(np.size(theta))/np.pi]
            ax.plot( temp[0],temp[1],temp[2], 
                    'm', alpha=0.2, lw=1)
            
    #set the range of phi depending on the regime we are in    
    if Rt>=(Rp+offset):
        theta0=2*np.pi
        flag_regime=0
    elif Rt<(Rp+offset):
        theta0=np.arccos(-b/Rp)
        flag_regime=1
    
    #define phi and calcualte b
    phi=np.linspace(-theta0, theta0, Nplot)
    if zsign==-1:
        (x,y,z)=(Rp*np.cos(phi)-offset ,Rp*np.sin(phi), + np.sqrt(2*offset*(b+Rp*np.cos(phi)))) #Calculate the positive z side of intersection
        
        # print('angles',theta_testy,theta_testz,theta_testy2)
        #plot the intersection
        if flag_plot_rot==1:
            temp=rotatey(x,y,z,theta_testy)
            temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
            temp=rotatey(temp[0],temp[1],temp[2],theta_testy2)
        else:
            temp=[x,y,z]
        ax.plot(temp[0],temp[1],temp[2], 'r', lw=3)
        
        (x,y,z)=(Rp*np.cos(phi)-offset ,Rp*np.sin(phi), - np.sqrt(2*offset*(b+Rp*np.cos(phi)))) #other side of solution
    
        # print('angles',theta_testy,theta_testz,theta_testy2)
        #plot the intersection
        if flag_plot_rot==1:
            temp=rotatey(x,y,z,theta_testy)
            temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
            temp=rotatey(temp[0],temp[1],temp[2],theta_testy2)
        else:
            temp=[x,y,z]
        ax.plot(temp[0],temp[1],temp[2], 'b', lw=3)
    
    else:
        (x,y,z)=(Rp*np.cos(phi)-offset ,Rp*np.sin(phi), + np.sqrt(2*offset*(b+Rp*np.cos(phi)))) #Calculate the positive z side of intersection

        # print('angles',theta_testy,theta_testz,theta_testy2)
        #plot the intersection
        if flag_plot_rot==1:
            temp=rotatey(x,y,z,theta_testy)
            temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
            temp=rotatey(temp[0],temp[1],temp[2],theta_testy2)
        else:
            temp=[x,y,z]
        ax.plot(temp[0],temp[1],temp[2], 'r', lw=3)

    #also plot the edge of the tangent
    if flag_regime==1:
        temp=np.linspace(y[-1],y[0],Nplot)
        temp=[-np.sqrt(Rt**2-temp**2), temp, np.zeros(Nplot)]
        if flag_plot_rot==1:
            temp=rotatey(temp[0],temp[1],temp[2],theta_testy)
            temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
            temp=rotatey(temp[0],temp[1],temp[2],theta_testy2)
        ax.plot(temp[0],temp[1],temp[2], 'r:', lw=1)


    # print('angles',theta_testy,theta_testz,theta_testy2)
    t=np.linspace(-5*zsign,0,10)
    # t=np.linspace(-1E-7,0,10)
    temp=calc_tangent(phitest,t,Rp,offset,b,zsign)
    if flag_plot_rot==1:
        temp=rotatey(temp[0],temp[1],temp[2],theta_testy)
        temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
        temp=rotatey(temp[0],temp[1],temp[2],theta_testy2)
    # print('plotting tangent', temp)

    scale=1.0
    ax.plot(temp[0]*scale,temp[1]*scale,temp[2]*scale, 'm', lw=1)
     
    #plot the opposite tangent
    # print('angles',theta_testy,theta_testz,theta_testy2)
    t=np.linspace(-5*zsign,0,10)
    # t=np.linspace(-1E-7,0,10)
    temp=calc_tangent(-phitest,t,Rp,offset,b,zsign)
    if flag_plot_rot==1:
        temp=rotatey(temp[0],temp[1],temp[2],theta_testy)
        temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
        temp=rotatey(temp[0],temp[1],temp[2],theta_testy2)
    # print('plotting tangent', temp)

    scale=1.0
    ax.plot(temp[0]*scale,temp[1]*scale,temp[2]*scale, 'm--', lw=1)
     
    ax.plot([test_rot[0]],[test_rot[1]], [test_rot[2]], 'g+')
    # ax.plot([test_rot1[0]],[test_rot1[1]], [test_rot1[2]], 'g+')

    #trying to find the opposite
    if flag_regime==1:
        Zmax=np.asarray([-b-offset,np.sqrt(Rp**2-b**2),0.])
        if flag_plot_rot==1:
            # print(theta_testy,theta_testz,theta_testy2)
            temp=rotatey(Zmax[0],Zmax[1],Zmax[2],theta_testy)
            # print('max',temp)
            temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
            # print(temp)
            Zmax_rot=np.asarray(rotatey(temp[0],temp[1],temp[2],theta_testy2))
        else:
            Zmax_rot=Zmax

        Zmin=np.asarray([-b-offset,-np.sqrt(Rp**2-b**2),0.])
        if flag_plot_rot==1:
            temp=rotatey(Zmin[0],Zmin[1],Zmin[2],theta_testy)
            temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
            Zmin_rot=np.asarray(rotatey(temp[0],temp[1],temp[2],theta_testy2))
        else:
            Zmin_rot=Zmin

        # print('Z',Zmin,Zmax)
        # print(Zmin_rot,Zmax_rot)
        
        if np.sign(Zmin_rot[2])!=np.sign(Zmax_rot[2]):
            # print('Zmax etc.')
            # print(Zmin)
            # print(Zmax)
            # print(Zmin_rot)
            # print(Zmax_rot)
            
            fac=np.abs(Zmin_rot[2]/Zmax_rot[2])

            # print('factor', fac)
            # print(type(Zmin_rot),type(fac),type(Zmax_rot))
            Zav_rot=Zmin_rot-fac*Zmax_rot

            #renormalize to the radius of the target
            scale=np.sqrt(Rt**2/(np.sum(Zav_rot**2)))
            Zav_rot=scale*Zav_rot
            
            op_rot=Zav_rot
            # print('a',op_rot)

            if flag_plot_rot==0:
                op_rot=rotatey(op_rot[0],op_rot[1],op_rot[2],-theta_testy2)
                op_rot=rotatez(op_rot[0],op_rot[1],op_rot[2],-theta_testyz)
                op_rot=rotatey(op_rot[0],op_rot[1],op_rot[2],-theta_testy)
            
            # print(Zav_rot)

            # print((point_rot[1]>0.0)&(np.abs(point_rot[0]/point_rot[1])<np.abs(op_rot[0]/op_rot[1])))
            

        else:
            r1 =np.random.default_rng(1)
            count=0
            init_op=np.pi+phitest
            temp=1E10
            while (np.abs(temp)>1E-10):
                count+=1
                
                if np.abs(init_op)>theta0:
                    temprand=r1.random(1)
                    if temprand>0.5:
                        init_op=np.sign(init_op)*theta0
                    else:
                        init_op=np.sign(init_op)*(theta0-1E-12*r1.random(1))
                    # init_op=np.sign(init_op)*(theta0-1E-12*r1.random(1))
                phi_calc,infodict,ier,msg=fsolve(func_find_great_circle_op, [init_op],args=(phitest,Rt,Rp,offset,zsign),full_output=True)
                temp=func_find_great_circle_op(phi_calc,phitest,Rt,Rp,offset,zsign)

                if np.abs(phi_calc)>theta0:
                    temp=1E10
                
                if count==100:
                    # print('hit itter limit')
                    temp=0.0
                else:
                    init_op+=(r1.random(1)-0.5)*np.pi

            # print(phi_calc,phitest)

            temp=calc_tangent(phi_calc,0.0,Rp,offset,b,zsign)
            if flag_plot_rot==1:
                temp=rotatey(temp[0],temp[1],temp[2],theta_testy)
                temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
                op_rot=rotatey(temp[0],temp[1],temp[2],theta_testy2)
            else:
                op_rot=temp

            # print('b',op_rot)

    else:
        r1 =np.random.default_rng(1)
        count=0
        init_op=np.pi+phitest
        temp=1E10
        while (np.abs(temp)>1E-10):
            count+=1
            
            if np.abs(init_op)>theta0:
                temprand=r1.random(1)
                if temprand>0.5:
                    init_op=np.sign(init_op)*theta0
                else:
                    init_op=np.sign(init_op)*(theta0-1E-12*r1.random(1))
            phi_calc,infodict,ier,msg=fsolve(func_find_great_circle_op, [init_op],args=(phitest,Rt,Rp,offset,zsign),full_output=True)
            temp=func_find_great_circle_op(phi_calc,phitest,Rt,Rp,offset,zsign)

            if np.abs(phi_calc)>theta0:
                temp=1E10
                
            if count==100:
                # print('hit itter limit')
                temp=0.0
            else:
                init_op+=(r1.random(1)-0.5)*np.pi

        temp=calc_tangent(phi_calc,0.0,Rp,offset,b,zsign)
        if flag_plot_rot==1:
            temp=rotatey(temp[0],temp[1],temp[2],theta_testy)
            temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
            op_rot=rotatey(temp[0],temp[1],temp[2],theta_testy2)
        else:
            op_rot=temp

        # print('c',op_rot)
    
    ax.plot(op_rot[0],op_rot[1],op_rot[2],'cx')
    

    #now rotate the point that was given
    # print('angles',theta_testy,theta_testz,theta_testy2)
    if flag_plot_rot==1:
        temp=rotatey(xc,yc,zc,theta_testy)
        temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
        temp=rotatey(temp[0],temp[1],temp[2],theta_testy2)
    else:
        temp=[xc,yc,zc]
    # print(temp)
    ax.plot(temp[0],temp[1],temp[2],'r.')
    
    # ax.set_aspect('equal')
    ax.set_box_aspect([1.0, 1.0, 1.0])
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    
    plot_radius = 0.5*max([x_range, y_range, z_range])
    
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    # Tidy up by switching off the axes and setting the view orientation.
    # ax.set_axis_off()
    ax.set_xlabel('x*')
    ax.set_ylabel('y*')
    ax.set_zlabel('z*')
    ax.view_init(90, 90)

    return ax

##############################################################################
# CALC_TANGENT:
# Function to calculate the tangent at an arbitary point:
def calc_tangent(phi,t,Rp,offset,b,zsign,flag_theta0=0):
    # print('passed to tangent',phi,t,Rp,offset,b,zsign,flag_theta0)
    # zsign=1
    if flag_theta0==0:
        # print('doing it right')
        (xg0,yg0,zg0)=(Rp*np.cos(phi)-offset ,Rp*np.sin(phi), zsign*np.sqrt(2*offset*(b+Rp*np.cos(phi))))
    else:
        # print('like, etf')
        (xg0,yg0,zg0)=(-b-offset,np.sign(flag_theta0)*np.sqrt(Rp**2-b**2),0.0)
        # print('xg0',flag_theta0,xg0,yg0,zg0)
        # ax.plot(xg0, yg0, zg0, 'm+', ms=10)
    # print(xg0,yg0,zg0)
    # print(np.sin(phi))
    
    # if (np.abs(2*np.sqrt((b+Rp*np.cos(phi))))<1E-14)|(flag_theta0!=0):
    if (flag_theta0!=0):
        xg=0.0*t +xg0
        yg=0.0*t +yg0
        zg=-1.*zsign*Rp*np.sin(phi)*np.sqrt(2*offset)*t+zg0
        # print('we were too close')
        # print(np.sin(phi),np.cos(phi),b+Rp*np.cos(phi),np.sqrt((b+Rp*np.cos(phi))))
    else:
        xg=-Rp*np.sin(phi)*(2*np.sqrt((b+Rp*np.cos(phi))))*t +xg0
        yg=Rp*np.cos(phi)*(2*np.sqrt((b+Rp*np.cos(phi))))*t +yg0
        zg=-1.*zsign*Rp*np.sin(phi)*np.sqrt(2*offset)*t+zg0
        # xg=-Rp*np.sin(phi)*t +xg0
        # yg=Rp*np.cos(phi)*t +yg0
        # zg=-1.*zsign*Rp*np.sin(phi)*np.sqrt(2*offset)/(2*np.sqrt((b+Rp*np.cos(phi))))*t +zg0
    
    # ax.plot(xg, yg, zg, 'm', lw=1)
    # print(xg,yg,zg)
    # print('tangent',phi,xg,yg,zg)

    #find the normal to both the positipon and perpendicular vector
    xpl=yg0*zg-zg0*yg + xg0
    ypl=zg0*xg-xg0*zg +yg0
    zpl=xg0*yg-yg0*xg +zg0

    return xpl, ypl, zpl


##############################################################################
# ROTATE X, Y, Z:
def rotatex(x, y, z, theta):
    
    xrot=1.0*x
    yrot=np.cos(theta)*y-np.sin(theta)*z
    zrot=np.sin(theta)*y +np.cos(theta)*z
    
    return xrot, yrot, zrot

def rotatey(x, y, z, theta):
    
    xrot=np.cos(theta)*x+np.sin(theta)*z
    yrot=1.0*y
    zrot=-np.sin(theta)*x +np.cos(theta)*z
    
    return xrot, yrot, zrot

def rotatez(x, y, z, theta):
    
    xrot=np.cos(theta)*x-np.sin(theta)*y
    yrot=np.sin(theta)*x +np.cos(theta)*y
    zrot=z*1.
    
    return xrot, yrot, zrot


##############################################################################
# FUNC_FIND_GREAT_CIRCLE:
#function to use to find the
def func_find_great_circle(phitest,x,y,z,Rt,Rp,offset,zsign,flag_output=0):


    #calc b
    b=(Rt**2-Rp**2-offset**2)/(2*offset)
    
    #set the range of phi depending on the regime we are in    
    if Rt>=(Rp+offset):
        theta0=2*np.pi
        flag_regime=0
    elif Rt<(Rp+offset):
        theta0=np.arccos(-b/Rp)
        flag_regime=1

    # if np.abs(phitest)>np.abs(theta0):
    #     flag_regime=2

    if np.abs(phitest)>theta0:
        if flag_output==0:
            # return -np.sign(theta0)*np.pi/2+theta0
            # return 0.1
            return 0.1+100.*(np.abs(phitest)-theta0)
            # return 1E6 
        else:
            # print('pummice')
            # return 0.1, np.nan
            return 0.1+100.*(np.abs(phitest)-np.abs(theta0)), np.nan
            # return -np.sign(theta0)*np.pi/2+theta0+0.1, np.nan

    # print('phitest',phitest,b,theta0)
    
    #find the test point (and 
    t=np.linspace(-5*zsign,0,2)
    temp=calc_tangent(phitest,t,Rp,offset,b,zsign)
    
    # if flag_regime==2:
    #     Zmax=np.asarray([-b-offset,np.sqrt(Rp**2-b**2),0.])
    #     Zmin=np.asarray([-b-offset,-np.sqrt(Rp**2-b**2),0.])

    #     fac=(np.abs(phitest)-theta0)/(np.pi-theta0)

    #     # print('factor', fac)
    #     # print(type(Zmin_rot),type(fac),type(Zmax_rot))
    #     if phitest>0:
    #         temp[0][1]=Zmin[0]-fac*Zmax[0]
    #         temp[1][1]=Zmin[1]-fac*Zmax[1]
    #         temp[2][1]=Zmin[2]-fac*Zmax[2]

    #         temp[0][0]=Zmin[0]-fac*Zmax[0]
    #         temp[1][0]=Zmin[1]-fac*Zmax[1]
    #         temp[2][0]=Zmin[2]-fac*Zmax[2] - Rt

            
    #     else:
    #         temp[0][1]=Zmax[0]-fac*Zmin[0]
    #         temp[1][1]=Zmax[1]-fac*Zmin[1]
    #         temp[2][1]=Zmax[2]-fac*Zmin[2]

    #         temp[0][0]=Zmax[0]-fac*Zmin[0]
    #         temp[1][0]=Zmax[1]-fac*Zmin[1]
    #         temp[2][0]=Zmax[2]-fac*Zmin[2] - Rt
              
    test=np.asarray([temp[0][1],temp[1][1],temp[2][1]])

    if test[0]>0:
        theta_testy=zsign*-1*(np.pi-np.arctan(np.abs(test[2]/test[0])))
    else:
        theta_testy=zsign*-1*np.arctan(np.abs(test[2]/test[0]))
    # if test[0]>0:
    #     theta_testy=-(np.pi-np.arctan(np.abs(test[2]/test[0])))
    # else:
    #     theta_testy=-np.arctan(np.abs(test[2]/test[0]))
    
    test_rot=rotatey(test[0],test[1],test[2],theta_testy)

   
    if test_rot[1]>0:
        theta_testz=-np.arctan(np.abs(test_rot[0]/test_rot[1]))
    else:
        theta_testz=-np.pi/2-np.arctan(np.abs(test_rot[1]/test_rot[0]))
    
    test_rot=rotatez(test_rot[0],test_rot[1],test_rot[2],theta_testz)

    
    #find the new tangent
    t=np.linspace(-5*zsign,0,2)
    temp=calc_tangent(phitest,t,Rp,offset,b,zsign)
    temp=rotatey(temp[0],temp[1],temp[2],theta_testy)
    temp=rotatez(temp[0],temp[1],temp[2],theta_testz)

    tang_rot=temp

    # if ((temp[2][0]>0)&(temp[0][0]<0))|((temp[2][0]<0)&(temp[0][0]>0)):
    #     theta_testy2=-np.arctan(np.abs(temp[2][0]/temp[0][0]))
    # else:
    #     theta_testy2=np.arctan(np.abs(temp[2][0]/temp[0][0]))
    if ((temp[2][0]>0)&(temp[0][0]<0))|((temp[2][0]<0)&(temp[0][0]>0)):
        theta_testy2=-np.arctan(np.abs(temp[2][0]/temp[0][0]))
    else:
        if zsign==-1:
            theta_testy2=-(np.pi-np.arctan(np.abs(temp[2][0]/temp[0][0])))
        else:
            theta_testy2=np.arctan(np.abs(temp[2][0]/temp[0][0]))
    
    tang_rot=rotatey(tang_rot[0],tang_rot[1],tang_rot[2],theta_testy2)
    test_rot=rotatey(test_rot[0],test_rot[1],test_rot[2],theta_testy2)

    #now rotate the point that was given
    # print('rotating')
    temp=rotatey(x,y,z,theta_testy)
    # print(temp)
    temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
    # print(temp)
    point_rot=rotatey(temp[0],temp[1],temp[2],theta_testy2)

    # print(theta_testy,theta_testz,theta_testy2,point_rot)

    # print('\t \t tang_rot',tang_rot)
    # print('\t \t point_rot',point_rot)
    # print('\t \t angles', theta_testy, theta_testz, theta_testy2)
    # print(x,y,z)
    # print('point rot',point_rot)
    #when where it sits relative to the line
    # if flag_output==1:
    #     print(tang_rot)
    #     print('angles of output',point_rot[0],tang_rot[0][0],tang_rot[0][1])
    if np.sign(point_rot[0])==np.sign(tang_rot[0][0]): #want it the same side of the y axis
        #return teh z coordinate as it should be in the plane
        # print(point_rot[2]/Rp)
        # print(dlfkjd)
        if flag_output==0:
            return point_rot[2]/Rp
        else:
            if point_rot[1]>Rt:
                lost=1
            else:
                lost=0
            
            return point_rot[2]/Rp, lost
    else:
        #need to find the point opposite the intersection to see if it is in the angle spanned by the impact site

        #check if the point is definitely outside the intersection cylinder
        # if (z<0)|((offset-Rp>0.0)&(x>0.)):
        #     print('I dont know what this condition is for')
        #     if flag_output==0:
        #         # return 1.0+np.abs(theta0-phitest)
        #         return -0.1
        #         # return -np.sign(theta0)*np.pi/2+theta0
        #         # return 1E6 
        #     else:
        #         # print('pummice')
        #         # return 1.0+np.abs(theta0-phitest), np.nan
        #         return 0.1, np.nan
        #         # return -np.sign(theta0)*np.pi/2+theta0, np.nan
                
        
        #first need to check if the intersection curve is open 
        if flag_regime==1:
            # print('in Z')
            Zmax=np.asarray([-b-offset,np.sqrt(Rp**2-b**2),0.])
            temp=rotatey(Zmax[0],Zmax[1],Zmax[2],theta_testy)
            temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
            Zmax_rot=np.asarray(rotatey(temp[0],temp[1],temp[2],theta_testy2))

            Zmin=np.asarray([-b-offset,-np.sqrt(Rp**2-b**2),0.])
            temp=rotatey(Zmin[0],Zmin[1],Zmin[2],theta_testy)
            temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
            Zmin_rot=np.asarray(rotatey(temp[0],temp[1],temp[2],theta_testy2))

            # print('Zm...',Zmax_rot,Zmin_rot)
            
            if np.sign(Zmin_rot[2])!=np.sign(Zmax_rot[2]):
                # print('Zmax etc.')
                # print(Zmin)
                # print(Zmax)
                # print(Zmin_rot)
                # print(Zmax_rot)
                
                # fac=np.abs(Zmin_rot[2]/Zmax_rot[2])

                # print('factor', fac)
                # print(type(Zmin_rot),type(fac),type(Zmax_rot))
                Zav_rot=Zmin_rot-Zmin_rot[2]*(Zmax_rot-Zmin_rot)/(Zmax_rot[2]-Zmin_rot[2])

                # print('Zav_mag',np.sqrt(np.sum(Zav_rot**2))/Rt)
                #renormalize to the radius of the target
                scale=np.sqrt(Rt**2/(np.sum(Zav_rot**2)))
                Zav_rot=scale*Zav_rot
                
                
                # temp=Zav_rot
                # temp=rotatey(temp[0],temp[1],temp[2],theta_testy)
                # temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
                # op_rot=rotatey(temp[0],temp[1],temp[2],theta_testy2)
                op_rot=Zav_rot
                
                # print('opposite',Zav_rot,op_rot)

                # print((point_rot[1]>0.0)&(np.abs(point_rot[0]/point_rot[1])<np.abs(op_rot[0]/op_rot[1])))

                phi_calc=np.nan
            
            #if both close to zero then just use a straight average
            elif (np.abs(Zmin_rot[2])<1E-14) & (np.abs(Zmax_rot[2])<1E-14):
                # print('\t baking potato')
                if phitest>0.:
                    Zav_rot=Zmax_rot
                else:
                    Zav_rot=Zmin_rot
                Zav_rot[2]=0.0
                scale=np.sqrt(Rt**2/(np.sum(Zav_rot**2)))
                Zav_rot=scale*Zav_rot
                op_rot=Zav_rot
                phi_calc=np.nan

            else:

                r1 =np.random.default_rng(1)
                count=0
                # init_op=np.pi+phitest
                init_op=-phitest
                temp=1E10
                if np.abs(phitest-np.sign(phitest)*theta0)<1E-6:
                    toll=1E-6
                else:
                    toll=1E-10

                while (np.abs(temp)>toll):
                    count+=1
                    if np.abs(init_op)>theta0:
                        temprand=r1.random(1)
                        if temprand>0.9:
                            init_op=np.sign(init_op)*theta0
                        else:
                            init_op=np.sign(init_op)*(theta0-1E-12*r1.random(1))
                    phi_calc,infodict,ier,msg=fsolve(func_find_great_circle_op, [init_op],args=(phitest,Rt,Rp,offset,zsign),full_output=True)
                    temp=func_find_great_circle_op(phi_calc,phitest,Rt,Rp,offset,zsign)

                    if (flag_regime==1)&(np.abs(phi_calc)>theta0):
                        # print('poptato')
                        temp=1E10
                        
                    # print(count,init_op,phi_calc,theta0,temp)
                    if count==50:
                        init_op=np.pi+phitest
                    elif count==100:
                        # print('no opposite 1')
                        temp=0.0
                    else:
                        init_op=init_op+(r1.random(1)-0.5)*np.pi
                    

                temp=calc_tangent(phi_calc,0.0,Rp,offset,b,zsign)
                temp=rotatey(temp[0],temp[1],temp[2],theta_testy)
                temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
                op_rot=rotatey(temp[0],temp[1],temp[2],theta_testy2)

        else:
            # print('finding the opposite point')
            r1 =np.random.default_rng(1)
            count=0
            # init_op=np.pi+phitest
            init_op=-phitest
            temp=1E10

            if np.abs(phitest-np.sign(phitest)*theta0)<1E-6:
                toll=1E-6
            else:
                toll=1E-10
                    
            while (np.abs(temp)>toll):
                count+=1
                
                if np.abs(init_op)>theta0:
                    temprand=r1.random(1)
                    if temprand>0.9:
                        init_op=np.sign(init_op)*theta0
                    else:
                        init_op=np.sign(init_op)*(theta0-1E-12*r1.random(1))
                phi_calc,infodict,ier,msg=fsolve(func_find_great_circle_op, [init_op],args=(phitest,Rt,Rp,offset,zsign),full_output=True)
                temp=func_find_great_circle_op(phi_calc,phitest,Rt,Rp,offset,zsign)
                
                # print(count,init_op,phi_calc,phitest,theta0,temp)
                
                if (flag_regime==1)&(np.abs(phi_calc)>theta0):
                    
                    temp=1E10
                    
                if count==50:
                    init_op=np.pi+phitest
                elif count==100:
                    # print('no opposite 2')
                    temp=0.0
                else:
                    init_op=init_op+(r1.random(1)-0.5)*np.pi

            
            temp=calc_tangent(phi_calc,0.0,Rp,offset,b,zsign)
            temp=rotatey(temp[0],temp[1],temp[2],theta_testy)
            temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
            op_rot=rotatey(temp[0],temp[1],temp[2],theta_testy2)

        if (np.sign(op_rot[0])==np.sign(point_rot[0]))&((((point_rot[1]>0.0)&(op_rot[1]>0.0)&(np.abs((point_rot[0])/point_rot[1])<np.abs((op_rot[0])/op_rot[1])))|\
            ((point_rot[1]<=0.0)&(op_rot[1]<=0.0)&(np.abs((point_rot[1])/point_rot[0])<np.abs((op_rot[1])/op_rot[0])))|\
                                                        (point_rot[1]>0.0)&(op_rot[1]<0.0))):
        # if (point_rot[1]>0.0)&(np.arctan(np.abs((point_rot[0]+Rt)/point_rot[1]))<np.arctan(np.abs((op_rot[0]+Rt)/op_rot[1])))|\
        # (point_rot[1]<0.0)&(np.arctan(np.abs(point_rot[0]/point_rot[1]))>np.arctan(np.abs(op_rot[0]/op_rot[1]))): #check whether in the angle of the cylinder
        # if (np.arctan(np.abs(point_rot[0]/point_rot[1]))<np.arctan(np.abs(op_rot[0]/op_rot[1]))): #check whether in the angle of the cylinder
        # if (np.arctan(np.abs(point_rot[0]/point_rot[1]))<np.arctan(np.abs(op_rot[0]/op_rot[1]))): #check whether in the angle of the cylinder
            # print(point_rot[2]/Rp)
            # print(dlfkjd)
            # print('returning at b',op_rot[0],point_rot[0])
            if flag_output==0:
                return point_rot[2]/Rp
            else:
                # print('ettuce')
                # print(point_rot)
                # print(op_rot)
                if np.sqrt(point_rot[0]**2+point_rot[1]**2)>Rt:
                    lost=1
                else:
                    lost=0
                return point_rot[2]/Rp, lost
        else:
            # print('returning at c',op_rot[0],point_rot[0])
            if flag_output==0:
                # return 1.0+np.abs(theta0-phitest)
                return 0.1
                # return -np.sign(theta0)*np.pi/2+theta0
                # return 1E6 
            else:
                # print('pummice')
                return 0.1, np.nan
                # return -np.sign(theta0)*np.pi/2+theta0+0.1, np.nan
                
                # return 1.0E6, np.nan

    # print(temp[2]/Rt)
            

##############################################################################
# FUNC_FIND_GREAT_CIRCLE_OP:
# Function to use to find the point opposite the point you have found
def func_find_great_circle_op(phiop,phitest,Rt,Rp,offset,zsign):

    if (np.abs(phiop)>1E-6)&(np.abs(phitest)>1E-6)&(np.abs(phiop-phitest)<1E-7):
        return 1./(phiop-phitest)
    elif (np.abs((phiop-2*np.pi))>1E-10)|(np.abs((2*np.pi-phiop))>1E-10):
        return np.abs(phiop)-2*np.pi+np.sign(np.abs(phiop)-2*np.pi)*1E-6
    

    # if (np.abs(phiop-phitest)<1E-6)&(np.abs((phitest-2*np.pi))>1E-10)&(np.abs((2*np.pi-phitest))>1E-10):
    #     return 1./(phiop-phitest)

    #calc b
    b=(Rt**2-Rp**2-offset**2)/(2*offset)
    
    #set the range of phi depending on the regime we are in    
    if Rt>=(Rp+offset):
        theta0=2*np.pi
        flag_regime=0
    elif Rt<(Rp+offset):
        theta0=np.arccos(-b/Rp)
        flag_regime=1
    
    #find the test point (and 
    t=np.linspace(-5*zsign,0,2)
    temp=calc_tangent(phitest,t,Rp,offset,b,zsign)
    test=np.asarray([temp[0][1],temp[1][1],temp[2][1]])

    if test[0]>0:
        theta_testy=zsign*-1*(np.pi-np.arctan(np.abs(test[2]/test[0])))
    else:
        theta_testy=zsign*-1*np.arctan(np.abs(test[2]/test[0]))
    # if test[0]>0:
    #     theta_testy=-(np.pi-np.arctan(np.abs(test[2]/test[0])))
    # else:
    #     theta_testy=-np.arctan(np.abs(test[2]/test[0]))
    
    test_rot=rotatey(test[0],test[1],test[2],theta_testy)
    
    if test_rot[1]>0:
        theta_testz=-np.arctan(np.abs(test_rot[0]/test_rot[1]))
    else:
        theta_testz=-np.pi/2-np.arctan(np.abs(test_rot[1]/test_rot[0]))
    
    test_rot=rotatez(test_rot[0],test_rot[1],test_rot[2],theta_testz)
    
    #find the new tangent
    t=np.linspace(-5*zsign,0,2)
    temp=calc_tangent(phitest,t,Rp,offset,b,zsign)
    temp=rotatey(temp[0],temp[1],temp[2],theta_testy)
    temp=rotatez(temp[0],temp[1],temp[2],theta_testz)

    # if ((temp[2][0]>0)&(temp[0][0]<0))|((temp[2][0]<0)&(temp[0][0]>0)):
    #     theta_testy2=-np.arctan(np.abs(temp[2][0]/temp[0][0]))
    # else:
    #     theta_testy2=np.arctan(np.abs(temp[2][0]/temp[0][0]))

    if ((temp[2][0]>0)&(temp[0][0]<0))|((temp[2][0]<0)&(temp[0][0]>0)):
        theta_testy2=-np.arctan(np.abs(temp[2][0]/temp[0][0]))
    else:
        if zsign==-1:
            theta_testy2=-(np.pi-np.arctan(np.abs(temp[2][0]/temp[0][0])))
        else:
            theta_testy2=np.arctan(np.abs(temp[2][0]/temp[0][0]))

    # test_rot=rotatey(test_rot[0],test_rot[1],test_rot[2],theta_testy2)

    temp=calc_tangent(phiop,0.0,Rp,offset,b,zsign)
    temp=rotatey(temp[0],temp[1],temp[2],theta_testy)
    temp=rotatez(temp[0],temp[1],temp[2],theta_testz)
    temp=rotatey(temp[0],temp[1],temp[2],theta_testy2)

    

    #return th z coordinate as it should be in the plane
    return temp[2]/Rp


##############################################################################
# FUNC_CALC_MAX_ANGLE:
# Find the maximum angle downwards of any point on the curve
def func_calc_max_angle(phi,Rp,offset,b,theta0):
    if np.abs(phi)>theta0:
        return 2*np.pi
    t=np.linspace(-2.0,0,2)
    temp=calc_tangent(phi,t,Rp,offset,b,1)
    rxy=np.sqrt((temp[0][1]-temp[0][0])**2+(temp[1][1]-temp[1][0])**2)
    return -np.arctan(np.abs((temp[2][0]-temp[2][1])/(rxy)))


##############################################################################
# CALC_NEAR_FAR_FIELD_OFFSET:
def calc_near_far_field_offset(x,y,z,Rt,Rp,offset,Nit=200):

    #calc b
    b=(Rt**2-Rp**2-offset**2)/(2*offset)
    
    #set the range of phi depending on the regime we are in    
    if Rt>=(Rp+offset):
        theta0=2*np.pi
        flag_regime=0
    elif Rt<(Rp+offset):
        theta0=np.arccos(-b/Rp)
        flag_regime=1

    #find the angle

    # print('calc_near_far_field_offset',np.abs(z)/Rt)
    

    #if close to theta0 then need to do a more suibtle root finding
    if (flag_regime==1)&((np.abs(z)/Rt)<1E-2):
        flag_sol_upper=1

        ref_theta0=func_find_great_circle(np.sign(y)*theta0,x,y,z,Rt,Rp,offset,1)

        # print('we are dong it')
        # print(ref_theta0)

        init=np.sign(y)*theta0*(1-1E-12)
        
        r2 =np.random.default_rng(1)
        temp=1E10
        count=0
        while (np.abs(temp)>1E-2):
            # print(count)
            count+=1
            # init+=(r2.random(1)-0.5)*(theta0*1E-6)
            init+=(r2.random(1)-0.5)*(theta0*1E-3)
            if np.abs(init)>theta0:
                init=np.sign(init)*(np.abs(theta0)-1E-14*r2.random(1))
            phi_calc,infodict,ier,msg=fsolve(func_find_great_circle, [init],args=(x,y,z,Rt,Rp,offset,1),full_output=True, xtol=1E-10)
            temp=func_find_great_circle(phi_calc,x,y,z,Rt,Rp,offset,1)
    
            # print(count,init,phi_calc,init-phi_calc,theta0,phi_calc-np.sign(phi_calc)*theta0,ier,temp,np.abs(temp-ref_theta0)/theta0)
    
            if (ier!=1)|(np.abs(phi_calc)>theta0):
                # print('flapjackl')
                # print(ier,msg)
                # print(infodict)
                # print(temp, phi_calc,theta0,np.sign(y)*phi_calc-theta0,(np.abs(phi_calc)>theta0))
                temp=1E10
            # else:
            #     print('tempter')
            #     # print(ier,msg)
            #     # print(infodict)
            #     print(temp, phi_calc,theta0,np.sign(y)*phi_calc-theta0,(np.abs(phi_calc)>theta0))
                
                

            # print('should we use theta0?', temp, ref_theta0, phi_calc, ier, ((((np.abs(temp)<1E-2)))),(np.abs(temp)>np.abs(ref_theta0)),\
            #              ((np.abs((temp-ref_theta0)/ref_theta0)<1E-2)),(np.abs(init-phi_calc)>1E-8))
            #if we have found a solution but theta0 is better, use theta0
            if (ier==1)&((((np.abs(temp)<1E-2))&(np.abs(temp)>np.abs(ref_theta0)))|\
                         ((np.abs((temp-ref_theta0)/ref_theta0)<1E-2)&(np.abs(init-phi_calc)>1E-8))):
            #     print('using theta0 instead', ref_theta0, temp)
            #     print('we did')
            # # print(msg)
            # if (ier!=1)|(np.abs(phi_calc)>theta0):
            #     print('flapjackl')
            #     print(ier,msg)
            #     # print(infodict)
            #     print(temp, phi_calc,theta0,(np.abs(phi_calc)>theta0))
            #     temp=1E10

            # print('should we use theta0?', temp, ref_theta0, phi_calc, ier, ((((np.abs(temp)<1E-2)))),(np.abs(temp)>np.abs(ref_theta0)),\
            #              ((np.abs((temp-ref_theta0)/theta0)<1E-2)),(np.abs(init-phi_calc)>1E-8))
            # #if we have found a solution but theta0 is better, use theta0
            # if (ier==1)&((((np.abs(temp)<1E-2))&(np.abs(temp)>np.abs(ref_theta0)))|\
            #              ((np.abs((temp-ref_theta0)/theta0)<1E-2)&(np.abs(init-phi_calc)>1E-8))):
                print('using theta0 instead', ref_theta0, temp)
                temp=0.0
                phi_calc=np.sign(y)*theta0
                
            
            # print(msg)
            if count==Nit:
                if (np.abs(ref_theta0)<1E-2):
                    print('using theta0 instead count out', ref_theta0, temp)
                    temp=ref_theta0
                    phi_calc=np.sign(y)*theta0
                else:
                    print('No root found from upper quadrant')
                    print(x,y,z)
                    print(phi_calc, theta0)
                    # print(infodict)
                    # print(temp)
                    
                    temp=0.
                    flag_sol_upper=0

    
                        

    else:

        if y>0.:
            init=np.amin([-theta0*0.99,np.pi/4])
        else:
            init=np.amax([theta0*0.99,-np.pi/4])

        # print('did we get here?')
        # init=0.97
        flag_sol_upper=1
        phi_calc,infodict,ier,msg=fsolve(func_find_great_circle, [init],args=(x,y,z,Rt,Rp,offset,1),full_output=True)
        temp=func_find_great_circle(phi_calc,x,y,z,Rt,Rp,offset,1)
        # print(msg)
        # print(phi_calc,temp)
        
        if (ier!=1)|(np.abs(temp)>1E-10):
            phi_calc,infodict,ier,msg=fsolve(func_find_great_circle, [-init],args=(x,y,z,Rt,Rp,offset,1),full_output=True)
            temp=func_find_great_circle(phi_calc,x,y,z,Rt,Rp,offset,1)
            # print(msg)
        
            if (ier!=1)|(np.abs(temp)>1E-10):
        
                r2 =np.random.default_rng(1)
                count=0
                while (np.abs(temp)>1E-10):
                    # print(count)
                    count+=1
                    init+=(r2.random(1)-0.5)*np.pi
                    if np.abs(init)>theta0:
                        init=np.sign(init)*(np.abs(theta0)-1E-12*r2.random(1))
                    phi_calc,infodict,ier,msg=fsolve(func_find_great_circle, [init],args=(x,y,z,Rt,Rp,offset,1),full_output=True)
                    temp=func_find_great_circle(phi_calc,x,y,z,Rt,Rp,offset,1)


                    temp2=calc_tangent(phi_calc,np.linspace(0,-5,2),Rp,offset,b,1)
                    # print(count,init,phi_calc,init-phi_calc,theta0,ier,temp)
                    # print('\t',temp2[0][0]/temp2[1][0],x/y,temp2[0][0]/temp2[2][0],x/z)
    
                    # print(count,init,phi_calc,init-phi_calc,theta0,phi_calc-np.sign(phi_calc)*theta0,temp)
                    if (ier!=1)|(np.abs(phi_calc)>theta0):
                        temp=1E10
                    
                    # print(msg)
                    if count==Nit:
                        print('No root found from upper quadrant')
                        print(x,y,z,temp)
                        # print(x,y,z)
                        # print(phi_calc, theta0)
                        # print(infodict)
                        # print(temp)
                        
                        temp=0.
                        flag_sol_upper=0
                        
                        # raise RuntimeWarning('no root found. aborting.')
    
    if flag_sol_upper==0:
        flag_sol_upper=-1
        phi_calc==np.nan
        # #find the angle
        # if y>0.:
        #     init=np.amin([-theta0*0.99,np.pi/4])
        # else:
        #     init=np.amax([theta0*0.99,-np.pi/4])
            
        # phi_calc,infodict,ier,msg=fsolve(func_find_great_circle, [init],args=(x,y,z,Rt,Rp,offset,-1),full_output=True)
        # temp=func_find_great_circle(phi_calc,x,y,z,Rt,Rp,offset,-1)
        # # print(msg)
        # # print(phi_calc,temp)
        
        # if (ier!=1)|(np.abs(temp)>1E-10):
        #     phi_calc,infodict,ier,msg=fsolve(func_find_great_circle, [-init],args=(x,y,z,Rt,Rp,offset,-1),full_output=True)
        #     temp=func_find_great_circle(phi_calc,x,y,z,Rt,Rp,offset,-1)
        #     # print(msg)
        
        #     if (ier!=1)|(np.abs(temp)>1E-10):
        
        #         r2 =np.random.default_rng(1)
        #         count=0
        #         while (np.abs(temp)>1E-10):
        #             # print(count)
        #             count+=1
        #             init+=(r2.random(1)-0.5)*np.pi
        #             if np.abs(init)>theta0:
        #                 randtemp=r2.random(1)
        #                 if randtemp>0.5:
        #                     init=np.sign(init)*theta0
        #                 else:
        #                     init=np.sign(init)*(np.abs(theta0)-1E-12*r2.random(1))
        #             phi_calc,infodict,ier,msg=fsolve(func_find_great_circle, [init],args=(x,y,z,Rt,Rp,offset,-1),full_output=True)
        #             temp=func_find_great_circle(phi_calc,x,y,z,Rt,Rp,offset,-1)

        #             if (ier!=1)|(np.abs(phi_calc)>theta0):
        #                 temp=1E10
                        
        #             # print(msg)
        #             # print(temp)
        #             if count==Nit:
        #                 print('No root found from lower quadrant either',time.time()-tstart)
        #                 print(x,y,z)
        #                 print(phi_calc, theta0)
        #                 print(infodict)
        #                 print(temp)
                        
        #                 raise RuntimeWarning('no root found. aborting.')
    
    

    # print('phi',phi_calc,theta0)

    # print('time taken',time.time()-tstart)
    
    
    if flag_sol_upper==-1:
        return phi_calc, flag_sol_upper, -1.
    else:
        temp=func_find_great_circle(phi_calc,x,y,z,Rt,Rp,offset,flag_sol_upper,flag_output=1)
        return phi_calc, flag_sol_upper, temp[1]


##############################################################################
# CALC_NEAR_FAR_HEAD_ON:
def calc_near_far_head_on(x,y,z,Rt,Rp,Nit=1000):

    H=np.sqrt(Rt**2-Rp**2)
    theta_crit=np.pi/2-np.arcsin(Rp/Rt)

    lost=-1*np.ones(np.size(x),dtype=int)

    rxy=np.sqrt(x**2+y**2)
    r=np.sqrt(x**2+y**2+z**2)
    theta=np.arctan((rxy-Rp)/(z-H))


    #particles inside planet
    # ind=np.where(r<Rt)[0]
    # lost[ind]=0

    #particles behind planet
    ind=np.where((rxy<Rt)&(z<0))[0]
    lost[ind]=0

    #particles below the horrizon
    ind=np.where((z<H)&(theta<theta_crit))[0]
    lost[ind]=0

    return lost
    

##############################################################################
# CALC_NEAR_FAR_FIELD:
# Function that calculates the loss for a given set of lat-long-r or (SWIFT) x,y,z points for particles and the properties of an impact
def calc_near_far_field(input,Rt,Rp,bimp,input_flag=0):

    #calculate the offset from the impact parameter
    offset=(Rt+Rp)*bimp

    # print(offset)

    #extract the values that we need
    if input_flag==0:
        lat=input[0]
        lon=input[1]
        r=input[2]

        #convert to internal x, y, and z
        z=r*np.cos(lon)*np.sin(np.pi/2-lat)
        x=-r*np.sin(np.pi/2-lat)*np.sin(lon)
        y=-r*np.cos(np.pi/2-lat)
        
    elif input_flag==1:
        xprime=input[0]
        yprime=input[1]
        zprime=input[2]

        x=-yprime
        y=zprime
        z=xprime

    #normalize inputs
    x=x/Rt
    y=y/Rt
    z=z/Rt
    Rp=Rp/Rt
    offset=offset/Rt
    Rt=1.
    

    nf=np.ones(np.size(x),dtype=int)*np.nan
    field=np.zeros(np.size(x),dtype=int)

    if bimp==0:
        field=calc_near_far_head_on(x,y,z,Rt,Rp)

        ind=np.where(field<0)[0]
        nf[ind]=1
        ind=np.where(field>=0)[0]
        nf[ind]=0

        return nf, field

    else:
        
        
        

        b=(Rt**2-Rp**2-offset**2)/(2*offset)

        #set the range of phi depending on the regime we are in    
        if Rt>=(Rp+offset):
            theta0=2*np.pi
            flag_regime=0
        elif Rt<(Rp+offset):
            theta0=np.arccos(-b/Rp)
            flag_regime=1

        r1 =np.random.default_rng(1)
        count=0
        init_maxslope=theta0*0.9
        test=True
        while test:
            count+=1
        
            if np.abs(init_maxslope)>theta0:
                temprand=r1.random(1)
                if temprand>0.9:
                    init_maxslope=np.sign(init_maxslope)*theta0
                else:
                    init_maxslope=np.sign(init_maxslope)*(theta0-1E-12*r1.random(1))
        
            out=minimize(func_calc_max_angle, init_maxslope,args=(Rp,offset,b,theta0))
            # print(count,init_maxslope,out)
            if count==100:
                print(out)
                raise RuntimeWarning('no root found for maximum slope. aborting.')
            elif out.status!=0:
                init_maxslope=init_maxslope+(r1.random(1)-0.5)*np.pi
            else:
                test=False
                
        phi_maxslope=out.x
        maxslope=-out.fun
        # print('max slope', init,phi_maxslope,theta0)
        # print(phi_maxslope,maxslope)
        
        (xmaxslope,ymaxslope,zmaxslope)=(Rp*np.cos(phi_maxslope)-offset ,Rp*np.sin(phi_maxslope), + np.sqrt(2*offset*(b+Rp*np.cos(phi_maxslope)))) 
        
        count_loop=0
        for i in np.arange(np.size(x)):
            # print(i)
            zpoint0=[x[i],y[i],z[i]]
            zpoint=[x[i],y[i],z[i]]
            
            rpoint=np.sqrt(zpoint[0]**2+zpoint[1]**2+zpoint[2]**2)
            rxypoint=np.sqrt(zpoint[0]**2+zpoint[1]**2)
        
            if (Rp+offset>Rt):
                temp=calc_tangent(theta0,np.asarray([-1.0,0.0]),Rp,offset,b,1,flag_theta0=1)
                ytheta0=temp[1][1]
                xtheta0=temp[0][1]
                # print(ytheta0,xtheta0)
                tanlon_theta0=np.arcsin(np.abs(temp[1][1])/np.sqrt(temp[0][1]**2 + temp[1][1]**2))
                if y[i]>0.:
                    tanlon_tan_from_theta0=np.abs((temp[1][0]-Rp*np.sqrt(1-(b/Rp)**2))/(temp[0][0]-(-b-offset)))
                else:
                    tanlon_tan_from_theta0=np.abs((-temp[1][0]+Rp*np.sqrt(1-(b/Rp)**2))/(temp[0][0]-(-b-offset)))
                    
                if y[i]>0.:
                    theta_tang=np.arctan(np.abs((temp[1][0]-ytheta0)/(temp[0][0]-xtheta0)))
                else:
                    theta_tang=np.arctan(np.abs((-temp[1][0]+ytheta0)/(temp[0][0]-xtheta0)))
                theta_point1=np.arctan(np.abs((y[i]-ytheta0)/(x[i]-xtheta0)))
                theta_point2=np.arctan(np.abs((y[i]+ytheta0)/(x[i]-xtheta0)))

                tanlon_point=np.arcsin(np.abs(y[i])/np.sqrt(x[i]**2+y[i]**2))

                # print('longitudes',tanlon_theta0,tanlon_point)
                # print('angles',theta_point1,theta_point2,theta_tang)
                # print((Rp+offset>Rt)&((zpoint[0]<-Rt)|\
                #                  ((zpoint[2]>0.0)&(zpoint[0]+offset<=-b))|\
                #                  ((zpoint[2]<=0.0)&(rxypoint>Rt)&(zpoint[0]+offset<=-b))|\
                #                  ((zpoint[1]>=ytheta0)&(tanlon_point>=tanlon_theta0)&(theta_point1>theta_tang))|\
                #                  ((zpoint[1]<=-ytheta0)&(tanlon_point>=tanlon_theta0)&(theta_point2>theta_tang))))
        
                # theta_rxyz_point=np.arctan(np.abs((zpoint[2])/(rxypoint)))
                theta_rxyz_point=np.arctan(np.abs((zpoint[2])/(rxypoint-Rt)))

                #find limit for the point at which the tangent intersects z=0
                xint_theta0=offset*(Rp-b**2/Rp)/(Rp+b*offset/Rp)-b-offset
                yint_theta0=np.sqrt(1-(b/Rp)**2)*((b+offset)*offset/(Rp+offset*b/Rp)+Rp)
                if y[i]>0.:
                    tanlon_int_from_theta0=np.abs((yint_theta0-Rp*np.sqrt(1-(b/Rp)**2))/(xint_theta0-(-b-offset)))
                    tanlon_point_from_theta0=np.abs((y[i]-Rp*np.sqrt(1-(b/Rp)**2))/(x[i]-(-b-offset)))
                    rxy_int_theta0=np.sqrt((xint_theta0-xtheta0)**2+(yint_theta0-ytheta0)**2)
                    rxy_theta0=np.sqrt((x[i]-xtheta0)**2+(y[i]-ytheta0)**2)
                else:
                    tanlon_int_from_theta0=np.abs((yint_theta0+Rp*np.sqrt(1-(b/Rp)**2))/(xint_theta0-(-b-offset)))
                    tanlon_point_from_theta0=np.abs((y[i]+Rp*np.sqrt(1-(b/Rp)**2))/(x[i]-(-b-offset)))
                    rxy_int_theta0=np.sqrt((xint_theta0-xtheta0)**2+(yint_theta0+ytheta0)**2)
                    rxy_theta0=np.sqrt((x[i]-xtheta0)**2+(y[i]+ytheta0)**2)

                # print('should we be going here?')
                # print((offset<Rt)&(Rp+offset>Rt)&(zpoint[2]<=0.0),((zpoint[1]>Rp*np.sqrt(1-(b/Rp)**2))&(zpoint[1]<yint_theta0)),((zpoint[0]>(-b-offset))&(zpoint[0]<xint_theta0)),\
                #       (tanlon_point_from_theta0<tanlon_int_from_theta0),((zpoint[1]<-Rp*np.sqrt(1-(b/Rp)**2))&(zpoint[1]>-yint_theta0)),((zpoint[0]>(-b-offset))&(zpoint[0]<xint_theta0)),\
                #       (tanlon_point_from_theta0<tanlon_int_from_theta0))



            else:
                ytheta0=np.nan
                tanlon_theta0=np.nan
                theta_tang=np.nan
                theta_point1=np.nan
                theta_point2=np.nan

                tanlon_point=np.nan
        
                theta_rxyz_point=np.arctan(np.abs((zpoint[2])/(rxypoint-Rt)))
           
                xint_theta0=np.nan
                yint_theta0=np.nan
                tanlon_tan_from_theta0=np.nan
                tanlon_int_from_theta0=np.nan
                tanlon_point_from_theta0=np.nan
                rxy_int_theta0=np.nan
                rxy_theta0=np.nan
        
            #anything in planet is not lost
            # if rpoint<Rt:
            #     lost=1
            #     # print('Point in planet')
            #anything in cylinder with z>0 is lost
            if (((zpoint[0]+offset)**2+zpoint[1]**2)<Rp**2)&(zpoint[0]<(Rp-offset))&(zpoint[2]>=0):
                lost=-1
                # print('In the loss cylinder')
            #anything in the cylinder but not intersecting the planet is lost
            elif (Rp+offset>Rt)&(zpoint[0]+offset<=-b)&(np.abs(zpoint[1])<=ytheta0)&(rxypoint>Rt):
                lost=-2
            #nothing in shadow of planet is lost (saves some dodgy computation)
            elif (zpoint[2]<0.0)&(zpoint[0]**2+zpoint[1]**2<=Rt**2):
                lost=2
            #if the target is impacting only on one hemisphere then can ignore the opposite quartile
            #also don't need to think about the stuff behind the planet in the x direction
            elif ((offset-Rp>0.0)&(((zpoint[2]<0)&(zpoint[0]>0))|((zpoint[0]>0)&(zpoint[1]**2+zpoint[2]**2<=Rt**2)))):
                lost=3
            #tangent blown off when the projectile is grazing
            #1) beyond the edge of the planet
            #2) #if in the top hemisphere and beyond the extrema of the intersection curve in x
            #3) #similarly, if not in shadow of planet below the plane
            #4) #otherwise based on the angle relative to the tangent plane at theta0
            elif (Rp+offset>Rt)&((zpoint[0]<-Rt)|\
                                 ((zpoint[2]>0.0)&(zpoint[0]+offset<=-b))|\
                                 ((zpoint[2]<=0.0)&(rxypoint>Rt)&(zpoint[0]+offset<=-b))|\
                                 ((zpoint[1]>=ytheta0)&(tanlon_point>=tanlon_theta0)&(theta_point1>theta_tang))|\
                                 ((zpoint[1]<=-ytheta0)&(tanlon_point>=tanlon_theta0)&(theta_point2>theta_tang))):#&(zpoint[0]<(Rp-offset)): 

                # print(zpoint,zpoint[0]<-Rt,ytheta0,tanlon_point,tanlon_theta0)
                lost=-3
                # print('lost due to tangent')

           
            # #stuff behind the planet and not in the tangent plane are safe
            # elif (Rp+offset>Rt)&(((zpoint[2]<=0.0)&(rxypoint<Rt))):#&(np.abs(zpoint[1])<=np.sqrt(Rp**2-b**2)))): 
                
            #     lost=4

             #if grazing, points below the plane can only be blown off at the points described above. However, this only works if the center of the impactor is off the target
            elif (offset>Rt)&(zpoint[2]<=0.0)&(((zpoint[1]>=ytheta0)&(tanlon_point>=tanlon_theta0)&(theta_point1<theta_tang))|\
                                 ((zpoint[1]<=-ytheta0)&(tanlon_point>=tanlon_theta0)&(theta_point2<theta_tang))):#&(zpoint[0]<(Rp-offset)): 
                # print('bobs your uncle?')
                lost=4
                
            #if the point is in the range where there is no z=0 intersection then we can get funky
            elif ((offset<Rt)&(Rp+offset>Rt)&(zpoint[2]<=0.0))\
                    &(((zpoint[1]>Rp*np.sqrt(1-(b/Rp)**2))&(zpoint[1]<yint_theta0)&(zpoint[0]>(-b-offset))&(rxy_theta0<=rxy_int_theta0)&(tanlon_point_from_theta0<tanlon_int_from_theta0))\
                   |((zpoint[1]<-Rp*np.sqrt(1-(b/Rp)**2))&(zpoint[1]>-yint_theta0)&(zpoint[0]>(-b-offset))&(rxy_theta0<=rxy_int_theta0)&(tanlon_point_from_theta0<tanlon_int_from_theta0))):

                # print('not lost due to weird intersect/tangent region')
                lost=4
           
        
            #stuff at too low an angle behind the planet is not going to 
            elif (zpoint[2]<0)&(theta_rxyz_point>maxslope):
                lost=5
            
            else: #have to find out where a point lies relative to the tangent
                lost=0.
                # print('ok, we have to do the whole thing')
                (phi_calc, flag_sol_upper, lost) = calc_near_far_field_offset(x[i],y[i],z[i],Rt,Rp,offset)
                count_loop+=1
                # if xplot[i]>-0.5:
                #     print('ok, so we are in the right ball park')
                #     print(xplot[i],yplot[j],zplot[k])
                #     print(lost)
                # lost=0
                if (lost==0)|(lost==0.):
                    lost=6
                elif (lost==1)|(lost==1.):
                    lost=-6
                elif (lost==-1)|(lost==-1.):
                    lost=7
                    # lost=8
                    # print('wtf is happening',lost)
        
                
            field[i]=lost

            if lost>=0:
                nf[i]=0
            else:
                nf[i]=1

        return nf, field