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

from eostable import extEOStable
import colormaps as local_cmaps
from copy import deepcopy

from scipy.signal import find_peaks
from scipy.interpolate import griddata

# EOS structure:
import GSesame_EOS as EOS
import HM80_EOS as EOS_HM80
from calc_hugoniot_from_EOS import *

# Import loss function:
from LS21_loss_function import *

# Constants:
R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg m^-3 
G = 6.67408e-11  # m^3 kg^-1 s^-2
vesc_Earth = 11.186E3

woma.load_eos_tables(A1_mat_input = ["ANEOS_iron",
                                     "ANEOS_forsterite",
                                     "SS08_water",
                                     "HM80_HHe",
                                     "HM80_HHe_extended",
                                     "CD21_HHe",
                                     "custom_0"]) # Loads the eos table for the calculation of entropy at later stage

##############################################################################
# 1D LOSS FUNCTIONS:
Npro = 4

vesc_planet = np.sqrt((2.0 * G * (M*M_earth)) / (M*R_earth))

p0_atmo = 0.72E9 #atmospheric pressure
T0_atmo = 480.68 #atmospheric temp
s0_mantle = 2.61636E3 #mantle entropy

p0_mantle = np.linspace(5E9, 15E9, 11)
ppeak_mantle = np.linspace(15E9, 1000E9, 60)

Nsteps_int = 100
vp_for_calc = np.linspace(100, 30E3, Nsteps_int)

Np0_mantle_test = 3
Nppeak_test = 4

Ncalc_mantle = 100
Ncalc_atmo = 100

water_EOS_file = 'eos_tables/h2o_table_v8.3bNT.txt'
forsterite_EOS_file_stdsesame = 'eos_tables/NEW-SESAME-STD-NOTENSION.TXT'
forsterite_EOS_file_extsesame = 'eos_tables/NEW-SESAME-EXT.TXT'

# Read in the EOS:
water_EOS = EOS.GSesame_EOS()
water_EOS.read_file(water_EOS_file)

for_EOS = EOS.GSesame_EOS()
for_EOS.read_file(forsterite_EOS_file_stdsesame, flag_read = 3, EOS_file2 = forsterite_EOS_file_extsesame)

atmo_EOS = EOS_HM80.HM80_EOS(flag_calc_cv = 0)

# Record the number of points to calc:
Nppeak = np.size(ppeak_mantle)
Np0_mantle = np.size(p0_mantle)
Nv = np.size(vp_for_calc)

# Set up function to allow parallel computation of release and match:
def temp_func_calc_hugoniot(p0, T0, for_EOS, vp_for_calc, i):
    vp_for = vp_for_calc[i]
    print(vp_for)
    
    try:
        # Initalize hugoniots/relese curves:
        Hug = Hugoniot_from_EOS(for_EOS)

        # Calc the forsterite hugoniot (use steps to avoid errors):
        Nsteps_int = int(np.around(vp_for/1E3)) + 8
        temp = np.linspace(100, vp_for, Nsteps_int)
        Hug.calc_hugoniot_up(temp, [p0, T0], 1E-14, flag_version = 1)

        #print(Hug.rho[-1],Hug.p[-1],Hug.T[-1],Hug.up[-1])
    except:
        return 0

    return Hug


def temp_func_calc_hugoniot_p(p0, T0, for_EOS, p_for_calc, Ncalc, toll, i):
    p_for = p_for_calc[i]
    print(p_for, p0)
    
    # Initalize hugoniots/relese curves
    Hug = Hugoniot_from_EOS(for_EOS)

    # Calc the forsterite hugoniot (use steps to avoid errors):
    temp = np.linspace(np.amax([(1 + 1E-5) * p0, np.amin([1.1 * p0, p_for * 0.9])]), p_for,Ncalc)
    # print(temp)
    Hug.calc_hugoniot_p(temp, [p0, T0], toll, flag_version = 1, flag_silent = 1, frac_slow = 0.8, min_frac_conv = 1E-5, Nslow = 200)

    # print(Hug.rho[-1],Hug.p[-1],Hug.T[-1],Hug.up[-1])
    
#     try:
#         #initalize hugoniots/relese curves
#         Hug=Hugoniot_from_EOS(for_EOS)

#         #calc the forsterite hugoniot (use steps to avoid errors)
#         temp=np.linspace(np.amin([2*p0,p_for*0.9]),p_for,Ncalc)
#         Hug.calc_hugoniot_p(temp,[p0,T0],toll,flag_version=1,flag_silent=0)

#         print(Hug.rho[-1],Hug.p[-1],Hug.T[-1],Hug.up[-1])
#     except:
#         return 0

    return Hug


def temp_func_calc_match(j, Ncalc, p0, T0, Hug_arr, EOS1, EOS2, init_guess, vp_for_calc, step_base, toll, i):
        
    vp_for = vp_for_calc[i]
    # print('vvvvvvvvvvvvvvvvvvvvvvvvvvv')
    # print(i,vp_for)
    #initalize hugoniot
    Hug_for = Hug_arr[i]

    # Find the impedance match solution:
    test = 1
    count = 0

    while test != 0:
        count += 1
        # print(count, init_guess[j,1,i])
        print(vp_for, init_guess[j, i], count)

        try:
            # print('frog', p0,T0)
            Hug_temp = Hugoniot_from_EOS(EOS1)

            release = Hugoniot_from_EOS(EOS1)

            Hug_match_temp = Hugoniot_from_EOS(EOS2)

            print('lets get started')
#         temp=find_impedance_match(Hug_temp,Hug_match_temp,p0,T0,Hug.up[-1],Hug.p[-1],Hug.T[-1],initial_guess_fac=0.9,Ncalc=1000)
            temp = find_impedance_match(Hug_temp, Hug_match_temp, p0, T0, Hug_for.up[-1], Hug_for.p[-1], Hug_for.T[-1],\
                                      initial_guess_fac = init_guess[j, i], Ncalc = Ncalc, toll = toll, flag_Hugp_calc_version = 1)

            print('Chocolate', Hug_match_temp.up[-1], Hug_match_temp.p[-1])

            # print(temp)
            rho_release = temp[0]
            if np.isinf(Hug_match_temp.up[-1]):
                print('match failed', rho_release, Hug_match_temp.up[-1])
                print(dlfkjd)
            else:
                print('match calculated',temp[0])
            
            # print('Do we alreadt gave it 2?', Hug_match_temp.up[-1])

            # Find the other properties on the release curve:
            temp = np.linspace(Hug_for.rho[-1], rho_release, Ncalc)
            release.calc_release(temp, Hug_for.up[-1], [Hug_for.p[-1], Hug_for.T[-1]], toll)
            #Hug_IG.calc_hugoniot_up_ideal_gas(Hug.up, [loss_arr[j,k,i,2],T0], gamma, ma)

            print('release calculated')
            # print('toad', p0,T0,release.p)
            Hug_match2 = Hugoniot_from_EOS(EOS2)
            # temp=EOS2.find_PT(p0, T0)
            # temp2=[1.0,temp[0][0],temp[1][0],2.0,T0]
            temp = np.linspace(np.amin([2 * p0, release.p[-1] * 0.9]), release.p[-1], Ncalc)
            # temp2=[release.up[-1],release.rho[-1],release.E[-1],\
                    # release.us[-1],release.T[-1]]
            # temp=[rel.p[-1]]
            # temp=np.logspace(np.log10(np.amin([2*p0,release.p[-1]*0.9])), np.log10(release.p[-1]),Ncalc)
            Hug_match2.calc_hugoniot_p(temp, [p0, T0], toll, flag_version = 1)#, toll_fail=1E-4, max_itter=100000, frac_slow=0.5,\
                                      # Nslow=100,min_frac_conv=1E-4,itter_slow=1000)#, init_guess=temp2)
            
            # Hug_match2.calc_hugoniot_p(temp,[p0,T0],1E-5)#, 
            
            # # print('oooohhh', Hug_match2.up[-1])
            # temp2=[Hug_match2.up[-1],Hug_match2.rho[-1],Hug_match2.E[-1],\
            #         Hug_match2.us[-1],Hug_match2.T[-1]]
            # temp=[release.p[-1]]
            # Hug_match2.calc_hugoniot_p(temp,[p0,T0],toll,init_guess=temp2)
            
            # print(Hug_match2.up/1E3)
            # print(Hug_match2.rho/1E3)
            # print('Tent', release.p[-1], release.up[-1])
            #temp=np.linspace(100.0,release.up[-1],Ncalc)
            #Hug_match.calc_hugoniot_up(temp,[p0,T0],toll)

            print('Hug me charon')
            # print((release.up[-1]-Hug_match2.up[-1])/release.up[-1], release.up[-1], Hug_match2.up[-1])
            if np.absolute((release.up[-1] - Hug_match2.up[-1]) / release.up[-1]) < 1E-3:
                test = 0
                print('XXXXX we have a match', i, vp_for, rho_release)
                # print('\t',release.p[-1],release.up[-1],release.rho[-1])
                # print('\t',Hug_match2.p[-1],Hug_match2.up[-1],Hug_match2.rho[-1])
                init_guess[j,i] = release.rho[-1] / Hug_for.rho[-1]
            else:
                # print('That didnt actually work', count, test, release.up[-1],Hug_match2.up[-1])
                init_guess[j, i] = init_guess[j, i] + step_base * (rand.random()-0.5)
                if init_guess[j, i] > 0.999999:
                    init_guess[j,i ] = 0.999999
                elif init_guess[j, i] < 0.02:
                    init_guess[j, i] = 0.02
#                     init_guess[k,i]=0.7
        except:
            # print('That didnt work', count)
            init_guess[j, i] = init_guess[j, i] + step_base * (rand.random()-0.5)
            if init_guess[j, i] > 0.999999:
                init_guess[j, i] = 0.999999
            elif init_guess[j, i] < 0.02:
                init_guess[j, i] = 0.02

        if count > 5:
            test = 0
            print('OK we are giving up....help!')
            init_guess[j, i] = 2
            
            return np.nan, init_guess[j, i], None
            # print(dlikjfsdjf)
            
    return release.up[-1], init_guess[j, i], release


def temp_func_calc_match_approx(Ncalc, Hug_for, EOS1, interp_hug1, interp_hug2, ppeak, toll, i):
        
    # print('vvvvvvvvvvvvvvvvvvvvvvvvvvv')
    # print(i,ppeak[i])
    #initalize hugoniot
    ppeak = ppeak[i]
    
    if np.isnan(ppeak) | (ppeak <= Hug_for.p[0]):
        release = Hugoniot_from_EOS(EOS1)
        release.p = [np.nan]
        return np.nan, np.nan, release
    
    min_rho = (interp_hug2[2](np.log10(ppeak))) # Take the minimum pressure possible
    test = 1
    count = 0

    init_guess_up = interp_hug1[0](np.log10(ppeak))

    temp = np.linspace(interp_hug1[2](np.log10(ppeak)), min_rho, Ncalc)
    release = Hugoniot_from_EOS(EOS1)
    release.calc_release(temp, interp_hug1[0](np.log10(ppeak)), [ppeak, interp_hug1[1](np.log10(ppeak))], 1E-14)

    interp_release = [interp1d(release.up, release.p, fill_value = np.nan),\
                      interp1d(release.up, release.T, fill_value = np.nan),\
                      interp1d(release.up, release.rho, fill_value = np.nan)]
        
    while (test == 1) & (count < 50):
        count += 1

        try:
            temp = fsolve(func_calc_IM, [init_guess_up], args = (interp_release, interp_hug2), xtol = toll)
            test = 0
            
        except:
            init_guess_up = 1.1 * init_guess_up
            # print('oh oh', ppeak, init_guess_up)
            
        if count > 49:
            # print('That didnt work')
            raise RuntimeError('No solution found in temp_func_calc_match_approx function')

    # print('oh happy day')
    # print(temp)
           
    temp = np.linspace(interp_hug1[2](np.log10(ppeak)), interp_release[2](temp), Ncalc)
    release = Hugoniot_from_EOS(EOS1)
    release.calc_release(temp, interp_hug1[0](np.log10(ppeak)), [ppeak, interp_hug1[1](np.log10(ppeak))], toll)

    # print('fish')
    # print(release.rho[-1])
    # print(interp_hug1[2](np.log10(ppeak)))
    init_guess = release.rho[-1]/interp_hug1[2](np.log10(ppeak))
            
    return release.up[-1], init_guess, release


def func_calc_IM(up, interp_release, interp_hug):
    phug = interp_hug[0](up)
    prel = interp_release[0](up)
    return (phug - prel) / phug


def func_calc_isentrope_p(rho, p, s, EOS):
    temp = EOS.find_ds(rho, s)

    return p - temp[1][0]

# Define a function to calculate loss:
def calc_loss_interp(ppeak_mantle_arr, p0_mantle_arr, vesc_arr, interp_ug, R = 0):
    
    # Test inputs:
    if (np.shape(ppeak_mantle_arr) != np.shape(p0_mantle_arr)) | ((np.shape(ppeak_mantle_arr) != np.shape(vesc_arr)) & (np.size(vesc_arr) != 1)):
        raise ValueError('Pressure arrays passed to interp_loss inconsistent sizes or shapes')
    
    if R == 0: # i.e., no ocean
        max_loss = 1.0
    else: # i.e., ocean
        max_loss = 2.0
    
    p0_range = [np.amin(interp_ug.grid[1]), np.amax(interp_ug.grid[1])]
    ppeak_range = [np.amin(interp_ug.grid[0]), np.amax(interp_ug.grid[0])]
    
    if np.size(ppeak_mantle_arr) == 1:
        # Check physical:
        if (p0_mantle_arr > ppeak_mantle_arr):
            raise ValueError('Unphysical point requested. Peak pressure below initial pressure.')
        
        # Deal with points out of the range:
        elif (p0_mantle_arr > p0_range[0]) & (p0_mantle_arr < p0_range[1]):

            if ppeak_mantle_arr < ppeak_range[0]:

                warnings.warn('Peak pressure below range of grid. Extrapolating.')
                temp = interp_ug((ppeak_mantle_arr,p0_mantle_arr))

                if np.sum(calculate_loss(temp, vesc = vesc_arr, R = R)) >= ((1-1E-8) * max_loss):
                    return max_loss
                
                elif np.sum(calculate_loss(temp, vesc = vesc_arr, R = R)) <= 1E-8:
                    return 0.
                
                else:
                    return np.sum(calculate_loss(temp, vesc = vesc_arr, R = R))
                
            elif ppeak_mantle_arr > ppeak_range[1]:

                temp = interp_ug((ppeak_range[1]-1E-6, p0_mantle_arr))

                if np.sum(calculate_loss(temp, vesc = vesc_arr, R = R)) >= ((1-1E-8) * max_loss):
                    return max_loss
                
                else:
                    raise ValueError('Peak pressure above range of grid without grid extending to total loss. Extend grid to higher peak pressures')
                
            else:
                ug_interp = interp_ug((ppeak_mantle_arr, p0_mantle_arr))
                return np.sum(calculate_loss(ug_interp, vesc = vesc_arr, R = R))
            
        elif (ppeak_mantle_arr > ppeak_range[0]) & (ppeak_mantle_arr < ppeak_range[1]):

            if p0_mantle_arr < p0_range[0]:
                temp = interp_ug((ppeak_mantle_arr, p0_range[0]+1E-8))

                if np.sum(calculate_loss(temp, vesc = vesc_arr, R = R)) >= ((1-1E-8) * max_loss):
                    return max_loss
                
                else:
                    raise ValueError('Initial pressure below range of grid without grid extending to total loss. Extend grid to lower initial pressures')
                
            elif p0_mantle_arr > p0_range[1]:

                temp = interp_ug((ppeak_mantle_arr, p0_range[1]-1E-8))

                if np.sum(calculate_loss(temp, vesc = vesc_arr, R = R)) >= ((1-1E-8) * max_loss):
                    return max_loss
                
                else:
                    raise ValueError('Initial pressure above range of grid without grid extending to total loss. Extend grid to higher initial pressures')
                
            else:
                ug_interp = interp_ug((ppeak_mantle_arr, p0_mantle_arr))
                return np.sum(calculate_loss(ug_interp, vesc = vesc_arr, R = R))
            
        else:
            raise ValueError('Both peak and inital pressure off the grid. Extend the grid')
                             
    else:

        loss = np.ones(np.size(p0_mantle_arr)) * np.nan
        
        for i in np.arange(np.size(p0_mantle_arr)):
            # Extract the values:
            p0_mantle = p0_mantle_arr[i]
            ppeak_mantle = ppeak_mantle_arr[i]

            if np.size(vesc_arr) == 1:
                vesc = vesc_arr
            else:
                vesc = vesc_arr[i]
                
            if (p0_mantle > ppeak_mantle):
                raise ValueError('Unphysical point requested. Peak pressure below initial pressure.')
            
            # Deal with points out of the range:
            elif (p0_mantle > p0_range[0]) & (p0_mantle < p0_range[1]):

                if ppeak_mantle < ppeak_range[0]:
                    warnings.warn('Peak pressure below range of grid. Extend grid to lower peak pressures')
                    temp = interp_ug((ppeak_mantle, p0_mantle))

                    if np.sum(calculate_loss(temp, vesc = vesc, R = R)) >= (0.99 * max_loss):
                        loss[i] = max_loss

                    elif np.sum(calculate_loss(temp, vesc = vesc, R = R)) <= 0.0:
                        loss[i] = 0.

                    else:
                        loss[i] = np.sum(calculate_loss(temp, vesc = vesc, R = R))

                elif ppeak_mantle > ppeak_range[1]:
                    temp = interp_ug((ppeak_range[1]-1E-6, p0_mantle))

                    if np.sum(calculate_loss(temp, vesc = vesc, R = R)) >= (0.99 * max_loss):
                        loss[i] = max_loss

                    else:
                        raise ValueError('Peak pressure above range of grid without grid extending to total loss. Extend grid to higher peak pressures')
                    
                else:
                    ug_interp = interp_ug((ppeak_mantle, p0_mantle))
                    loss[i] = np.sum(calculate_loss(ug_interp, vesc = vesc, R = R))

            elif (ppeak_mantle > ppeak_range[0]) & (ppeak_mantle < ppeak_range[1]):

                if p0_mantle < p0_range[0]:
                    temp = interp_ug((ppeak_mantle, p0_range[0]+1E-6))

                    if np.sum(calculate_loss(temp, vesc = vesc, R = R)) >= (0.99 * max_loss):
                        loss[i] = max_loss

                    else:
                        raise ValueError('Initial pressure below range of grid without grid extending to total loss. Extend grid to lower initial pressures')
                    
                elif p0_mantle > p0_range[1]:
                    temp = interp_ug((ppeak_mantle, p0_range[1]-1E-6))

                    if np.sum(calculate_loss(temp, vesc = vesc, R = R)) >= (0.99 * max_loss):
                        loss[i] = max_loss

                    else:
                        raise ValueError('Initial pressure above range of grid without grid extending to total loss. Extend grid to higher initial pressures')
                    
                else:
                    ug_interp = interp_ug((ppeak_mantle, p0_mantle))
                    loss[i] = np.sum(calculate_loss(ug_interp, vesc = vesc, R = R))

            else:
                raise ValueError('Both peak and inital pressure off the grid. Extend the grid')

        return loss
    
# Define a function to calculate loss:
def calc_loss_interp_releasep(prelease_arr, p0_mantle_arr, vesc_arr, interp_ug, prelease_range, p0_range, R = 0):
    
    # Test inputs:
    if (np.shape(prelease_arr) != np.shape(p0_mantle_arr)) | ((np.shape(prelease_arr) != np.shape(vesc_arr)) & (np.size(vesc_arr) != 1)):
        raise ValueError('Pressure arrays passed to interp_loss inconsistent sizes or shapes')
    
    if R == 0:
        max_loss = 1.0
    else:
        max_loss = 2.0
    
    if np.size(prelease_arr) == 1:
        # Deal with points out of the range:
        if (p0_mantle_arr > p0_range[0]) & (p0_mantle_arr < p0_range[1]):
            if prelease_arr < prelease_range[0]:
#                 raise ValueError('Peak pressure below range of grid. Extend grid to lower peak pressures')
                warnings.warn('Peak pressure below range of grid. Extrapolating.')
                temp = interp_ug((prelease_arr,p0_mantle_arr))

                if np.sum(calculate_loss(temp, vesc = vesc_arr, R = R)) >= ((1-1E-8) * max_loss):
                    return max_loss
                
                elif np.sum(calculate_loss(temp, vesc = vesc_arr, R = R)) <= 1E-8:
                    return 0.
                
                else:
                    return np.sum(calculate_loss(temp, vesc = vesc_arr, R = R))
                
            elif prelease_arr > prelease_range[1]:
                temp = interp_ug((prelease_range[1]-1E-6, p0_mantle_arr))

                if np.sum(calculate_loss(temp, vesc = vesc_arr, R = R)) >= ((1-1E-8) * max_loss):
                    return max_loss
                
                else:
                    raise ValueError('Peak pressure above range of grid without grid extending to total loss. Extend grid to higher peak pressures')
                
            else:
                ug_interp = interp_ug((prelease_arr, p0_mantle_arr))
                return np.sum(calculate_loss(ug_interp, vesc = vesc_arr, R = R))
            
        elif (prelease_arr > prelease_range[0]) & (prelease_arr < prelease_range[1]):
            if p0_mantle_arr < p0_range[0]:
                temp = interp_ug((prelease_arr, p0_range[0]+1E-8))

                if np.sum(calculate_loss(temp, vesc = vesc_arr, R = R)) >= ((1-1E-8) * max_loss):
                    return max_loss
                
                else:
                    raise ValueError('Initial pressure below range of grid without grid extending to total loss. Extend grid to lower initial pressures')
                
            elif p0_mantle_arr > p0_range[1]:
                temp = interp_ug((prelease_arr, p0_range[1]-1E-8))

                if np.sum(calculate_loss(temp, vesc = vesc_arr, R = R)) >= ((1-1E-8) * max_loss):
                    return max_loss
                
                else:
                    raise ValueError('Initial pressure above range of grid without grid extending to total loss. Extend grid to higher initial pressures')
                
            else:
                ug_interp = interp_ug((prelease_arr, p0_mantle_arr))
                return np.sum(calculate_loss(ug_interp, vesc = vesc_arr, R = R))
            
        else:
            raise ValueError('Both peak and inital pressure off the grid. Extend the grid')
                                    
    else:
        loss = np.ones(np.size(p0_mantle_arr)) * np.nan
        
        for i in np.arange(np.size(p0_mantle_arr)):
            # Extract the values:
            p0_mantle = p0_mantle_arr[i]
            ppeak_mantle = prelease_arr[i]

            if np.size(vesc_arr) == 1:
                vesc = vesc_arr
            else:
                vesc = vesc_arr[i]
                
            # Deal with points out of the range:
            if (p0_mantle > p0_range[0]) & (p0_mantle < p0_range[1]):
                if ppeak_mantle < prelease_range[0]:
#                     raise ValueError('Peak pressure below range of grid. Extend grid to lower peak pressures')
                    warnings.warn('Peak pressure below range of grid. Extend grid to lower peak pressures')
                    temp = interp_ug((ppeak_mantle,p0_mantle))

                    if np.sum(calculate_loss(temp, vesc = vesc, R = R)) >= (0.99 * max_loss):
                        loss[i] = max_loss

                    elif np.sum(calculate_loss(temp, vesc = vesc, R = R)) <= 0.0:
                        loss[i] = 0.

                    else:
                        loss[i] = np.sum(calculate_loss(temp, vesc = vesc, R = R))

                elif ppeak_mantle > prelease_range[1]:
                    temp = interp_ug((prelease_range[1]-1E-6, p0_mantle))

                    if np.sum(calculate_loss(temp, vesc = vesc, R = R)) >= (0.99 * max_loss):
                        loss[i] = max_loss

                    else:
                        raise ValueError('Peak pressure above range of grid without grid extending to total loss. Extend grid to higher peak pressures')
                    
                else:
                    ug_interp = interp_ug((ppeak_mantle, p0_mantle))
                    loss[i] = np.sum(calculate_loss(ug_interp, vesc = vesc, R = R))

            elif (ppeak_mantle > prelease_range[0]) & (ppeak_mantle < prelease_range[1]):
                if p0_mantle < p0_range[0]:
                    temp = interp_ug((ppeak_mantle, p0_range[0]+1E-6))

                    if np.sum(calculate_loss(temp, vesc = vesc, R = R)) >= (0.99 * max_loss):
                        loss[i] = max_loss

                    else:
                        raise ValueError('Initial pressure below range of grid without grid extending to total loss. Extend grid to lower initial pressures')
                    
                elif p0_mantle > p0_range[1]:
                    temp = interp_ug((ppeak_mantle, p0_range[1]-1E-6))

                    if np.sum(calculate_loss(temp, vesc = vesc, R = R)) >= (0.99 * max_loss):
                        loss[i] = max_loss

                    else:
                        raise ValueError('Initial pressure above range of grid without grid extending to total loss. Extend grid to higher initial pressures')
                    
                else:
                    ug_interp = interp_ug((ppeak_mantle, p0_mantle))
                    loss[i] = np.sum(calculate_loss(ug_interp, vesc = vesc, R = R))

            else:
                raise ValueError('Both peak and inital pressure off the grid. Extend the grid')

        return loss