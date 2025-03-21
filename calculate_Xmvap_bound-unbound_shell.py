import numpy as np
import sys
import os
import glob
from sklearn.cluster import KMeans

import csv
import pandas as pd

import swiftsimio as sw

import warnings
import re

# Package to use wildcards: 
import fnmatch

from near_vs_far_functions import *

# Constants:
R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg m^-3
G = 6.67408e-11  # m^3 kg^-1 s^-2
vesc_Earth = 11.186E3

#############################################
def calculate_global_loss_near_far2(folder_path,
                                   max_remnant,
                                   npt = 1e9,
                                   m_tot = 1,
                                   output_id = 1,
                                   atmos_id = 200,
                                   near_far_idx = None
):

    """ Calculates the global distribution of atmosphere loss fraction from a given impact, split into loss from the near- and far-field.

        Note: requires the bound_mass function.

        Args:
            folder_path: path to folder containing output files
            max_remnant (int, optional): Maximum numer of remnants you would like to search. Defaults to 1. Set to 1 will be very fast,
                but can not be sure that's the largest one, since searching will stop once the code find one remnant group. Set to 5 or 10 
                would be safe if you have multiple remnant groups
            npt: number of particles in the target
            m_tot (float, optional): total mass of target and impactor. Defaults to 1 (Earth masses)
            output_id (int): id of the remnant you would like to output. Output: remnant mass, remnant mass ratio (m_rem/m_initial_total),
                iron fraction of the remnant, water mass in the remnant, atmosphere mass in the remnant (mass of iron from the target, mass of si from the target, 
                mass of iron from the impactor, mass of si from the impactor). 
            atmos_id: specify the material ID for the atmosphere EoS used. Defaults to 200.
            near_far_idx: path to .npz file of particles in the near- and far-field, returned by the calc_near_far_field function.
    """

    # Filter out only the snapshot files:
    snapshots = sorted([file for file in os.listdir(folder_path) if 'snapshot' in file and '.xmf' not in file],
                      key = lambda x: int(re.search(r'\d+', x).group()))
    no_snapshots = len(snapshots)

    for k, file_name in enumerate([snapshots[0], snapshots[-1]]):
        
        # Find the particle ids of bound atmosphere particles:
        (time, 
        npt,
        bound,
        id_bound,
        id_unbound, 
        m, 
        z, 
        m_si, 
        z_si, 
        si_tar, 
        si_imp, 
        m_fe, 
        z_fe, 
        fe_tar, 
        fe_imp, 
        m_water, 
        z_water, 
        water_tar, 
        water_imp, 
        m_atmos, 
        z_atmos, 
        atmos_tar, 
        atmos_imp) = bound_mass(os.path.join(folder_path, file_name), 
                                    max_remnant = max_remnant, 
                                    EOS = 'iron',
                                    atmos_id = atmos_id,
                                    npt = npt, 
                                    quickcheck = 1, 
                                    m_tot = m_tot, 
                                    verbose = 0,
                                    output_id = output_id)

        ####################################################################################

        # Load data:
        data = sw.load(os.path.join(folder_path, file_name))
        box_mid = 0.5*data.metadata.boxsize[0]
        data.gas.coordinates.convert_to_mks()

        id_body = 2e8

        id = data.gas.particle_ids
        sort = np.argsort(id)
        id = id[sort]

        mat_id = data.gas.material_ids.value
        mat_id = mat_id[sort]
        mat_id[npt<=id] = mat_id[npt<=id] + id_body
        
        m = np.array(data.gas.masses)
        m = m[sort]
            
        pos = data.gas.coordinates - box_mid
        pos = np.array(pos)
        pos = pos[sort]

        time = int(data.metadata.t)
        
        ####################################################################################

        # If snapshot_0000.hdf5, calculate the radii of the target and impactor, and initialise some arrays:
        if file_name == snapshots[0]:

            # Index particles from the target and impactor separately based on their material IDs:
            ind_tar = np.where(np.logical_or(mat_id == 400, mat_id == 401))[0]
            pos_tar_solid = pos[ind_tar]

            # Centre the position and velocity arrays on the centre of mass of the target and impactor, respectively:
            pos_com_tar_solid = np.sum(pos_tar_solid * m[ind_tar, np.newaxis], axis = 0) / np.sum(m[ind_tar])
            pos_tar_solid -= pos_com_tar_solid

            # Index the atmosphere particles from the mass array:
            temp = np.load(near_far_idx)
            
            ind_atmos = np.where(mat_id == 200)[0]
            ind_atmos_near = np.where(temp['nf_MC'] == 1)[0]
            ind_atmos_far = np.where(temp['nf_MC'] == 0)[0]

            m_atmos = m[ind_atmos]
            m_atmos_near_MC = m_atmos[ind_atmos_near]
            m_atmos_far_MC = m_atmos[ind_atmos_far]

            # Select the entire atmosphere to define as the shell:
            id_tar_atmos_shell = id[ind_atmos]

            # Find the indices of atmosphere particles from their ID numbers:
            tar_atmos_shell_from_id = np.in1d(id_tar_atmos_shell, id)
            ind = id.searchsorted(id_tar_atmos_shell)

            # Select the particles from the target which lie within this shell:
            pos_tar_atmos_base = pos[ind]
            pos_tar_atmos_base[tar_atmos_shell_from_id == False] = 0
            pos_tar_atmos_base -= pos_com_tar_solid

            # Calculate latitude and longitude:
            latitude_tar = np.arctan(pos_tar_atmos_base[:,2]/np.sqrt(pos_tar_atmos_base[:,0]**2 + pos_tar_atmos_base[:,1]**2))
            longitude_tar = np.arccos(pos_tar_atmos_base[:,0]/np.sqrt(pos_tar_atmos_base[:,0]**2 + pos_tar_atmos_base[:,1]**2)) * np.sign(pos_tar_atmos_base[:,1])

            bound_id_tar_atmos_shell = id_tar_atmos_shell

            ####################################################################################

            # Define the grid boundaries:
            lat_min, lat_max = -90, 90
            lon_min, lon_max = -180, 180
            d = 10  # side length of grid cells in degrees

            # Calculate the number of grid cells in each direction:
            n_lat = int(np.ceil((lat_max - lat_min) / d))
            n_lon = int(np.ceil((lon_max - lon_min) / d))

            # Initialize arrays to store the atmosphere mass per grid cell:
            initial_cell_atmos_mass = np.zeros((n_lat, n_lon))
            initial_cell_atmos_mass_near_MC = np.zeros((n_lat, n_lon))
            initial_cell_atmos_mass_far_MC = np.zeros((n_lat, n_lon))
            lat_vals = np.zeros((n_lat, n_lon))
            lon_vals = np.zeros((n_lat, n_lon))

            # Loop over each grid cell:
            for i in range(n_lat):

                for j in range(n_lon):

                    # Calculate the latitude and longitude ranges for this grid cell:
                    lat_range = [lat_min + i*d, lat_min + (i+1)*d]
                    lon_range = [lon_min + j*d, lon_min + (j+1)*d]

                    # Store the first value of lat_range and lon_range for this grid cell:
                    lat_vals[i, j] = lat_range[0] + d/2
                    lon_vals[i, j] = lon_range[0] + d/2

                    # Find the indices of data points within this grid cell:
                    mask_tar = (latitude_tar*180/np.pi >= lat_range[0]) & (latitude_tar*180/np.pi < lat_range[1]) & (longitude_tar*180/np.pi >= lon_range[0]) & (longitude_tar*180/np.pi < lon_range[1])
                    idx_tar = np.where(mask_tar)[0]

                    mask_tar_near = (latitude_tar[ind_atmos_near]*180/np.pi >= lat_range[0]) & (latitude_tar[ind_atmos_near]*180/np.pi < lat_range[1]) & (longitude_tar[ind_atmos_near]*180/np.pi >= lon_range[0]) & (longitude_tar[ind_atmos_near]*180/np.pi < lon_range[1])
                    idx_tar_near = np.where(mask_tar_near)[0]

                    mask_tar_far = (latitude_tar[ind_atmos_far]*180/np.pi >= lat_range[0]) & (latitude_tar[ind_atmos_far]*180/np.pi < lat_range[1]) & (longitude_tar[ind_atmos_far]*180/np.pi >= lon_range[0]) & (longitude_tar[ind_atmos_far]*180/np.pi < lon_range[1])
                    idx_tar_far = np.where(mask_tar_far)[0]

                    # Calculate the initial atmosphere mass within this grid cell:
                    if len(idx_tar) > 0:
                        initial_cell_atmos_mass[i, j] = np.sum(m_atmos[idx_tar])
                    else:
                        initial_cell_atmos_mass[i, j] = 0 # np.nan

                    if len(idx_tar_near) > 0:
                        initial_cell_atmos_mass_near_MC[i, j] = np.sum(m_atmos_near_MC[idx_tar_near])
                    else:
                        initial_cell_atmos_mass_near_MC[i, j] = 0

                    if len(idx_tar_far) > 0:
                        initial_cell_atmos_mass_far_MC[i, j] = np.sum(m_atmos_far_MC[idx_tar_far])
                    else:
                        initial_cell_atmos_mass_far_MC[i, j] = 0
            
            total_bound_atmos_mass_time = np.ones((len(ind_atmos), no_snapshots)) * np.NaN
            total_bound_atmos_mass_time_near_MC = np.ones((len(ind_atmos_near), no_snapshots)) * np.NaN
            total_bound_atmos_mass_time_far_MC = np.ones((len(ind_atmos_far), no_snapshots)) * np.NaN

            cell_atmos_mass = np.zeros((n_lat, n_lon, no_snapshots))
            cell_atmos_mass_near_MC = np.zeros((n_lat, n_lon, no_snapshots))
            cell_atmos_mass_far_MC = np.zeros((n_lat, n_lon, no_snapshots))

            bound_cell_atmos_mass_time = np.zeros((n_lat, n_lon, no_snapshots))
            bound_cell_atmos_mass_time_near_MC = np.zeros((n_lat, n_lon, no_snapshots))
            bound_cell_atmos_mass_time_far_MC = np.zeros((n_lat, n_lon, no_snapshots))
            
            t = []

        ####################################################################################

        else:

            # Find the indices of the bound atmosphere particles from their ID numbers to deal with deleted particles:
            bound_tar_atmos_from_id = np.in1d(ind_atmos, id_bound)
            ind = id.searchsorted(ind_atmos)

            m_bound_atmos = m[ind]
            m_bound_atmos[bound_tar_atmos_from_id == False] = 0
            total_bound_atmos_mass_time[:, k] = m_bound_atmos

            bound_tar_atmos_from_id_near = np.in1d(ind_atmos[ind_atmos_near], id_bound)
            ind_near = id.searchsorted(ind_atmos[ind_atmos_near])

            m_bound_atmos_near = m[ind_near]
            m_bound_atmos_near[bound_tar_atmos_from_id_near == False] = 0
            total_bound_atmos_mass_time_near_MC[:, k] = m_bound_atmos_near

            bound_tar_atmos_from_id_far = np.in1d(ind_atmos[ind_atmos_far], id_bound)
            ind_far = id.searchsorted(ind_atmos[ind_atmos_far])

            m_bound_atmos_far = m[ind_far]
            m_bound_atmos_far[bound_tar_atmos_from_id_far == False] = 0
            total_bound_atmos_mass_time_far_MC[:, k] = m_bound_atmos_far

            t.append(time)

            # Loop over each grid cell:
            for i in range(n_lat):

                for j in range(n_lon):

                    # Calculate the latitude and longitude ranges for this grid cell:
                    lat_range = [lat_min + i*d, lat_min + (i+1)*d]
                    lon_range = [lon_min + j*d, lon_min + (j+1)*d]

                    # Store the first value of lat_range and lon_range for this grid cell:
                    lat_vals[i, j] = lat_range[0] + d/2
                    lon_vals[i, j] = lon_range[0] + d/2

                    # Find the indices of data points within this grid cell:
                    mask_tar = (latitude_tar*180/np.pi >= lat_range[0]) & (latitude_tar*180/np.pi < lat_range[1]) & (longitude_tar*180/np.pi >= lon_range[0]) & (longitude_tar*180/np.pi < lon_range[1])
                    idx_tar = np.where(mask_tar)[0]

                    mask_tar_near = (latitude_tar[ind_atmos_near]*180/np.pi >= lat_range[0]) & (latitude_tar[ind_atmos_near]*180/np.pi < lat_range[1]) & (longitude_tar[ind_atmos_near]*180/np.pi >= lon_range[0]) & (longitude_tar[ind_atmos_near]*180/np.pi < lon_range[1])
                    idx_tar_near = np.where(mask_tar_near)[0]

                    mask_tar_far = (latitude_tar[ind_atmos_far]*180/np.pi >= lat_range[0]) & (latitude_tar[ind_atmos_far]*180/np.pi < lat_range[1]) & (longitude_tar[ind_atmos_far]*180/np.pi >= lon_range[0]) & (longitude_tar[ind_atmos_far]*180/np.pi < lon_range[1])
                    idx_tar_far = np.where(mask_tar_far)[0]

                    # Calculate the bound atmosphere mass within this grid cell, and calculate the ratio of bound particles to that in the initial snapshot:
                    if len(idx_tar) > 0:
                        cell_atmos_mass[i, j, k] = np.sum(total_bound_atmos_mass_time[idx_tar, k])
                        bound_cell_atmos_mass_time[i, j, k] = cell_atmos_mass[i, j, k] / initial_cell_atmos_mass[i, j]
                    else:
                        cell_atmos_mass[i, j, k] = np.nan # np.nan
                        bound_cell_atmos_mass_time[i, j, k] = 1

                    if len(idx_tar_near) > 0:
                        cell_atmos_mass_near_MC[i, j, k] = np.sum(total_bound_atmos_mass_time_near_MC[idx_tar_near, k])
                        bound_cell_atmos_mass_time_near_MC[i, j, k] = cell_atmos_mass_near_MC[i, j, k] / initial_cell_atmos_mass_near_MC[i, j]
                    else:
                        cell_atmos_mass_near_MC[i, j, k] = np.nan
                        bound_cell_atmos_mass_time_near_MC[i, j, k] = np.nan # for mantle vapourisation comparison only. Set to 1 otherwise

                    if len(idx_tar_far) > 0:
                        cell_atmos_mass_far_MC[i, j, k] = np.sum(total_bound_atmos_mass_time_far_MC[idx_tar_far, k])
                        bound_cell_atmos_mass_time_far_MC[i, j, k] = cell_atmos_mass_far_MC[i, j, k] / initial_cell_atmos_mass_far_MC[i, j]
                    else:
                        cell_atmos_mass_far_MC[i, j, k] = np.nan
                        bound_cell_atmos_mass_time_far_MC[i, j, k] = np.nan # for 1D comparison only. Set to 1 otherwise

    return lat_vals, lon_vals, initial_cell_atmos_mass, initial_cell_atmos_mass_near_MC, initial_cell_atmos_mass_far_MC, bound_cell_atmos_mass_time, bound_cell_atmos_mass_time_near_MC, bound_cell_atmos_mass_time_far_MC

#############################################
def calculate_mantle_shell_vap(folder_path,
                               eos_struct_path,
                               for_eos_path,
                               npt = 1e9,
):

    """ Calculates the global distribution of mantle loss fraction from a given impact.

        Note: requires the bound_mass function.

        Args:
            folder_path: path to folder containing output files
            eos_struct_path: path to folder containing functions to calculate a particle's position on the phase diagram
            for_eos_path: path to the forsterite EoS table
            npt: number of particles in the target
            
    """
    
    # Change path to location of the Gadget_EOS_structure2.py file:
    sys.path.append(eos_struct_path)
    import Gadget_EOS_structure2 as EOS_structure

    # Set up structure:
    EOS_curve_fors = EOS_structure.Gadget_EOS()
    EOS_curve_fors.read_Dome(for_eos_path)

    # Filter out only the snapshot files:
    snapshots = sorted([file for file in os.listdir(folder_path) if 'snapshot' in file and '.xmf' not in file],
                      key = lambda x: int(re.search(r'\d+', x).group()))
    no_snapshots = len(snapshots)

    for k, file_name in enumerate([snapshots[0], snapshots[-1]]):
        
        # Load data:
        data = sw.load(os.path.join(folder_path, file_name))
        box_mid = 0.5*data.metadata.boxsize[0]
        data.gas.coordinates.convert_to_mks()

        id_body = 2e8

        id = data.gas.particle_ids
        sort = np.argsort(id)
        id = id[sort]

        mat_id = data.gas.material_ids.value
        mat_id = mat_id[sort]
        mat_id[npt<=id] = mat_id[npt<=id] + id_body

        m = np.array(data.gas.masses)
        m = m[sort]

        pos = data.gas.coordinates - box_mid
        pos = np.array(pos)
        pos = pos[sort]
        
        data.gas.densities.convert_to_mks()
        rho = np.array(data.gas.densities)
        rho = rho[sort]

        data.gas.internal_energies.convert_to_mks()
        u = np.array(data.gas.internal_energies)
        u = u[sort]

        data.gas.pressures.convert_to_mks()
        p = np.array(data.gas.pressures)
        p = p[sort]

        if file_name == snapshots[0]:
                
            # Index particles from the target and impactor separately based on their material IDs:
            ind_tar = np.where(np.logical_or(mat_id == 400, mat_id == 401))[0]
            pos_tar_solid = pos[ind_tar]

            # Centre the position and velocity arrays on the centre of mass of the target and impactor, respectively:
            pos_com_tar_solid = np.sum(pos_tar_solid * m[ind_tar, np.newaxis], axis = 0) / np.sum(m[ind_tar])
            pos_tar_solid -= pos_com_tar_solid

            # Calculate the target radius:
            r_tar = np.sqrt(pos_tar_solid[:,0]**2 + pos_tar_solid[:,1]**2 + pos_tar_solid[:,2]**2)

            # Select the top 20% of particles:
            tar_shell_0d2 = np.where(r_tar >= np.percentile(r_tar, 80))[0]
            id_tar_shell_0d2 = id[ind_tar[tar_shell_0d2]] # Keep these indices constant for each snapshot

            # Find the indices of the surface particles from their ID numbers:
            tar_shell_from_id_0d2 = np.in1d(id_tar_shell_0d2, id)
            ind_tar_0d2 = id.searchsorted(id_tar_shell_0d2)

            pos_tar_surface_0d2 = pos[ind_tar_0d2]
            pos_tar_surface_0d2[tar_shell_from_id_0d2 == False] = 0
            pos_tar_surface_0d2 -= pos_com_tar_solid

            r_tar_surface_0d2 = np.sqrt(pos_tar_surface_0d2[:,0]**2 + pos_tar_surface_0d2[:,1]**2 + pos_tar_surface_0d2[:,2]**2)

            p_tar_surface_0d2 = p[ind_tar_0d2] 
            p_tar_surface_0d2[tar_shell_from_id_0d2 == False] = 0

            # Calculate latitude and longitude:
            latitude_tar_0d2 = np.arctan(pos_tar_surface_0d2[:,2]/np.sqrt(pos_tar_surface_0d2[:,0]**2 + pos_tar_surface_0d2[:,1]**2))
            longitude_tar_0d2 = np.arccos(pos_tar_surface_0d2[:,0]/np.sqrt(pos_tar_surface_0d2[:,0]**2 + pos_tar_surface_0d2[:,1]**2)) * np.sign(pos_tar_surface_0d2[:,1])

            ##############################################################################
            # Use k-means clustering to define the surface:
            data = list(zip(r_tar_surface_0d2, p_tar_surface_0d2))

            # Fit KMeans and get labels:
            kmeans = KMeans(n_clusters = 3)
            kmeans.fit(data)
            labels = kmeans.labels_

            # Calculate the mean radius and mean pressure for each cluster:
            mean_radii = []
            mean_pressures = []
            for i in range(3): 
                mean_radii.append(np.mean(r_tar_surface_0d2[labels == i]))
                mean_pressures.append(np.mean(p_tar_surface_0d2[labels == i]))

            # Determine the correct cluster labels based on mean radius and pressure:
            # Sort by mean radius ascending (smallest to largest):
            sorted_by_radius = np.argsort(mean_radii)

            # To distinguish between middle and inner clusters, consider the mean pressure:
            if mean_pressures[sorted_by_radius[0]] > mean_pressures[sorted_by_radius[1]]:
                idx_inner = np.where(labels == sorted_by_radius[0])[0]
                idx_middle = np.where(labels == sorted_by_radius[1])[0]
            else:
                idx_inner = np.where(labels == sorted_by_radius[1])[0]
                idx_middle = np.where(labels == sorted_by_radius[0])[0]

            idx_outer = np.where(labels == sorted_by_radius[2])[0]

            ##############################################################################
            # Now discard the outer particles and take 3% of the upper mantle to define as the surface:
            ind_tar_reduced = np.setdiff1d(ind_tar, ind_tar_0d2[idx_outer])
            pos_tar = pos[ind_tar_reduced]

            # Centre the arrays on the centre of mass of the target and impactor, respectively:
            pos_com_tar = np.sum(pos_tar * m[ind_tar_reduced, np.newaxis], axis = 0) / np.sum(m[ind_tar_reduced])
            pos_tar -= pos_com_tar

            # Calculate the radii of the planets and sort the array:
            r_tar = np.sqrt(pos_tar[:,0]**2 + pos_tar[:,1]**2 + pos_tar[:,2]**2)

            tar_shell_0d03 = np.where(r_tar >= np.percentile(r_tar, 97))[0] 
            id_tar_shell_0d03 = id[ind_tar_reduced[tar_shell_0d03]] # Keep these indices constant for each snapshot

            # Find the indices of the surface particles from their ID numbers:
            tar_shell_from_id_0d03 = np.in1d(id_tar_shell_0d03, id)
            ind_tar_0d03 = id.searchsorted(id_tar_shell_0d03)

            pos_tar_surface_0d03 = pos[ind_tar_0d03]
            pos_tar_surface_0d03[tar_shell_from_id_0d03 == False] = 0
            pos_tar_surface_0d03 -= pos_com_tar

            r_tar_surface_0d03 = np.sqrt(pos_tar_surface_0d03[:,0]**2 + pos_tar_surface_0d03[:,1]**2 + pos_tar_surface_0d03[:,2]**2)

            p_tar_surface_0d03 = p[ind_tar_0d03] 
            p_tar_surface_0d03[tar_shell_from_id_0d03 == False] = 0

            # Calculate latitude and longitude:
            latitude_tar_0d03 = np.arctan(pos_tar_surface_0d03[:,2]/np.sqrt(pos_tar_surface_0d03[:,0]**2 + pos_tar_surface_0d03[:,1]**2))
            longitude_tar_0d03 = np.arccos(pos_tar_surface_0d03[:,0]/np.sqrt(pos_tar_surface_0d03[:,0]**2 + pos_tar_surface_0d03[:,1]**2)) * np.sign(pos_tar_surface_0d03[:,1])

            ##############################################################################
            m_mantle = m[ind_tar_0d03]

            # Define the grid boundaries:
            lat_min, lat_max = -90, 90
            lon_min, lon_max = -180, 180
            d = 10  # side length of grid cells in degrees

            # Calculate the number of grid cells in each direction:
            n_lat = int(np.ceil((lat_max - lat_min) / d))
            n_lon = int(np.ceil((lon_max - lon_min) / d))

            # Initialize arrays to store the mantle mass per grid cell:
            initial_cell_mantle_mass = np.zeros((n_lat, n_lon))
            lat_vals = np.zeros((n_lat, n_lon))
            lon_vals = np.zeros((n_lat, n_lon))

            # Loop over each grid cell:
            for i in range(n_lat):

                for j in range(n_lon):

                    # Calculate the latitude and longitude ranges for this grid cell:
                    lat_range = [lat_min + i*d, lat_min + (i+1)*d]
                    lon_range = [lon_min + j*d, lon_min + (j+1)*d]

                    # Store the first value of lat_range and lon_range for this grid cell:
                    lat_vals[i, j] = lat_range[0] + d/2
                    lon_vals[i, j] = lon_range[0] + d/2

                    # Find the indices of data points within this grid cell:
                    mask_tar = (latitude_tar_0d03*180/np.pi >= lat_range[0]) & (latitude_tar_0d03*180/np.pi < lat_range[1]) & (longitude_tar_0d03*180/np.pi >= lon_range[0]) & (longitude_tar_0d03*180/np.pi < lon_range[1])
                    idx_tar = np.where(mask_tar)[0]

                    # Calculate the initial mantle mass within this grid cell:
                    if len(idx_tar) > 0:
                        initial_cell_mantle_mass[i, j] = np.sum(m_mantle[idx_tar])
                    else:
                        initial_cell_mantle_mass[i, j] = np.nan

        ####################################################################################

        else:
            
            # Find the indices of the mantle particles from their ID numbers to deal with deleted particles:
            tar_mantle_shell_from_id = np.in1d(id_tar_shell_0d03, id)
            ind = id.searchsorted(id_tar_shell_0d03)
        
            rho_mantle = rho[ind] 
            rho_mantle[tar_mantle_shell_from_id == False] = 0

            u_mantle = u[ind] 
            u_mantle[tar_mantle_shell_from_id == False] = 0

            p_mantle = p[ind] 
            p_mantle[tar_mantle_shell_from_id == False] = 0
            
            mat_id_mantle = mat_id[ind]
            mat_id_mantle[tar_mantle_shell_from_id == False] = 0
            
            # Calculate the specific entropies of target mantle particles:
            sp_S = woma.eos.eos.A1_s_u_rho(u_mantle, rho_mantle, mat_id_mantle)

            # Initialize arrays to store the mantle mass per grid cell:
            mantle_vap_frac = np.zeros((n_lat, n_lon))

            # Loop over each grid cell:
            for i in range(n_lat):

                for j in range(n_lon):

                    # Calculate the latitude and longitude ranges for this grid cell:
                    lat_range = [lat_min + i*d, lat_min + (i+1)*d]
                    lon_range = [lon_min + j*d, lon_min + (j+1)*d]
                    
                    # Store the first value of lat_range and lon_range for this grid cell:
                    lat_vals[i, j] = lat_range[0] + d/2
                    lon_vals[i, j] = lon_range[0] + d/2
                    
                    # Find the indices of data points within this grid cell:
                    mask_tar = (latitude_tar_0d03*180/np.pi >= lat_range[0]) & (latitude_tar_0d03*180/np.pi < lat_range[1]) & (longitude_tar_0d03*180/np.pi >= lon_range[0]) & (longitude_tar_0d03*180/np.pi < lon_range[1])
                    idx_tar = np.where(mask_tar)[0]
                    
                    # Calculate the bound mantle mass within this grid cell, and calculate the ratio of bound particles to that in the initial snapshot:
                    if len(idx_tar) > 0:
                        v_frac = 0
                        
                        for k in range(len(idx_tar)):
                            s_tot_tmp = float(sp_S[idx_tar[k]])
                            P_tmp = float(p_mantle[idx_tar[k]])

                            dome_flag = EOS_curve_fors.calc_Dome(s_tot_tmp, P_tmp)[1]

                            if dome_flag == 0:   #we are in/on dome
                                v_frac += 1
                            elif dome_flag == 1: #above dome on vapour side
                                v_frac += 1
                            else:
                                v_frac += 0
                                
                        mantle_vap_frac[i, j] = np.sum(v_frac) / len(idx_tar)
                            
                    else:
                        mantle_vap_frac[i, j] = np.nan

    return lat_vals, lon_vals, initial_cell_mantle_mass, mantle_vap_frac


#############################################
# Loop:
folder = ''
excluded_dirs = {''}

count = -1

# Loop through the directory of impacts:
for root, dirs, files in os.walk(folder): 
    count += 1

    if count >= 0:
        
        if 'Earth' in root:
            
            if any(dir_name.endswith('gamma') for dir_name in dirs):
                
                for gamma_dir in dirs:
                    
                    if gamma_dir.endswith('gamma'):
                        gamma_path = os.path.join(root, gamma_dir)

                        for impact in os.listdir(gamma_path):
                            try:
                                if impact.endswith('_HVOF'):
                                    continue

                                parent_dir = os.path.basename(os.path.dirname(gamma_path))
                                
                                if parent_dir.endswith('_HOF'):
                                    parent_dir = parent_dir.rsplit('_HOF', 1)[0]
                                else:
                                    parent_dir = parent_dir
                                
                                print(impact)
                                temp_dir = f'{os.path.join(gamma_path, impact)}/output_snip_0d5h_3h_10dt_snap_10h_100dt/'
                                
                                if not os.path.exists(temp_dir):
                                    print(f"Directory not found: {temp_dir}. Skipping...")
                                    continue

                                npt = int(impact.split('npt')[1].split('_impactor')[0].replace('d', '.'))
                                M_t = float(impact.split('target_')[1].split('Earth')[0].replace('d', '.'))
                                M_i = float(impact.split('impactor_')[1].split('Earth')[0].replace('d', '.'))
                                b = float(impact.split('kms_')[1].split('b')[0].replace('d', '.'))
                                gamma = float(impact.split('b_')[1].split('gamma')[0].replace('d', '.'))
                                
                                (_, 
                                 _, 
                                 _, 
                                 _, 
                                 _, 
                                 _, 
                                 bound_cell_atmos_mass_time_near, 
                                 _) = calculate_global_loss_near_far2(folder_path = temp_dir,
                                                                                      max_remnant = 1,
                                                                                      npt = npt,
                                                                                      m_tot = M_t + M_i,
                                                                                      output_id = 1,
                                                                                      atmos_id = 200,
                                                                                      near_far_idx = (''
                                                                                                     )
                                                                                     )

                                (_,
                                 _,
                                 initial_cell_mantle_mass,
                                 mantle_vap_frac) = calculate_mantle_shell_vap(temp_dir,
                                                                         'fors_curve.txt',
                                                                         npt = npt,
                                                                        )
                                
                                FF_mask = np.zeros_like(bound_cell_atmos_mass_time_near[:, :, 1], dtype = bool)
                                FF_mask[:np.isnan(bound_cell_atmos_mass_time_near[:, :, 1]).shape[0], :] = np.isnan(bound_cell_atmos_mass_time_near[:, :, 1])

                                # Apply the mask:
                                masked_mantle_vap_frac = np.ma.masked_where(FF_mask, mantle_vap_frac)
                                masked_initial_cell_mantle_mass = np.ma.masked_where(FF_mask, initial_cell_mantle_mass)

                                X_mvap = ((np.nansum(masked_initial_cell_mantle_mass[:, :] * masked_mantle_vap_frac[:, :]) / np.nansum(masked_initial_cell_mantle_mass[:, :])))
                                data = {'X_mvap': [X_mvap]}
                                df = pd.DataFrame(data)
                                df.to_parquet(f'Xmvap_data_bound-unbound_shell/Xmvap_data_{impact}', index = False)
                                
                            except FileNotFoundError as e:
                                print(f"File or directory not found: {e}. Skipping...")
                                continue

                            except Exception as e:
                                print(f"Unexpected error: {e}. Skipping...")
                                continue