#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:38:43 2021

@author: tibor
"""

from copy import deepcopy
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import h5py
import make_grid_cic
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

SNAP_MAX = 16
SNAP_DEST = '/data/highz3/AFdata3/AF_WDM_LS_2021/BL256b40/snapdir_000'
SNAP_ABB = "000"
L_BOX = 40 # in cMpc/h
N = 256 # DM res

def getHDF5Data(with_gas, with_stars): # To be improved
    star_x = np.empty(0, dtype = np.float32)
    star_y = np.empty(0, dtype = np.float32)
    star_z = np.empty(0, dtype = np.float32)
    star_masses = np.empty(0, dtype = np.float32)
    dm_x = np.empty(0, dtype = np.float32)
    dm_y = np.empty(0, dtype = np.float32)
    dm_z = np.empty(0, dtype = np.float32)
    dm_masses = np.empty(0, dtype = np.float32)
    gas_x = np.empty(0, dtype = np.float32)
    gas_y = np.empty(0, dtype = np.float32)
    gas_z = np.empty(0, dtype = np.float32)
    gas_masses = np.empty(0, dtype = np.float32)
    nb_jobs_to_do = SNAP_MAX
    perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
    do_sth = rank <= nb_jobs_to_do-1
    count_dm = 0
    count_star = 0
    count_gas = 0
    last = rank == size - 1 # Whether or not last process
    for snap_run in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
        f = h5py.File(r'{0}/snap_{1}.{2}.hdf5'.format(SNAP_DEST, SNAP_ABB, snap_run), 'r')
        dm_x = np.hstack((dm_x, np.float32(f['PartType1/Coordinates'][:,0]/1000))) # in Mpc = 3.085678e+27 cm
        dm_y = np.hstack((dm_y, np.float32(f['PartType1/Coordinates'][:,1]/1000))) 
        dm_z = np.hstack((dm_z, np.float32(f['PartType1/Coordinates'][:,2]/1000))) 
        dm_masses = np.hstack((dm_masses, np.float32(np.ones_like(f['PartType1/Coordinates'][:,0])*np.float32(f['Header'].attrs['MassTable'][1])))) # in 1.989e+43 g
        if with_stars == True:
            star_x = np.hstack((star_x, np.float32(f['PartType4/Coordinates'][:,0]/1000))) # in Mpc = 3.085678e+27 cm
            star_y = np.hstack((star_y, np.float32(f['PartType4/Coordinates'][:,1]/1000))) 
            star_z = np.hstack((star_z, np.float32(f['PartType4/Coordinates'][:,2]/1000))) 
            star_masses = np.hstack((star_masses, np.float32(f['PartType4/Masses'][:]))) # in 1.989e+43 g
            count_star += f['PartType4/Coordinates'][:].shape[0]
        if with_gas == True:
            gas_x = np.hstack((gas_x, np.float32(f['PartType0/Coordinates'][:,0]/1000))) # in Mpc = 3.085678e+27 cm
            gas_y = np.hstack((gas_y, np.float32(f['PartType0/Coordinates'][:,1]/1000))) 
            gas_z = np.hstack((gas_z, np.float32(f['PartType0/Coordinates'][:,2]/1000))) 
            gas_masses = np.hstack((gas_masses, np.float32(f['PartType0/Masses'][:]))) # in 1.989e+43 g
            count_gas += f['PartType0/Coordinates'][:].shape[0]
        
        count_dm += f['PartType1/Coordinates'][:].shape[0]
    
    if with_gas == True:
        count_new_gas = comm.gather(count_gas, root=0)
        count_new_gas = comm.bcast(count_new_gas, root = 0)
        nb_gas = np.sum(np.array(count_new_gas))
        comm.Barrier()
        recvcounts_gas = np.array(count_new_gas)
        rdispls_gas = np.zeros_like(recvcounts_gas)
        for j in range(rdispls_gas.shape[0]):
            rdispls_gas[j] = np.sum(recvcounts_gas[:j])
        gas_x_total = np.empty(nb_gas, dtype = np.float32)
        gas_y_total = np.empty(nb_gas, dtype = np.float32)
        gas_z_total = np.empty(nb_gas, dtype = np.float32)
        gas_masses_total = np.empty(nb_gas, dtype = np.float32)
        comm.Gatherv(gas_x, [gas_x_total, recvcounts_gas, rdispls_gas, MPI.FLOAT], root = 0)
        comm.Gatherv(gas_y, [gas_y_total, recvcounts_gas, rdispls_gas, MPI.FLOAT], root = 0)
        comm.Gatherv(gas_z, [gas_z_total, recvcounts_gas, rdispls_gas, MPI.FLOAT], root = 0)
        comm.Gatherv(gas_masses, [gas_masses_total, recvcounts_gas, rdispls_gas, MPI.FLOAT], root = 0)
        comm.Bcast(gas_x_total, root=0)
        comm.Bcast(gas_y_total, root=0)
        comm.Bcast(gas_z_total, root=0)
        comm.Bcast(gas_masses_total, root=0)
        gas_xyz = np.hstack((np.reshape(gas_x_total, (gas_x_total.shape[0],1)), np.reshape(gas_y_total, (gas_y_total.shape[0],1)), np.reshape(gas_z_total, (gas_z_total.shape[0],1))))
    else:
        gas_xyz = None
        gas_masses_total = None
        
    if with_stars == True:
        count_new_star = comm.gather(count_star, root=0)
        count_new_star = comm.bcast(count_new_star, root = 0)
        nb_star = np.sum(np.array(count_new_star))
        comm.Barrier()
        recvcounts_star = np.array(count_new_star)
        rdispls_star = np.zeros_like(recvcounts_star)
        for j in range(rdispls_star.shape[0]):
            rdispls_star[j] = np.sum(recvcounts_star[:j])
        star_x_total = np.empty(nb_star, dtype = np.float32)
        star_y_total = np.empty(nb_star, dtype = np.float32)
        star_z_total = np.empty(nb_star, dtype = np.float32)
        star_masses_total = np.empty(nb_star, dtype = np.float32)
        comm.Gatherv(star_x, [star_x_total, recvcounts_star, rdispls_star, MPI.FLOAT], root = 0)
        comm.Gatherv(star_y, [star_y_total, recvcounts_star, rdispls_star, MPI.FLOAT], root = 0)
        comm.Gatherv(star_z, [star_z_total, recvcounts_star, rdispls_star, MPI.FLOAT], root = 0)
        comm.Gatherv(star_masses, [star_masses_total, recvcounts_star, rdispls_star, MPI.FLOAT], root = 0)
        comm.Bcast(star_x_total, root=0)
        comm.Bcast(star_y_total, root=0)
        comm.Bcast(star_z_total, root=0)
        comm.Bcast(star_masses_total, root=0)
        star_xyz = np.hstack((np.reshape(star_x_total, (star_x_total.shape[0],1)), np.reshape(star_y_total, (star_y_total.shape[0],1)), np.reshape(star_z_total, (star_z_total.shape[0],1))))
    else:
        star_xyz = None
        star_masses_total = None
    
    count_new_dm = comm.gather(count_dm, root=0)
    count_new_dm = comm.bcast(count_new_dm, root = 0)
    nb_dm = np.sum(np.array(count_new_dm))
    comm.Barrier()
    recvcounts_dm = np.array(count_new_dm)
    rdispls_dm = np.zeros_like(recvcounts_dm)
    for j in range(rdispls_dm.shape[0]):
        rdispls_dm[j] = np.sum(recvcounts_dm[:j])
    dm_x_total = np.empty(nb_dm, dtype = np.float32)
    dm_y_total = np.empty(nb_dm, dtype = np.float32)
    dm_z_total = np.empty(nb_dm, dtype = np.float32)
    dm_masses_total = np.empty(nb_dm, dtype = np.float32)
    comm.Gatherv(dm_x, [dm_x_total, recvcounts_dm, rdispls_dm, MPI.FLOAT], root = 0)
    comm.Gatherv(dm_y, [dm_y_total, recvcounts_dm, rdispls_dm, MPI.FLOAT], root = 0)
    comm.Gatherv(dm_z, [dm_z_total, recvcounts_dm, rdispls_dm, MPI.FLOAT], root = 0)
    comm.Gatherv(dm_masses, [dm_masses_total, recvcounts_dm, rdispls_dm, MPI.FLOAT], root = 0)
    comm.Bcast(dm_x_total, root=0)
    comm.Bcast(dm_y_total, root=0)
    comm.Bcast(dm_z_total, root=0)
    comm.Bcast(dm_masses_total, root=0)
    dm_xyz = np.hstack((np.reshape(dm_x_total, (dm_x_total.shape[0],1)), np.reshape(dm_y_total, (dm_y_total.shape[0],1)), np.reshape(dm_z_total, (dm_z_total.shape[0],1))))
    
    return dm_xyz, dm_masses_total, gas_xyz, gas_masses_total, star_xyz, star_masses_total

def plot2D(data_2D, L_BOX, axis, material):
    plt.figure()
    plt.imshow(data_2D,interpolation='None',origin='upper', extent=[0, L_BOX, L_BOX, 0], cmap = "hot")
    plt.gca().xaxis.set_major_locator(MultipleLocator(L_BOX/4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(L_BOX/20))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
    plt.gca().yaxis.set_major_locator(MultipleLocator(L_BOX/4))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(L_BOX/20))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
    plt.xlabel(r"{0} (cMpc/h)".format('y' if axis == 'z' else 'z' if axis == 'y' else 'x'), fontweight='bold')
    plt.ylabel(r"{0} (cMpc/h)".format('x' if axis == 'z' else 'x' if axis == 'y' else 'y'), fontweight='bold')
    plt.colorbar()
    plt.title(r'{0}-Projected {1}'.format(axis, material))
    plt.savefig("{0}Projected{1}.pdf".format(axis, material))
    plt.figure()
    data = deepcopy(data_2D)
    second_smallest = np.unique(data)[1]
    assert second_smallest != 0.0 # For positive-semidefinite datasets
    data[data < second_smallest] = second_smallest
    plt.imshow(data,interpolation='None',origin='upper', extent=[0, L_BOX, L_BOX, 0], cmap = "hot", norm=colors.LogNorm(vmin=second_smallest, vmax=np.max(data)))
    plt.gca().xaxis.set_major_locator(MultipleLocator(L_BOX/4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(L_BOX/20))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
    plt.gca().yaxis.set_major_locator(MultipleLocator(L_BOX/4))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(L_BOX/20))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
    plt.xlabel(r"{0} (cMpc/h)".format('y' if axis == 'z' else 'z' if axis == 'y' else 'x'), fontweight='bold')
    plt.ylabel(r"{0} (cMpc/h)".format('x' if axis == 'z' else 'x' if axis == 'y' else 'y'), fontweight='bold')
    plt.colorbar()
    plt.title(r'{0}-Projected {1}Log'.format(axis, material))
    plt.savefig("{0}Projected{1}Log.pdf".format(axis, material))
    
def project2D(grid, axis):
    data_proj_cic = np.zeros((grid.shape[0], grid.shape[0]))
    for h in range(grid.shape[0]):
        if axis == 'x':
            data_proj_cic += grid[h,:,:]
        elif axis == 'y':
            data_proj_cic += grid[:,h,:]
        else:
            data_proj_cic += grid[:,:,h]
    data_proj_cic /= grid.shape[0]
    return data_proj_cic
    
def projectNineTimes(dm_xyz, dm_masses, gas_xyz, gas_masses, star_xyz, star_masses, L_BOX, N, with_gas, with_stars):
    print("hello", dm_xyz[dm_xyz[:,0] < 20].shape[0])
    print("masses", dm_masses[:10])
    dm_xyz = dm_xyz[np.argsort(dm_xyz[:,0])]
    np.savetxt('pos_hdf5.txt', np.reshape(dm_xyz, (dm_xyz.shape[0], -1)), fmt='%1.7e')
    print(dm_xyz[-10:])
    material_l = {'DM': dm_xyz}
    masses_l = {'DM': dm_masses}
    names = ['DM']
    if with_gas == True:
        material_l['Gas'] = gas_xyz
        masses_l['Gas'] = gas_masses 
        names.append('Gas')
    if with_stars == True:
        material_l['Stars'] = star_xyz
        masses_l['Stars'] = star_masses
        names.append('Stars')
    for name in names:
        grid = make_grid_cic.makeGridWithCICPBC(material_l[name][:,0].astype('float32'), material_l[name][:,1].astype('float32'), material_l[name][:,2].astype('float32'), masses_l[name].astype('float32'), L_BOX, N)
        for axis in ['x', 'y', 'z']:
            data_proj_cic = project2D(grid, axis)  
            plot2D(data_proj_cic, L_BOX, axis, name)
    

num_tot = h5py.File(r'{0}/snap_{1}.0.hdf5'.format(SNAP_DEST, SNAP_ABB), 'r')['Header'].attrs['NumPart_Total']
if num_tot[0] == 0:
    with_gas = False
else:
    with_gas = True
if num_tot[4] == 0:
    with_stars = False
else:
    with_stars = True

dm_xyz, dm_masses, gas_xyz, gas_masses, star_xyz, star_masses = getHDF5Data(with_gas, with_stars)
print(dm_xyz.shape)
if rank == 0:
    projectNineTimes(dm_xyz, dm_masses, gas_xyz, gas_masses, star_xyz, star_masses, L_BOX, N, with_gas, with_stars)
