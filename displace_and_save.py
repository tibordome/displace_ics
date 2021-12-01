#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 13:16:00 2021

@author: tibor
"""

from pygadgetreader import readsnap, readheader
import numpy as np
import h5py
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def displace(x, y, z, L_BOX, epsilon):

    vec = np.array([x,y,z]) + (np.random.normal(loc = 0.0, scale = epsilon, size = 3)).astype('float{0}'.format(bits_in_float))
    for dim in range(3):
        if vec[dim] > L_BOX:
            vec[dim] = vec[dim] - L_BOX
        if vec[dim] < 0:
            vec[dim] = L_BOX + vec[dim]
    return vec[0], vec[1], vec[2]

# Parameters
L_BOX = 10 # cMpc/h
N = 1024
epsilon = 0.00000001 # cMpc/h, standard deviation of my activate displacement in all directions
path_ = "/data/highz3/BLGas1024Issue/voronoi/becdmGas_{0}mpc_1e22_{1}_ic".format(L_BOX, N)
type_ = "gas"
high_word = 0
nb_files_per_snap = 1 # Assumes that nb_files_per_snap divdes N**3 without rest and nb_files_per_snap is either 1 or the number of IC type 1 files
flag_double_precision = 0
if flag_double_precision == 1:
    bits_in_float = 64
    float_type = np.float64
    mpi_float_type = MPI.DOUBLE
else:
    bits_in_float = 32
    float_type = np.float32
    mpi_float_type = MPI.FLOAT
if rank == 0:
    print("The mass of the DM particles in units of 10^10 M_sun/h is:", readsnap(path_, 'mass', 'dm'))
    print("The mass of the gas particles in units of 10^10 M_sun/h is:", readsnap(path_, 'mass', 'gas'))
    print("The hubble param is", readheader(path_, 'h'))

# Loading and sorting in rank 0x
if rank == 0:
    NumPart_ThisFile = readheader(path_, 'npartTotal') # Instead of npartThisFile
    NumPart_Total = readheader(path_, 'npartTotal') # Includes HighWord
    NumPart_Total_HighWord = [high_word,high_word,high_word,high_word,high_word,high_word] # High word particles
    MassTable = [0., np.float32(readsnap(path_, 'mass', 'dm')[0]), 0., 0., 0., 0.]
    Time = readheader(path_, 'time')
    Redshift = readheader(path_, 'redshift')
    BoxSize = readheader(path_, 'boxsize')
    NumFilesPerSnapshot = nb_files_per_snap
    Omega0 = readheader(path_, 'O0')
    OmegaLambda = readheader(path_, 'Ol')
    HubbleParam = readheader(path_, 'h')
    Flag_Sfr = readheader(path_, 'f_sfr')
    Flag_Cooling = readheader(path_, 'f_cooling')
    Flag_StellarAge = readheader(path_, 'f_age')
    Flag_Metals = readheader(path_, 'f_metals')
    Flag_Feedback = readheader(path_, 'f_fb')
    Flag_DoublePrecision = flag_double_precision


    pos = (readsnap(path_, 'pos', type_)/1000).astype('float{0}'.format(bits_in_float)) # cMpc/h
    pos_x_tmp = pos[:,0]
    pos_y_tmp = pos[:,1]
    pos_z_tmp = pos[:,2]
    print("The number of {0} particles is {1}".format(type_, pos_x_tmp.shape[0]))
else:
    pos_x_tmp = np.empty(N**3, dtype = float_type)
    pos_y_tmp = np.empty(N**3, dtype = float_type)
    pos_z_tmp = np.empty(N**3, dtype = float_type)
pos_x_tmp = np.ascontiguousarray(pos_x_tmp, dtype = float_type)
pos_y_tmp = np.ascontiguousarray(pos_y_tmp, dtype = float_type)
pos_z_tmp = np.ascontiguousarray(pos_z_tmp, dtype = float_type)

pieces = 1 + (N**3>=3*10**8)*N**3//(3*10**8) # Not too high since this is a slow-down!
chunk = N**3//pieces
pos_x = np.empty(0, dtype = float_type)
pos_y = np.empty(0, dtype = float_type)
pos_z = np.empty(0, dtype = float_type)
for i in range(pieces):
    to_bcast = pos_x_tmp[i*chunk:(i+1)*chunk+(i==(pieces-1))*(N**3-pieces*chunk)]
    comm.Bcast(to_bcast, root=0)
    pos_x = np.hstack((pos_x, to_bcast))
    to_bcast = pos_y_tmp[i*chunk:(i+1)*chunk+(i==(pieces-1))*(N**3-pieces*chunk)]
    comm.Bcast(to_bcast, root=0)
    pos_y = np.hstack((pos_y, to_bcast))
    to_bcast = pos_z_tmp[i*chunk:(i+1)*chunk+(i==(pieces-1))*(N**3-pieces*chunk)]
    comm.Bcast(to_bcast, root=0)
    pos_z = np.hstack((pos_z, to_bcast))

# Displace particles
nb_jobs_to_do = N**3
perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
do_sth = rank <= nb_jobs_to_do-1
last = rank == size - 1 # Whether or not last process
count = do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))
for i in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
    pos_x[i], pos_y[i], pos_z[i] = displace(pos_x[i], pos_y[i], pos_z[i], L_BOX, epsilon)

# Reduce size of pos to those that have been displaced
pos_x = pos_x[rank*perrank:rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))]
pos_y = pos_y[rank*perrank:rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))]
pos_z = pos_z[rank*perrank:rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))]
pos_x = np.ascontiguousarray(pos_x, dtype = float_type)
pos_y = np.ascontiguousarray(pos_y, dtype = float_type)
pos_z = np.ascontiguousarray(pos_z, dtype = float_type)
count_new = comm.gather(count, root=0)
count_new = comm.bcast(count_new, root = 0)
nb_ptcs = np.sum(np.array(count_new))
comm.Barrier()
recvcounts = np.array(count_new)
rdispls = np.zeros_like(recvcounts)
for j in range(rdispls.shape[0]):
    rdispls[j] = np.sum(recvcounts[:j])
pos_x_tmp = np.empty(nb_ptcs, dtype = float_type)
pos_x_tmp = np.ascontiguousarray(pos_x_tmp, dtype = float_type)
pos_y_tmp = np.empty(nb_ptcs, dtype = float_type)
pos_y_tmp = np.ascontiguousarray(pos_y_tmp, dtype = float_type)
pos_z_tmp = np.empty(nb_ptcs, dtype = float_type)
pos_z_tmp = np.ascontiguousarray(pos_z_tmp, dtype = float_type)
comm.Gatherv(pos_x, [pos_x_tmp, recvcounts, rdispls, mpi_float_type], root = 0)
comm.Gatherv(pos_y, [pos_y_tmp, recvcounts, rdispls, mpi_float_type], root = 0)
comm.Gatherv(pos_z, [pos_z_tmp, recvcounts, rdispls, mpi_float_type], root = 0)

if rank == 0:

    pos = np.hstack((np.reshape(pos_x_tmp, (pos_x_tmp.shape[0],1)), np.reshape(pos_y_tmp, (pos_y_tmp.shape[0],1)), np.reshape(pos_z_tmp, (pos_z_tmp.shape[0],1))))

    # Save in hdf5 file
    print('Start writing file becdmGas_{0}mpc_1e22_{1}_ic.* with float precision {2}'.format(L_BOX, N, bits_in_float))
    
    for file_ in range(nb_files_per_snap):
        hf = h5py.File('/data/highz3/BLGas1024Issue/voronoi/becdmGas_{0}mpc_1e22_{1}_ic.{2}'.format(L_BOX, N, file_), 'w')
    
        header = hf.create_group('Header')
        if nb_files_per_snap != 1:
            header.attrs.__setitem__('NumPart_ThisFile', np.uint32(readheader(path_+".{0}".format(file_), 'npartThisFile')))
            print("NumPart_ThisFile in file {0} is hopefully N**3/16 = {1} for both gas and dm".format(file_, N**3/nb_files_per_snap))
        else:
            assert nb_files_per_snap == 1
            header.attrs.__setitem__('NumPart_ThisFile', np.uint32(NumPart_Total))
        header.attrs.__setitem__('NumPart_Total', np.uint32(NumPart_Total))
        header.attrs.__setitem__('NumPart_Total_HighWord', np.uint32(NumPart_Total_HighWord))
        header.attrs.__setitem__('MassTable',np.float32(MassTable))
        header.attrs.__setitem__('Time',np.float32(Time))
        header.attrs.__setitem__('Redshift',np.float32(Redshift))
        header.attrs.__setitem__('BoxSize',np.float32(BoxSize))
        header.attrs.__setitem__('NumFilesPerSnapshot', np.uint32(NumFilesPerSnapshot))
        header.attrs.__setitem__('Omega0',np.float32(Omega0))
        header.attrs.__setitem__('OmegaLambda',np.float32(OmegaLambda))
        header.attrs.__setitem__('HubbleParam',np.float32(HubbleParam))
        header.attrs.__setitem__('Flag_Sfr', np.uint32(Flag_Sfr))
        header.attrs.__setitem__('Flag_Cooling', np.uint32(Flag_Cooling))
        header.attrs.__setitem__('Flag_StellarAge', np.uint32(Flag_StellarAge))
        header.attrs.__setitem__('Flag_Metals', np.uint32(Flag_Metals))
        header.attrs.__setitem__('Flag_Feedback', np.uint32(Flag_Feedback))
        header.attrs.__setitem__('Flag_DoublePrecision', np.uint32(Flag_DoublePrecision))
    
        nb_gas_ptcs_per_file = N**3/nb_files_per_snap
        part0 = hf.create_group('PartType0')
        part0.create_dataset('Coordinates', data = pos[nb_gas_ptcs_per_file*file_, nb_gas_ptcs_per_file*(file_+1)]*1000) # in ckpc/h, has bits_in_float type
        part0.create_dataset('ParticleIDs', data = (readsnap(path_, 'pid', 'gas')[nb_gas_ptcs_per_file*file_, nb_gas_ptcs_per_file*(file_+1)]).astype('int32'))
        part0.create_dataset('Velocities', data = (readsnap(path_, 'vel', 'gas')[nb_gas_ptcs_per_file*file_, nb_gas_ptcs_per_file*(file_+1)]).astype('float{0}'.format(bits_in_float)))
        part0.create_dataset('Masses', data = (readsnap(path_, 'mass', 'gas')[nb_gas_ptcs_per_file*file_, nb_gas_ptcs_per_file*(file_+1)]).astype('float{0}'.format(bits_in_float)))
        part0.create_dataset('InternalEnergy', data = (readsnap(path_, 'u', 'gas')[nb_gas_ptcs_per_file*file_, nb_gas_ptcs_per_file*(file_+1)]).astype('float{0}'.format(bits_in_float)))
    
        nb_dm_ptcs_per_file = N**3/nb_files_per_snap
        part1 = hf.create_group('PartType1')
        part1.create_dataset('Coordinates', data = (readsnap(path_, 'pos', 'dm')[nb_dm_ptcs_per_file*file_, nb_dm_ptcs_per_file*(file_+1)]).astype('float{0}'.format(bits_in_float)))
        part1.create_dataset('ParticleIDs', data = (readsnap(path_, 'pid', 'dm')[nb_dm_ptcs_per_file*file_, nb_dm_ptcs_per_file*(file_+1)]).astype('int32'))
        part1.create_dataset('Velocities', data = (readsnap(path_, 'vel', 'dm')[nb_dm_ptcs_per_file*file_, nb_dm_ptcs_per_file*(file_+1)]).astype('float{0}'.format(bits_in_float)))
    
        hf.close()
        print('Finished writing file becdmGas_{0}mpc_1e22_{1}_ic.{2} with float precision {3}'.format(L_BOX, N, file_, bits_in_float))