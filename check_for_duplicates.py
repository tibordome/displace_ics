#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 13:16:00 2021

@author: tibor
"""

from pygadgetreader import readsnap
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def getDist(x, y, L_BOX):

    dist_x = abs(x[0]-y[0])
    if dist_x > L_BOX/2:
        dist_x = L_BOX-dist_x
    dist_y = abs(x[1]-y[1])
    if dist_y > L_BOX/2:
        dist_y = L_BOX-dist_y
    dist_z = abs(x[2]-y[2])
    if dist_z > L_BOX/2:
        dist_z = L_BOX-dist_z
    return np.sqrt(dist_x**2+dist_y**2+dist_z**2)

# Parameters
L_BOX = 40 # cMpc/h
cut = 0.0001 # cMpc/h, if two successive ptcs are closer than this, print
N = 1024
h = 0.667
path_ = "/data/highz3/BLGas1024Issue/becdmGas_40mpc_1e22_1024_ic"
type_ = "gas"

# Loading and sorting in rank 0
if rank == 0:
    pos = readsnap(path_, 'pos', type_)/1000 # cMpc/h
    # Sort position vectors according to their x-coordinate
    pos = np.float32(pos[np.argsort(pos[:,0])])
    pos_x_tmp = pos[:,0]
    pos_y_tmp = pos[:,1]
    pos_z_tmp = pos[:,2]
else:
    pos_x_tmp = np.empty(N**3, dtype = np.float32)
    pos_y_tmp = np.empty(N**3, dtype = np.float32)
    pos_z_tmp = np.empty(N**3, dtype = np.float32)
pos_x_tmp = np.ascontiguousarray(pos_x_tmp, dtype = np.float32)
pos_y_tmp = np.ascontiguousarray(pos_y_tmp, dtype = np.float32)
pos_z_tmp = np.ascontiguousarray(pos_z_tmp, dtype = np.float32)
pieces = 1 + (N**3>=3*10**8)*N**3//(3*10**8) # Not too high since this is a slow-down!
chunk = N**3//pieces
pos_x = np.empty(0, dtype = np.float32)
pos_y = np.empty(0, dtype = np.float32)
pos_z = np.empty(0, dtype = np.float32)
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
pos = np.hstack((np.reshape(pos_x, (pos_x.shape[0],1)), np.reshape(pos_y, (pos_y.shape[0],1)), np.reshape(pos_z, (pos_z.shape[0],1))))


# Search for inconsistencies
nb_jobs_to_do = N**3-1
perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
do_sth = rank <= nb_jobs_to_do-1
last = rank == size - 1 # Whether or not last process
for i in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
    if getDist(pos[i], pos[i+1], L_BOX) < cut:
        print("Particle ", i, "too close to next one", pos[i], pos[i+1])
