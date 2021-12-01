#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:42:07 2021

@author: tibor
"""

from pygadgetreader import readheader,readsnap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import make_grid_cic
from copy import deepcopy

# Parameters
L_BOX = 40 # cMpc/h
N = 1024
h = 0.667
path_ = "/home/tibor/Documents/PhD_in_Astronomy/DiRAC/Issues/BLGas1024/becdm_40mpc_1e22_256_ic"
type_ = "dm"

# Loading
ds = readheader(path_, 'redshift')
print(ds)
pos = readsnap(path_, 'pos', type_)/1000 # cMpc/h
masses = readsnap(path_, 'mass', type_) # in 10^10 M_sun/h
assert masses.shape[0] == pos.shape[0]
pos_ = deepcopy(pos)
print("The first ten coordinate vectors are:", pos_[np.argsort(pos_[:,0])][:10])
print("Pos shape", pos.shape, "number of particles in upper plane:", pos[pos[:,0]<20].shape[0])
print("Nans:", np.count_nonzero(np.isnan(pos)), np.count_nonzero(np.isnan(masses)))

# Getting Grid
grid = make_grid_cic.makeGridWithCICPBC(pos[:,0].astype('float32'), pos[:,1].astype('float32'), pos[:,2].astype('float32'), masses.astype('float32'), L_BOX, N)

# Project onto axis
rho_proj_cic = np.zeros((N, N))
for height in range(0, N):
    rho_proj_cic += grid[:,:,height]
rho_proj_cic /= N
second_smallest = np.unique(rho_proj_cic)[1]
assert second_smallest != 0.0 # For positive-semidefinite datasets
rho_proj_cic[rho_proj_cic < second_smallest] = second_smallest
plt.imshow(rho_proj_cic, interpolation='None', origin='upper', extent=[0, L_BOX, L_BOX, 0], cmap = "hot")
plt.gca().xaxis.set_major_locator(MultipleLocator(L_BOX/4))
plt.gca().xaxis.set_minor_locator(MultipleLocator(L_BOX/20))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
plt.gca().yaxis.set_major_locator(MultipleLocator(L_BOX/4))
plt.gca().yaxis.set_minor_locator(MultipleLocator(L_BOX/20))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
plt.xlabel(r"y (cMpc/h)", fontweight='bold')
plt.ylabel(r"x (cMpc/h)", fontweight='bold')
plt.colorbar()
plt.title(r'z-Projected, PyGV, {0}, IC'.format(type_.upper()))
plt.savefig("zProjectedDM_PyGV_{0}_IC.pdf".format(type_.upper()))
