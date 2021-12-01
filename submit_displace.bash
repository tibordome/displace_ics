#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/vault/td448/DisPerSE/external/gmp/gmp-4.2.4/install/usr/local/lib # only for HTCondor

mpirun -n 8 python3 displace_and_save.py
