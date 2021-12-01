#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/vault/td448/DisPerSE/external/gmp/gmp-4.2.4/install/usr/local/lib # only for HTCondor

mpirun -n 5 python3 check_for_duplicates.py
