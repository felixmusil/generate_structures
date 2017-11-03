from Pool.mpi_pool import MPIPool
import sys
from libs.utils import unskewCell,get_standard_frame,stdchannel_to_null
from libs.input_structure import input2crystal
from libs.LJ_pressure import LJ_vcrelax
import cPickle as pck
import numpy as np
from libs.io import Frame_Dataset_h5
from tqdm import tqdm

def generate_crystal_step_1(sites_z, seed, vdw_ratio, isotropic_external_pressure=1e-2, symprec=1e-5):
    crystal, sg, wki = input2crystal(sites_z, seed, vdw_ratio)

    crystal = unskewCell(crystal)

    crystal = LJ_vcrelax(crystal,isotropic_external_pressure,debug=False)

    crystal = get_standard_frame(crystal, to_primitive=False, symprec=symprec)

    return crystal

def generate_crystal_step_1_wrapper(kwargs):
    return generate_crystal_step_1(**kwargs)

if __name__ == '__main__':
    pool = MPIPool()

    if not pool.is_master():
        # Wait for instructions from the master process.
        pool.wait()
        sys.exit(0)

    fname = './test.h5'
    fout = Frame_Dataset_h5(fname)

    vdw_ratio = 1.5
    sites_z = [14]

    strides = np.arange(10)*10
    for st,nd in zip(strides[:-1],strides[1:]):
        print '################',st,nd
        inputs = [{'sites_z':sites_z,'seed':seed,'vdw_ratio':vdw_ratio} for seed in range(st,nd)]

        crystals = pool.map(generate_crystal_step_1_wrapper,inputs,disable_pbar=True)

        print len(crystals)
        fout.dump_frames(crystals,inputs)
    #print crystals
    pool.close()