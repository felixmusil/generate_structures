from Pool.mpi_pool import MPIPool
import sys
from libs.utils import unskewCell,get_standard_frame,s2hms
from libs.input_structure import input2crystal
from libs.LJ_pressure import LJ_vcrelax
import cPickle as pck
import numpy as np
from libs.io import Frame_Dataset_h5
from tqdm import tqdm
from time import time,ctime
from Pool.logger import log

def generate_crystal_step_1(sites_z, seed, vdw_ratio, isotropic_external_pressure=1e-2, symprec=1e-5):
    crystal, sg, wki = input2crystal(sites_z, seed, vdw_ratio)

    crystal = unskewCell(crystal)

    crystal = LJ_vcrelax(crystal,isotropic_external_pressure,debug=False)

    if crystal is None:
        return None
    else:
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

    fname = './relaxed_structures_step1.h5'
    fout = Frame_Dataset_h5(fname)

    vdw_ratio = 1.5
    sites_z = [14]


    print 'Starting '+ctime()
    print 'Dumping structures to {}'.format(fout.fname)
    start = time()
    strides = np.arange(20)*50000 + int(2e7)
    # strides = np.arange(10) * 10 + int(1e7)
    for st,nd in zip(strides[:-1],strides[1:]):
        print 'Seed from {} to {}'.format(st,nd)
        inputs = [{'sites_z':sites_z,'seed':seed,'vdw_ratio':vdw_ratio} for seed in range(st,nd)]

        crystals = pool.map(generate_crystal_step_1_wrapper,inputs,disable_pbar=False)

        fout.dump_frames(crystals,inputs)


    print 'Elapsed time: {}'.format(s2hms(time()-start))
    pool.close()