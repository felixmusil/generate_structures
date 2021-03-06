from Pool.mpi_pool import MPIPool
import sys,os
from libs.utils import unskewCell,get_standard_frame,s2hms
from libs.input_structure import input2crystal
from libs.LJ_pressure import LJ_vcrelax
import cPickle as pck
import numpy as np
from libs.io import Frame_Dataset_h5
from tqdm import tqdm
from time import time,ctime

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

    crystal = generate_crystal_step_1(**kwargs)

    if crystal is None:
        return kwargs
    else:
        fout.dump_frames([crystal],[kwargs])
        return None


if __name__ == '__main__':




    pool = MPIPool()

    comm = pool.comm
    basename = './structures/relaxed_structures_step1_r'
    if not pool.is_master():
        rank = comm.Get_rank()
        fout = Frame_Dataset_h5(basename + str(rank) + '.h5')
        print 'Dumping structures to {}'.format(fout.fname)
        # Wait for instructions from the master process.
        pool.wait()
        sys.exit(0)

    vdw_ratio = 1.5
    sites_z = [14]


    print 'Starting '+ctime()

    start = time()
    strides = np.arange(2)*10 + int(2e7)
    # strides = np.arange(10) * 10 + int(1e7)
    for st,nd in zip(strides[:-1],strides[1:]):
        print 'Seed from {} to {}  {}'.format(st,nd,s2hms(time() - start))
        inputs = [{'sites_z':sites_z,'seed':seed,'vdw_ratio':vdw_ratio} for seed in range(st,nd)]

        out = pool.map(generate_crystal_step_1_wrapper,inputs,disable_pbar=False)

        print out

    print 'Elapsed time: {}'.format(s2hms(time()-start))
    pool.close()