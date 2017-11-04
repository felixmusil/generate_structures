from Pool.mpi_pool import MPIPool
import sys,argparse
from libs.utils import unskewCell,get_standard_frame,s2hms
from libs.input_structure import input2crystal
from libs.LJ_pressure import LJ_vcrelax
import cPickle as pck
import numpy as np
from libs.io import Frame_Dataset_h5
from tqdm import tqdm
from time import time,ctime
from ase.io.trajectory import Trajectory
import concurrent.futures as cf



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
        fout.dump_frame(crystal, inp_dict=kwargs)
        # fout.write(crystal)
        # executor.submit(fout.write, crystal)
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Generates random structures and relax them. Needs MPI to run, 
        mpiexec -n 4 python  """)

    parser.add_argument("basename", nargs=1, help="Name of the output files")
    parser.add_argument("-sl", "--seed-limits", type=str, default='0,1e3',
                        help="Limits for the seeds, i.e. seeds = range(lower,upper) . (comma separated)")
    parser.add_argument("-at", "--asymmetric-types", type=str, default='14',
                        help="Atomic number of the asymmetric atoms, i.e. atom types to populate wyckoff positions. (comma separated)")
    parser.add_argument("-wr","--vdw-ratio", type=float, default=1.5, help="Initial density target.")

    args = parser.parse_args()

    basename = args.basename[0]
    try:
        seedlim = [int(float(seed)) for seed in args.seed_limits.split(',')]
        sites_z = [int(z) for z in args.asymmetric_types.split(',')]
    except:
        raise ValueError('seedlim and sites_z must be coma separated int or float respectively')



    vdw_ratio = args.vdw_ratio

    pool = MPIPool()

    if pool.is_master():
        print 'Starting ' + ctime()
        print 'Generating structures randomly with seeds from {} to {}, ' \
              'for sites {} and Van der Wall volume ratio {}'.format(seedlim[0],seedlim[1],sites_z,vdw_ratio)


    comm = pool.comm

    if not pool.is_master():
        rank = comm.Get_rank()
        fname = basename + str(rank) + '.traj'
        # fout = Trajectory(fname,mode='a',master=True)

        # executor = cf.ThreadPoolExecutor(max_workers=1)

        fout = Frame_Dataset_h5(basename + str(rank) + '.h5',debug=False)
        print 'Dumping structures from process {} to {}'.format(rank,fout.fname)
        fout.open(mode='a')

        # Wait for instructions from the master process.
        pool.wait(callbacks=[fout.close])
        # pool.wait(callbacks=[fout.close,executor.shutdown])
        sys.exit(0)


    start = time()

    inputs = [{'sites_z':sites_z,'seed':seed,'vdw_ratio':vdw_ratio,
               'isotropic_external_pressure':1e-2, 'symprec':1e-5} for seed in range(seedlim[0],seedlim[1])]

    outs = pool.map(generate_crystal_step_1_wrapper,inputs,disable_pbar=False)


    with open(basename+'_out.pck','wb') as f:
        pck.dump(outs,f,protocol=pck.HIGHEST_PROTOCOL)

    print 'Elapsed time: {}'.format(s2hms(time()-start))
    pool.close()