from Pool.mpi_pool import MPIPool
import sys
from libs.utils import unskewCell,get_standard_frame,stdchannel_to_null
from libs.input_structure import input2crystal
from libs.LJ_pressure import LJ_vcrelax


def generate_crystal_step_1(sites_z, seed, vdw_ratio, isotropic_external_pressure=1e-2, symprec=1e-5):
    crystal, sg, wki = input2crystal(sites_z, seed, vdw_ratio)

    crystal = unskewCell(crystal)

    crystal = LJ_vcrelax(crystal,isotropic_external_pressure,debug=True)

    crystal = get_standard_frame(crystal, to_primitive=False, symprec=symprec)

    return crystal


if __name__ == '__main__':
    # pool = MPIPool()
    #
    # if not pool.is_master():
    #     # Wait for instructions from the master process.
    #     pool.wait()
    #     sys.exit(0)



    vdw_ratio = 1.5
    sites_z = [14]


    inputs = [[sites_z,seed,vdw_ratio] for seed in [29]]

    for it,inp in enumerate(inputs):
        print '############################'
        print it,inp
        generate_crystal_step_1(*inp)

