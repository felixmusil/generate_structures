from Pool.mpi_pool import MPIPool
import sys,argparse
import numpy as np
from libs.io import Frame_Dataset_h5
from tqdm import tqdm
import h5py
from glob import glob
from time import ctime
import spglib as spg




def add_sg_w(fn):
    frame_reader = Frame_Dataset_h5(fn,mode='r',swmr_mode=False,disable_pbar=True)

    frame_names = frame_reader.names

    frame_reader.open(mode='r+')

    for frame_name in frame_names:
        ff = frame_reader.load_frame(frame_name)
        sym_data = spg.get_symmetry_dataset(ff, symprec=1e-5)
        sg = sym_data['number']
        wyck = np.unique(sym_data['wyckoffs'])[0]
        frame_reader.f[frame_name].attrs['space group'] = sg
        frame_reader.f[frame_name].attrs['wyckoff site'] = wyck
        frame_reader.f[frame_name].attrs['sym tag'] = '{}:{}'.format(sg,wyck)
    frame_reader.close()



if __name__ == '__main__':
    pool = MPIPool()

    if not pool.is_master():

        pool.wait()
        # pool.wait(callbacks=[fout.close,executor.shutdown])
        sys.exit(0)


    dataPath = '/home/musil/workspace/qmat/structures/'

    fns = glob(dataPath + 'relaxed_structures_step1_*.h5')
    print len(fns)
    inputs = [fn for fn in fns]
    # Ns = map(add_Nequivalent,inputs)
    Ns = pool.map(add_sg_w,inputs)

    for N in Ns:
        print N

    pool.close()