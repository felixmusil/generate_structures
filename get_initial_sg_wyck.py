from Pool.mpi_pool import MPIPool
import sys,argparse
import numpy as np
from libs.input_structure import  input2crystal
from libs.io import Frame_Dataset_h5
from tqdm import tqdm
import h5py
from glob import glob
from time import ctime
import spglib as spg
import pandas as pd



def get_initial_sg_wyck(fn):
    frame_reader = Frame_Dataset_h5(fn,mode='r',swmr_mode=False,disable_pbar=True)

    frame_names = frame_reader.names

    frame_reader.open(mode='r')
    data_init = {'Space Group': [], 'Wyckoff Position': [], 'tag': [], 'fn': [], 'frame_name': []}

    for frame_name in frame_names:
        inp = {}
        for k, v in frame_reader.f[frame_name]['inputs'].iteritems():
            inp[k] = v.value
        inp.pop('symprec')
        inp.pop('isotropic_external_pressure')
        cc, _, _ = input2crystal(**inp)

        sym_data = spg.get_symmetry_dataset(cc)
        sg = sym_data['number']
        wyck = np.unique(sym_data['wyckoffs'])
        # if len(wyck) > 1:
        #     print(fn, frame_reader, wyck)

        tag = '{}:{}'.format(sg, wyck[0])
        data_init['fn'].append(fn)
        data_init['frame_name'].append(frame_name)
        data_init['tag'].append(tag)
        data_init['Space Group'].append(sg)
        data_init['Wyckoff Position'].append(wyck[0])
    frame_reader.close()

    return data_init

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
    res = pool.map(get_initial_sg_wyck,inputs)

    data_inits = {'Space Group': [], 'Wyckoff Position': [], 'tag': [], 'fn': [], 'frame_name': []}

    for data_init in res:
        data_inits['fn'].extend(data_init['fn'])
        data_inits['frame_name'].extend(data_init['frame_name'])
        data_inits['tag'].extend(data_init['tag'])
        data_inits['Space Group'].extend(data_init['Space Group'])
        data_inits['Wyckoff Position'].extend(data_init['Wyckoff Position'])

    df = pd.DataFrame.from_dict(data_inits)

    df.to_pickle('sg_wyck_tag_step0.pck')

    pool.close()