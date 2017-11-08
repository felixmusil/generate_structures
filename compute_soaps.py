from Pool.mpi_pool import MPIPool
import sys,argparse
import numpy as np
from libs.io import Frame_Dataset_h5
from tqdm import tqdm
import h5py
from glob import glob
from time import ctime

sys.path.insert(0,'/home/musil/git/glosim2/')
sys.path.insert(0,'/local/git/glosim2/')
from libmatch.soap import get_Soaps


def compute_soap(fn,soap_params,nprocess=1,string_dtype ='S200'):
    frame_reader = Frame_Dataset_h5(fn,mode='r',disable_pbar=True)
    frame_names = frame_reader.names
    ffs = frame_reader.load_frames(frame_names,frame_type='quippy')
    frames = [ffs[frame_name] for frame_name in frame_names]

    fings = get_Soaps(frames, nprocess=nprocess, **soap_params)

    soaps = []
    idx2frame = []

    for fing, frame_name in zip(fings, frame_names):
        soaps.append(fing['AVG'])
        idx2frame.append(np.array([fn, frame_name], dtype=string_dtype))

    soaps = np.asarray(soaps,dtype='f8')
    idx2frame = np.asarray(idx2frame,dtype=string_dtype)

    fn_soaps = fn[:-3] + '-soaps.npy'
    fn_idx2frame = fn[:-3] + '-idx2frame.npy'

    np.save(fn_soaps,soaps)
    np.save(fn_idx2frame,idx2frame)

    return (fn,fn_soaps,fn_idx2frame)

def compute_soap_wrapper(kwargs):
    return compute_soap(**kwargs)

if __name__ == '__main__':
    pool = MPIPool()

    if not pool.is_master():

        pool.wait()
        # pool.wait(callbacks=[fout.close,executor.shutdown])
        sys.exit(0)



    dataPath = '/home/musil/workspace/qmat/structures/'

    fns = glob(dataPath + 'relaxed_structures_step1_*.h5')

    chunkSize = 50
    fout = dataPath + 'descriptor-chunk{}.h5'.format(chunkSize)

    centerweight = 1.
    gaussian_width = 0.5
    cutoff = 3.5
    cutoff_transition_width = 0.5
    nmax = 10
    lmax = 15
    nocenters = []
    is_fast_average = True


    print len(fns)

    Ntot = 0
    frame_readers = {}
    sizes = [0]
    Nstr = []
    for fn in fns:
        Nstr.append(len(fn))
        rr = Frame_Dataset_h5(fn, mode='r')
        sizes.append(len(rr.names))
        Ntot += len(rr.names)
        frame_readers[fn] = rr
    print Ntot
    Nstr = np.max(Nstr) + 1
    print Nstr
    strides = np.cumsum(sizes)

    strides = {fn:(strides[it],strides[it+1]) for it,fn in enumerate(fns)}
    print strides

    soap_params = {
              'centerweight': centerweight,
              'gaussian_width': gaussian_width,'cutoff': cutoff,
              'cutoff_transition_width': cutoff_transition_width,
              'nmax': nmax, 'lmax': lmax, 'is_fast_average':is_fast_average,
              'chem_channels': False ,'nocenters': nocenters,'dispbar':True,
                   }

    frame = frame_readers[fns[0]].load_frame('frame_0')

    fings = get_Soaps([frame],nprocess=1, **soap_params)[0]['AVG']
    Nsoap = fings.shape[0]
    print Nsoap



    # fout = dataPath + 'descriptor_test-chunk{}.h5'.format(chunkSize)
    with h5py.File(fout, mode='w', libver='latest') as f:
        idx2frame = f.create_dataset("idx2frame", (Ntot, 2),
                                     dtype="S{}".format(Nstr), chunks=(chunkSize, 2))
        data = f.create_dataset("data", (Ntot, Nsoap), dtype='f8', chunks=(chunkSize, Nsoap))

        f.attrs['created'] = ctime()
        for k,v in soap_params.iteritems():
            f.attrs[k] = v

    inputs = [dict(fn=fn,soap_params=soap_params,
                   nprocess=1,string_dtype ="S{}".format(Nstr)) for fn in fns]


    fn_names = pool.map(compute_soap_wrapper,inputs)

    print 'End of pool'

    with h5py.File(fout, mode='r+', libver='latest') as f:
        data = f['data']
        idx2frame = f['idx2frame']

        for fn,fn_soaps,fn_idx2frame in tqdm(fn_names,desc='save h5'):
            st,nd = strides[fn]
            data[st:nd,:] = np.load(fn_soaps)
            idx2frame[st:nd,:] = np.load(fn_idx2frame)


    pool.close()