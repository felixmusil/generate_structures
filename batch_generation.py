import numpy as np
import numpy.random as npr
import spglib as spg
import ase
from ase.io import read,write
from ase.spacegroup import crystal
from ase.visualize import view
import cPickle as pck
import pandas as pd
from tqdm import tqdm


from libs.utils import unskewCell,ase2qp,qp2ase,get_standard_frame,isLayered
from libs.input_structure import input2crystal,getCellParam
from libs.LJ_pressure import make_LJ_input,LJ_vcrelax,LJ_vcrelax_alternative
from libs.raw_data import z2symb,z2VdWradius,z2Covalentradius,SG2BravaisLattice,WyckTable

import sys
sys.path.insert(0,'/local/git/glosim2/')
sys.path.insert(0,'/home/musil/git/glosim2/')
from libmatch.soap import get_Soaps
from libmatch.utils import get_soapSize,get_spkit,get_spkitMax,ase2qp,qp2ase
from libmatch.chemical_kernel import deltaKernel,PartialKernels
from GlobalSimilarity import get_environmentalKernels,get_globalKernel

d2r = np.pi/180.


def get_Nsoap(spkitMax, nmax, lmax):
    Nsoap = 0
    for sp1 in spkitMax:
        for sp2 in spkitMax:
            if sp1 == sp2:
                Nsoap += nmax * (nmax + 1) * (lmax + 1) / 2
            elif sp1 > sp2:
                Nsoap += nmax ** 2 * (lmax + 1)
    return Nsoap + 1


def get_fingerprints(frames, soap_params, nprocess):
    fings = get_Soaps(frames, nprocess=nprocess, **soap_params)
    N = len(frames)
    Nsoap = get_Nsoap(get_spkitMax(frames), soap_params['nmax'], soap_params['lmax'])
    soaps = np.zeros((N, Nsoap))
    ii = 0
    for iframe, fing in enumerate(fings):
        soaps[iframe] = fing['AVG']

    return soaps


def s2hms(time):
    m = time // 60
    s = int(time % 60)
    h = int(m // 60)
    m = int(m % 60)
    return '{:02d}:{:02d}:{:02d} (h:m:s)'.format(h, m, s)


def fpsSelection(data=None, distance_func=None, threshold=3e-3, Nmin=0, Nmax=20, seed=None):
    import numpy.random as npr
    from tqdm import tqdm_notebook
    nbOfFrames, Nfeature = data.shape

    if seed is None:
        isel = 0
    else:
        npr.seed(seed)
        isel = npr.randint(0, nbOfFrames)

    ldist = 1e100 * np.ones(nbOfFrames, float)
    dsel = np.zeros(nbOfFrames, float)
    idx_to_compute = np.ones(nbOfFrames, bool)
    idx_to_ignore = np.zeros(nbOfFrames, bool)

    LandmarksIdx = []

    nsel = 0
    Nidx = 0
    cond = True
    pbar = tqdm_notebook(total=nbOfFrames)
    while cond:
        LandmarksIdx.append(isel)
        Nidx += 1
        idx_to_compute[isel] = False
        idx_to_ignore[isel] = True
        dsel.fill(0.)

        imax = 0
        distLine = distance_func(data[isel, :].reshape((1, Nfeature)),
                                 data[idx_to_compute, :].reshape((-1, Nfeature)))

        dsel[idx_to_compute] = distLine.reshape((-1,))

        low = (dsel < ldist) * idx_to_compute
        ldist[low] = dsel[low]

        ldist[idx_to_ignore] = 0.
        isel = ldist.argmax()

        ids = (dsel < threshold) * idx_to_compute

        idx_to_compute[ids] = False
        idx_to_ignore[ids] = True

        if Nidx >= Nmin:
            if ldist[isel] < threshold:
                cond = False
            elif Nidx >= Nmax:
                cond = False
        pbar.update()
    pbar.close()

    return LandmarksIdx


def distance_func(XA, XB):
    # mkl.set_num_threads(10)
    kernel = np.dot(XB, XA.T)
    # the feature vectors are normalized
    dd = 2 - 2 * kernel
    dd[dd < 0.] = 0.
    distance = np.sqrt(dd)
    return distance


def distance_func2(kernel):
    # the feature vectors are normalized
    dd = 2 - 2 * kernel
    dd[dd < 0.] = 0.
    distance = np.sqrt(dd)
    return distance


def fpsSelection_with_restart(data=None, distance_func=None, restart_ref=None, disable_pbar=False, nthread=10,
                              intermediate_copy=True, stride=100, threshold=3e-3, Nmin=0, Nmax=20, seed=None, fn=None):
    import numpy.random as npr
    import cPickle as pck
    from tqdm import tqdm_notebook
    try:
        import mkl
        mkl.set_num_threads(nthread)
    except:
        pass

    # if fn is None:
    #     fn = 'restart_ref_thr{}.pck'.format(threshold)

    nbOfFrames, Nfeature = data.shape

    if nbOfFrames == Nfeature:
        iskernel = True
    else:
        iskernel = False

    if seed is None:
        isel = 0
    else:
        npr.seed(seed)
        isel = npr.randint(0, nbOfFrames)

    dsel = np.zeros(nbOfFrames, float)
    idx_to_compute = np.ones(nbOfFrames, bool)
    idx_to_ignore = np.zeros(nbOfFrames, bool)

    if restart_ref is None:
        ldist = 1e100 * np.ones(nbOfFrames, float)
        LandmarksIdx = []
        minmax = []
    else:
        LandmarksIdx = restart_ref['LandmarksIdx'][:-1]
        ldist = restart_ref['ldist']
        isel = restart_ref['LandmarksIdx'][-1]
        minmax = restart_ref['minmax']
        idx_to_compute[LandmarksIdx] = False
        idx_to_ignore[LandmarksIdx] = True

    Nidx = len(LandmarksIdx)
    cond = True
    pbar = tqdm(total=Nmax, disable=disable_pbar)

    while cond:

        LandmarksIdx.append(isel)

        if Nidx % (stride - 1) == 0 and fn is not None:
            with open(fn, 'wb') as f:
                pck.dump({'LandmarksIdx': LandmarksIdx, 'ldist': ldist, 'minmax': minmax}, f,
                         protocol=pck.HIGHEST_PROTOCOL)

        Nidx += 1
        idx_to_compute[isel] = False
        idx_to_ignore[isel] = True
        dsel.fill(0.)

        imax = 0
        if iskernel:
            distLine = distance_func(data[isel, idx_to_compute])
            dsel[idx_to_compute] = distLine
        else:
            if intermediate_copy:
                distLine = distance_func(data[isel, :].reshape((1, Nfeature)),
                                         data[idx_to_compute, :].reshape((-1, Nfeature)))
                dsel[idx_to_compute] = distLine.reshape((-1,))
            else:
                distLine = distance_func(data[isel, :].reshape((1, Nfeature)), data)
                dsel = distLine.reshape((-1,))

        low = (dsel < ldist) * idx_to_compute
        ldist[low] = dsel[low]

        ldist[idx_to_ignore] = 0.
        isel = ldist.argmax()
        minmax.append(ldist[isel])
        ids = (dsel < threshold) * idx_to_compute

        idx_to_compute[ids] = False
        idx_to_ignore[ids] = True

        if Nidx >= Nmin:
            if ldist[isel] < threshold:
                cond = False
            elif Nidx >= Nmax:
                cond = False

        pbar.update()
    pbar.close()
    return LandmarksIdx, minmax


def generate_crystal(sites_z):
    crystal, sg, wyckoff_letters = input2crystal(sites_z)

    initial_crystal = qp2ase(crystal)

    crystal = unskewCell(crystal)

    crystal = LJ_vcrelax_alternative(crystal, isotropic_external_pressure=20, debug=True)

    crystal = unskewCell(crystal)

    thr = np.min([z2Covalentradius[z] for z in sites_z])

    if isLayered(crystal,cutoff=thr*1.5, aspect_ratio=0.75):
        crystal = LJ_vcrelax_alternative(crystal, isotropic_external_pressure=200, debug=True)
        crystal = unskewCell(crystal)

    kwargs = dict(sg=sg, wyckoff_letters=wyckoff_letters,sites_z=sites_z)
    kwargs.update(**atoms2dict(initial_crystal))
    sym_data = spg.get_symmetry_dataset(crystal)
    kwargs.update(**dict(sg_spg=sym_data['number'], wyckoffs_spg=sym_data['wyckoffs'], equivalent_atoms_spg=sym_data['equivalent_atoms']))
    fout.dump_frames([crystal], [kwargs])

    return True

def atoms2dict(crystal):
    positions = crystal.get_positions()
    cell = crystal.get_cell()
    numbers = crystal.get_atomic_numbers()
    pbc = crystal.get_pbc()
    return dict(numbers=numbers, cell=cell, positions=positions, pbc=pbc)

from Pool.mpi_pool import MPIPool
from libs.io import Frame_Dataset_h5

if __name__ == '__main__':

    pool = MPIPool()
    seed = 10
    print seed+pool.rank
    np.random.seed(seed+pool.rank)

    comm = pool.comm
    basename = './structures/relaxed_structures_r'

    if not pool.is_master():
        rank = comm.Get_rank()
        fout = Frame_Dataset_h5(basename + str(rank) + '.h5')
        print 'Dumping structures to {}\n'.format(fout.fname)
        pool.wait()
        sys.exit(0)



    # with open('./structures/structures_downsampled.pck', 'rb') as f:
    #     crystals = pck.load(f)


    # np.random.seed(10)

    sites_z = [14]

    new_cc = pool.map(generate_crystal,[sites_z for it in range(10000)])

    # from ase.visualize import view
    #
    # view(new_cc)

    # soap_params = dict(nmax=9, cutoff=4, gaussian_width=0.4, lmax=9,
    #                    centerweight=1., cutoff_transition_width=0.5,
    #                    nocenters=[], is_fast_average=True, chem_channels=False, dispbar=True
    #                    )
    # nprocess = 1
    #
    # new_crystals = []
    # new_crystals.extend(crystals)
    # new_crystals.extend(new_cc)
    #
    # fings = get_fingerprints([ase2qp(crystal) for crystal in new_crystals], soap_params, nprocess)
    # kernel = np.dot(fings, fings.T)
    #
    # fps_ids, minmax = fpsSelection_with_restart(data=kernel, distance_func=distance_func2, restart_ref=None,
    #                                             disable_pbar=True, nthread=10,
    #                                             intermediate_copy=True, stride=10000,
    #                                             threshold=5e-3, Nmin=0,
    #                                             Nmax=kernel.shape[0], seed=None,
    #                                             fn=None)
    #
    # new_crystals = [new_crystals[it] for it in fps_ids]
    #


    # with open('./structures/structures_new.pck', 'wb') as f:
    #     pck.dump(new_cc, f, protocol=pck.HIGHEST_PROTOCOL)


    pool.close()