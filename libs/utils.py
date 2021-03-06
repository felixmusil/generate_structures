import contextlib
import os
from ase.neighborlist import NeighborList

import numpy as np
import spglib as spg


def s2hms(time):
    m = time // 60
    s = int(time % 60)
    h = int(m // 60)
    m = int(m % 60)
    return '{:02d}:{:02d}:{:02d} (h:m:s)'.format(h,m,s)

def qp2ase(qpatoms):
    from ase import Atoms as aseAtoms
    positions = qpatoms.get_positions()
    cell = qpatoms.get_cell()
    numbers = qpatoms.get_atomic_numbers()
    pbc = qpatoms.get_pbc()
    atoms = aseAtoms(numbers=numbers, cell=cell, positions=positions, pbc=pbc)

    for key, item in qpatoms.arrays.iteritems():
        if key in ['positions', 'numbers', 'species', 'map_shift', 'n_neighb']:
            continue
        atoms.set_array(key, item)

    return atoms

def ase2qp(aseatoms):
    from quippy import Atoms as qpAtoms
    positions = aseatoms.get_positions()
    cell = aseatoms.get_cell()
    numbers = aseatoms.get_atomic_numbers()
    pbc = aseatoms.get_pbc()
    return qpAtoms(numbers=numbers,cell=cell,positions=positions,pbc=pbc)


def atoms2np(crystal):
    lattice = crystal.get_cell()
    positions = crystal.get_positions()
    numbers = crystal.get_atomic_numbers()
    return numbers,positions,lattice
def np2atoms(numbers,positions,lattice,type='ase'):
    if type == 'ase':
        from ase import Atoms as aseAtoms
        crystal = aseAtoms(numbers=numbers,cell=lattice,positions=positions)
    elif type == 'quippy':
        from quippy import Atoms as qpAtoms
        crystal = qpAtoms(numbers=numbers,cell=lattice,positions=positions)
    else:
        raise TypeError('Not recognised input type {}'.format(type))
    return crystal


def unskewCell(frame):
    lattice = frame.get_cell()
    lengths = frame.get_cell_lengths_and_angles()[:3]
    angles = np.cos(frame.get_cell_lengths_and_angles()[3:]*np.pi / 180.)

    max_angle2ij = {0:(0,1),1:(0,2),2:(1,2)}
    for max_angle in np.where(np.abs(angles)>0.5)[0]:

        i,j = max_angle2ij[max_angle]
        if lengths[i] > lengths[j]:
            i, j = j, i
        lattice[j,:] = lattice[j,:] - np.round(angles[max_angle]) * lattice[i,:]
    cc = frame.copy()
    cc.set_cell(lattice)
    cc.wrap()
    return cc

def get_standard_frame(frame,to_primitive=True,symprec=1e-5):
    '''
    Standardize the frame using spglib
    :param frame: 
    :param to_primitive: 
    :param symprec: 
    :return: 
    '''
    from ase import Atoms
    pbc = frame.get_pbc()
    (lattice, positions, numbers) = spg.standardize_cell(
                            frame, to_primitive=to_primitive,no_idealize=False,
                            symprec=symprec, angle_tolerance=-1.0)
    std_atoms = Atoms(cell=lattice, scaled_positions=positions, numbers=numbers,pbc=pbc)
    return std_atoms

def is_notebook():
    from IPython import get_ipython
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


@contextlib.contextmanager
def stdchannel_to_null(disable=False):
    # see http://www.greghaskins.com/archive/python-forcibly-redirect-stdout-stderr-from-extension-modules
    # for more details
    if not disable:
        null_fds = [os.open(os.devnull, os.O_RDWR) for x in xrange(2)]
        # save the current file descriptors to a tuple
        save = os.dup(1), os.dup(2)
        # put /dev/null fds on 1 and 2
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)

    yield

    if not disable:
        os.dup2(save[0], 1)
        os.dup2(save[1], 2)
        # close the temporary fds
        os.close(null_fds[0])
        os.close(null_fds[1])
        os.close(save[0])
        os.close(save[1])

def isTooClose(frame,threshold=2.):
    crystal = ase2qp(frame)
    if len(crystal) > 1:
        dd = crystal.get_all_distances()
        dd = dd + 2.5*threshold*np.eye(dd.shape[0])
    else:
        dd = 3.*threshold*np.ones((2,2))
    return dd[dd<threshold].size > 0

def isLayered(frame, cutoff=1.5, aspect_ratio=0.5, debug=False):

    crystal = qp2ase(frame)
    length = crystal.get_cell_lengths_and_angles()[:3]
    scaled = crystal.get_scaled_positions()
    rr = np.ones(3, int)

    for it in range(3):
        min, max = np.min(scaled[:, it]), np.max(scaled[:, it])
        mask2 = (max - min) < (1 - aspect_ratio)
        mask = length / length[it] < aspect_ratio
        if np.any(mask) or mask2:
            rr[it] = 2

    crystal = crystal.repeat(list(rr))
    if debug:
        print rr
    Nat = len(crystal)
    nl = NeighborList([cutoff, ] * Nat, skin=0.0, sorted=False, self_interaction=False, bothways=True)
    nl.build(crystal)
    indices, _ = nl.get_neighbors(0)

    to_visit = list(indices)

    visited = [0]
    while len(to_visit) > 0:
        center_id = to_visit.pop(0)
        if debug:
            print to_visit
        indices, _ = nl.get_neighbors(center_id)
        visited.append(center_id)
        ll = list(np.setdiff1d(indices, visited))
        to_visit.extend(ll)
        to_visit = list(np.unique(to_visit))

    not_visited = np.setdiff1d(np.arange(Nat), visited)
    if debug:
        print not_visited

    return len(not_visited) > 0