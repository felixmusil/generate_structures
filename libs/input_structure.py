import numpy as np
import numpy.random as npr
from ase.spacegroup import crystal
from utils import getCellParam
from ..reference_info.raw_data import SG2BravaisLattice,z2symb,z2VdWradius

def input2crystal(sites_z ,seed ,vdw_ratio, WyckTable):
    '''
    
    :param sites_z: 
    :param seed: 
    :param vdw_ratio: 
    :param WyckTable: 
    :return: 
    '''
    npr.seed(seed)

    # Return random integers from `low` (inclusive) to `high` (exclusive).
    sg = npr.randint(1 ,high= 230 +1)

    bravaisLattice = SG2BravaisLattice[sg]

    asym_positions = []
    Natoms = []
    VdW_radii = []
    symbols = []
    wyckoff_letters = []
    for site_z in sites_z:

        symbols.append(z2symb[site_z])

        # get a site generator at random (random wyckoff site)
        st ,nd = 0 ,len(WyckTable[sg])
        wyckoff_site_idx = npr.randint(st ,high=nd)
        wyckoff_letters.append(WyckTable[sg]['Wyckoff letter'][wyckoff_site_idx])
        site_generator = WyckTable[sg]['Site generator'][wyckoff_site_idx]

        cc = crystal(symbols=['Si'], basis=[0 ,0.4 ,0.6], spacegroup=sg, cellpar=[1 ,1 ,1 ,90 ,90 ,90] ,symprec=1e-5, pbc=True)
        Natoms.append(cc.get_number_of_atoms())
        VdW_radii.append(z2VdWradius[site_z])
        # get rand xyz position from 0.1 to 0.9 with rounded at 3 decimals
        x ,y ,z = np.around(npr.random((3,) ) *(0.9 - 0.1) + 0.1, decimals=3)

        asym_position = []
        for gen in site_generator:
            if isinstance(gen, float):
                asym_position.append(gen)
            else:
                asym_position.append(eval(gen))
        asym_positions.append(asym_position)

    min, max = 4, 10
    a, b, c = np.around(npr.random((3,)) * (max - min) + min, decimals=3)
    min, max = 30, 160
    alpha, beta, gamma = np.around(npr.random((3,)) * (max - min) + min, decimals=0)

    cellparam_guess = a, b, c, alpha, beta, gamma

    cellparam = getCellParam(cellparam_guess, vdw_ratio, VdW_radii, Natoms, bravaisLattice)

    cc = crystal(symbols=symbols, basis=asym_positions, spacegroup=sg, cellpar=cellparam, symprec=1e-7, pbc=True)

    return cc, sg, wyckoff_letters