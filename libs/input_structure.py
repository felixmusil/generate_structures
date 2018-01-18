import numpy as np
import numpy.random as npr
from raw_data import SG2BravaisLattice,z2symb,z2VdWradius,WyckTable



def input2crystal(sites_z ,seed ,vdw_ratio, sg=None):
    '''
    Generate a random atomic structure through space group and wyckoff sites
    :param sites_z: list of atomic numbers (corresponds to the number of asymmetric sites)
    :param seed: seed for random number generation
    :param vdw_ratio: cell filling (1.5 is compact cell and 0.5 is spread cell)
    :param WyckTable: dictionary of pandas dataframes from ./reference_info/SpaceGroup-multiplicity-wickoff-info.pck
    :return: ase.Atoms , space group idx, list of wyckoff letters
    '''
    from ase.spacegroup import crystal

    npr.seed(seed)

    # Return random integers from `low` (inclusive) to `high` (exclusive).
    if sg is None:
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
        st ,nd = 0 ,len(WyckTable[sg]['Wyckoff letter'].keys())
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



    min, max = 30, 150
    # James Foadi et Gwyndaf Evans,  On the allowed values for the triclinic unit-cell angles  DOI 10.1107/S0108767310044296
    alpha, beta, gamma = np.around(npr.random((3,)) * (max - min) + min, decimals=0)
    if sg < 3:
        while(not triclinic_conditions(alpha, beta, gamma)):
            alpha, beta, gamma = np.around(npr.random((3,)) * (max - min) + min, decimals=0)

    min, max = 4, 10
    a, b, c = np.around(npr.random((3,)) * (max - min) + min, decimals=3)
    cellparam_guess = a, b, c, alpha, beta, gamma
    cellparam = getCellParam(cellparam_guess, vdw_ratio, VdW_radii, Natoms, bravaisLattice)

    # makes sure a does not become too small because b or c are too large
    while np.any(cellparam[:3] < 1.):
        a, b, c = np.around(npr.random((3,)) * (max - min) + min, decimals=3)
        cellparam_guess = a, b, c, alpha, beta, gamma
        cellparam = getCellParam(cellparam_guess, vdw_ratio, VdW_radii, Natoms, bravaisLattice)

    cc = crystal(symbols=symbols, basis=asym_positions, spacegroup=sg,
                 cellpar=cellparam, symprec=1e-7, pbc=True,primitive_cell=False)

    return cc, sg, wyckoff_letters


def getCellParam(cellparam, cellFilling, Rs, Na, bravaisLattice):
    '''
    Take a cellparam guess and return cellparam compatible with
    :param cellparam:  
    :param cellFilling: 
    :param Rs: 
    :param Na: 
    :param bravaisLattice: 
    :return: 
    '''
    import numpy as np

    V_tot = 0
    for r, n in zip(Rs, Na):
        V_tot += n * 4 / 3. * np.pi * r ** 3
    # total volume filled by the atoms divided by the filling ratio (using Van der Walls radius, this ratio is usually 1 since it already takes into account bounds)
    fac = V_tot / cellFilling
    a, b, c, alpha, beta, gamma = cellparam

    d2r = np.pi / 180.
    if bravaisLattice in ['cubic F', 'cubic I', 'cubic P']:
        a = np.power(fac, 1 / 3.)
        alpha = beta = gamma = 90.
        b = c = a
    elif bravaisLattice in ['hexagonal P', 'trigonal P', 'trigonal R']:
        a = np.sqrt(fac / c * np.sqrt(2 / 3.))
        alpha = beta = 90.
        gamma = 120.
        b = a
    elif bravaisLattice in ['tetragonal I', 'tetragonal P']:
        a = np.sqrt(fac / c)
        alpha = beta = gamma = 90.
        b = a
    elif bravaisLattice in ['orthorhombic A', 'orthorhombic C', 'orthorhombic F', 'orthorhombic I', 'orthorhombic P']:
        a = fac / b / c
        alpha = beta = gamma = 90.
    elif bravaisLattice in ['monoclinic C', 'monoclinic P', 'orthorhombic A']:
        a = fac / b / c / np.sqrt(1 - np.cos(beta * d2r) ** 2)
        alpha = gamma = 90.
    elif bravaisLattice == 'triclinic P':
        ca, cb, cg = np.cos(alpha * d2r), np.cos(beta * d2r), np.cos(gamma * d2r)
        ee = b * c * np.sqrt(1 - ca ** 2 - cb ** 2 - cg ** 2 + 2 * ca * cb * cg)
        a = fac / ee
    else:
        raise Exception('Error with the Bravais lattice')

    return np.array([a, b, c, alpha, beta, gamma])


def triclinic_conditions(alpha,beta,gamma):
    a = alpha + beta + gamma
    b = alpha + beta - gamma
    c = alpha - beta + gamma
    d =-alpha + beta + gamma
    return np.all([0<a<360,0<b<360,0<c<360,0<d<360])