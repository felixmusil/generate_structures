import numpy as np
import numpy.random as npr



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

    return [a, b, c, alpha, beta, gamma]