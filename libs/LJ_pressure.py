from __future__ import division
import numpy as np
from utils import ase2qp,qp2ase,isLayered
from processify import processify
from raw_data import z2Covalentradius,separationData,z2epsilon,GPa2eV_per_A3
from utils import stdchannel_to_null
from scipy.spatial.distance import pdist,cdist,squareform
from ase.neighborlist import NeighborList
import gc
timeout = 10

def LJ_vcrelax(crystal,isotropic_external_pressure=1e-2,debug=False):
    ''' isotropic_external_pressure is in GPa'''
    from quippy.potential import Potential

    # do a copy and change the object type
    crystal = ase2qp(crystal)

    # get the string to setup the quippy LJ potential (parameters and species)
    LJ_parameters = get_LJ_parameters(crystal)
    max_cutoff = LJ_parameters['cutoffs'].max()
    param_str = make_LJ_input(crystal, LJ_parameters)

    pot = Potential('IP LJ', param_str=param_str)
    # supress output from quippy minimisation routine
    with stdchannel_to_null(disable=debug):
        crystal.set_calculator(pot)
        crystal.set_cutoff(max_cutoff, 0.5)

        # use ASE fire implementation to relax the internal d.o.g. to make sure atoms are not too close to each other
        # when optimizing with quippy's fire implemnetation (it crashes otherwise)
        crystal = vc_relax_ase(crystal, fmax=1e1, steps=1e5)

        ## 1st round of vc relax with external isotropic pressure
        cc = vc_relax_qp(crystal,fmax=5e-3, steps=1e5,
                     isotropic_external_pressure=isotropic_external_pressure)
        # if quippy failled to return before timeout cc is None
        if cc is None:
            cc = vc_relax_ase(crystal,fmax=5e-3, steps=1e5,
                         isotropic_external_pressure=isotropic_external_pressure)

        ## 2nd round of relaxation without external pressure
        cc.set_calculator(pot)
        cc.set_cutoff(max_cutoff, 0.5)
        cc = vc_relax_qp(cc,fmax=7e-5, steps=1e5)

        if cc is None:
            crystal = vc_relax_ase(crystal, fmax=7e-5, steps=1e5)

            crystal = qp2ase(crystal)
        else:
            crystal = qp2ase(cc)

    return crystal


def LJ_vcrelax_alternative(crystal, isotropic_external_pressure=20, debug=False):
    ''' isotropic_external_pressure is in GPa'''
    from quippy.potential import Potential

    # do a copy and change the object type
    dd = ase2qp(crystal)

    # get the string to setup the quippy LJ potential (parameters and species)
    LJ_parameters = get_LJ_parameters(dd)
    max_cutoff = max(LJ_parameters['cutoffs'].values()) * 1.1
    param_str = make_LJ_input(dd, LJ_parameters)

    pot = Potential('IP LJ', param_str=param_str)

    sites_z = np.unique(crystal.get_atomic_numbers())
    thr = np.min([z2Covalentradius[z] for z in sites_z])

    # supress output from quippy minimisation routine
    with stdchannel_to_null(disable=debug):
        dd.set_calculator(pot)
        dd.set_cutoff(max_cutoff, 0.5)

        sep = AtomSeparator(dd)
        sep.run(Nmax=20)

        # dd = vc_relax_ase(dd, fmax=1e1, steps=1e5)
        for iii in range(500):
            if isLayered(dd, cutoff=thr * 1.5, aspect_ratio=0.75):
                vc_relax_ase(dd, fmax=5e-1, steps=1e4)

                vc_relax_ase(dd, isotropic_external_pressure=isotropic_external_pressure,
                             relax_positions=False, fmax=5e-5, steps=1e3)
            else:
                break

        vc_relax_ase(dd, fmax=5e-4, steps=1e4)
        # dd = vc_relax_ase(dd, fmax=1e1, steps=1e5)
        # ## 1st round of vc relax with external isotropic pressure
        # # if isLayered(dd):
        # vc_relax_qp(dd,fmax=5e-2, steps=1e5,relax_positions=True,
        #              isotropic_external_pressure=isotropic_external_pressure)
        # dd.set_calculator(pot)
        # dd.set_cutoff(max_cutoff, 0.5)
        ## 2nd round of relaxation without external pressure
        # crystal = vc_relax_qp(dd,fmax=7e-5, steps=1e5)
        # vc_relax_qp(dd, fmax=5e-3, steps=1e5)
    cc = qp2ase(dd)
    del dd
    gc.collect()
    return cc


# @processify(timeout=timeout)
def vc_relax_qp( crystal, relax_positions=True,isotropic_external_pressure=None,fmax=5e-3, steps=5e4):
    ''' isotropic_external_pressure is in GPa'''
    from quippy.potential import Minim
    import numpy as np

    if isotropic_external_pressure == 0. or isotropic_external_pressure is None:
        pressure_tensor = None
    else:
        pressure_tensor = np.eye(3) * isotropic_external_pressure * GPa2eV_per_A3

    minimiser = Minim(crystal, relax_positions=relax_positions, relax_cell=True, logfile='-', method='fire',
                      external_pressure=pressure_tensor, eps_guess=0.2, fire_dt0=0.1, fire_dt_max=1.0, use_precond=None)

    minimiser.run(fmax=fmax, steps=steps)
    #print minimiser.nsteps
    crystal.wrap()

    return crystal


def vc_relax_ase(crystal, relax_positions=True, isotropic_external_pressure=None, fmax=5e-3, steps=5e4):
    ''' isotropic_external_pressure is in GPa'''
    from libs.custom_unitcellfilter import UnitCellFilter, StrainFilter
    from libs.raw_data import GPa2eV_per_A3
    # from ase.constraints import StrainFilter
    # from ase.constraints import FixAtoms
    from ase.optimize import FIRE

    # if relax_positions is False:
    #     c = FixAtoms(mask=[True,]*crystal.get_number_of_atoms())
    #     crystal.set_constraint(c)
    if isotropic_external_pressure is not None:
        isotropic_external_pressure *= GPa2eV_per_A3

    if relax_positions:
        V = crystal.get_volume()
        N = crystal.get_number_of_atoms()
        J = V ** (1 / 3.) * N ** (1 / 6.)
        ucf = UnitCellFilter(crystal, mask=[1, 1, 1, 1, 1, 1], cell_factor=V / J, hydrostatic_strain=False,
                             constant_volume=False,
                             isotropic_pressure=isotropic_external_pressure)
    else:
        ucf = StrainFilter(crystal, isotropic_pressure=isotropic_external_pressure)
    dyn = FIRE(ucf, logfile=None)

    dyn.run(fmax=fmax, steps=steps)

    crystal.wrap()

    return crystal


def get_LJ_parameters(crystal):
    # TODO: change value with formation energy and mixing rules for QMAT-2
    # https://en.wikipedia.org/wiki/Gas_constant
    # https://en.wikipedia.org/wiki/Lennard-Jones_potential
    # https://en.wikipedia.org/wiki/Combining_rules
    # using Lorentz-Berthelot rules to combine the LJ parameters

    atomic_nums = np.unique(crystal.get_atomic_numbers())
    fac = 2 ** (1. / 6.)
    sigmas = {}
    epsilons = {}
    cutoffs = {}
    for z1 in atomic_nums:
        for z2 in atomic_nums:
            sigma = np.mean([2 * z2Covalentradius[z1] / fac,2 * z2Covalentradius[z2] / fac])
            epsilon = np.sqrt(np.multiply(z2epsilon[z1],z2epsilon[z2]))

            sigmas[z1,z2] = sigma
            epsilons[z1,z2] = epsilon
            cutoffs[z1,z2] = 3. * sigma
    return dict(sigmas=sigmas, epsilons=epsilons, cutoffs=cutoffs)


def make_LJ_input(crystal, LJ_parameters):
    atomic_nums = np.unique(crystal.get_atomic_numbers())
    n_types = len(atomic_nums)
    types = range(1, n_types + 1)

    sigmas, epsilons, cutoffs = LJ_parameters['sigmas'], LJ_parameters['epsilons'], LJ_parameters['cutoffs']

    param_str = []
    param_str.append('<LJ_params n_types="{}" label="default">'.format(n_types))
    for tp, atomic_num in zip(types, atomic_nums):
        param_str.append('<per_type_data type="{}" atomic_num="{}" />'.format(tp, atomic_num))

    for  tp1, z1 in zip( types, atomic_nums):
        for  tp2, z2 in zip( types, atomic_nums):
            if tp1 > tp2:
                continue
            ss = '<per_pair_data type1="{}" type2="{}" sigma="{}" eps6="{}" eps12="{}" cutoff="{}" energy_shift="F" linear_force_shift="F" />'.format(
                tp1, tp2, sigmas[z1,z2], epsilons[z1,z2], epsilons[z1,z2], cutoffs[z1,z2])
            param_str.append(ss)

    param_str.append('</LJ_params>')

    return ' '.join(param_str)


class AtomSeparator(object):
    def __init__(self, atoms, tol=1e-6):
        super(AtomSeparator, self).__init__()
        self.atoms = atoms
        self.numbers = atoms.get_atomic_numbers()
        self.Natom = atoms.get_number_of_atoms()
        self.nl = NeighborList([z2Covalentradius[z] for z in self.numbers], self_interaction=False, bothways=False)
        self.tol = tol

    def run(self, Nmax=50):
        ii = 0
        fff = True
        while fff:
            fff = self.step()
            atoms = self.atoms

            self.atoms = vc_relax_ase(atoms, relax_positions=False,
                                      fmax=5e-2, steps=2)
            ii += 1
            if ii >= Nmax:
                fff = False

    def step(self):
        atoms = self.atoms
        nl = self.nl

        nl.update(atoms)

        Natom = self.Natom
        numbers = self.numbers
        dr = np.zeros((Natom, 3))
        r = atoms.get_positions()
        cell = atoms.get_cell()

        for icenter in range(Natom):

            indices, offsets = nl.get_neighbors(icenter)
            icenter_pos = r[icenter].reshape((1, 3))
            neighbor_pos = np.zeros((len(indices), 3))
            refSep = np.zeros((len(indices),1))
            for it, (ineighbor, offset) in enumerate(zip(indices, offsets)):
                neighbor_pos[it] = r[ineighbor] + np.dot(offset, cell)
                refSep[it] = separationData[(numbers[icenter], numbers[ineighbor])]

            sepVec = np.subtract(icenter_pos, neighbor_pos)
            sep = np.linalg.norm(sepVec, axis=1).reshape((-1, 1))
            sepDiff = np.subtract(refSep, sep)

            move = np.multiply(0.5, np.multiply(np.divide(sepDiff, sep), sepVec))

            for it, ineighbor in enumerate(indices):
                dr[icenter] += move[it]
                dr[ineighbor] -= move[it]

        if np.linalg.norm(dr) < self.tol:
            return False
        else:
            atoms.set_positions(r + dr)
            atoms.wrap()
            return True


def LJ_vcrelax_ase_simple( crystal, isotropic_external_pressure=1e-2):
    from ase.optimize import FIRE
    import numpy as np
    from quippy.potential import Potential
    # from ase.constraints import UnitCellFilter
    from libs.custom_unitcellfilter import UnitCellFilter
    # do a copy and change the object type
    crystal = ase2qp(crystal)

    LJ_parameters = get_LJ_parameters(crystal)
    max_cutoff = LJ_parameters['cutoffs'].max()

    param_str = make_LJ_input(crystal,LJ_parameters)

    pot = Potential('IP LJ', param_str=param_str)
    crystal.set_calculator(pot)
    crystal.set_cutoff(max_cutoff, 0.5)



    V = crystal.get_volume()
    N = crystal.get_number_of_atoms()
    J = V ** (1 / 3.) * N ** (1 / 6.)
    ucf = UnitCellFilter(crystal, mask=[1, 1, 1, 1, 1, 1], cell_factor=V / J, hydrostatic_strain=False,
                         constant_volume=False, isotropic_pressure=isotropic_external_pressure)
    dyn = FIRE(ucf, logfile=None)

    dyn.run(**{'fmax': 1e-6, 'steps': 1e5})

    crystal.wrap()

    return crystal


def LJ_vcrelax_qp_simple( crystal, isotropic_external_pressure=1e-2):
    from quippy.potential import Potential, Minim
    import numpy as np

    crystal = ase2qp(crystal)

    if isotropic_external_pressure == 0.:
        pressure_tensor = None
    else:
        pressure_tensor = np.eye(3) * isotropic_external_pressure
    # get the string to setup the quippy LJ potential (parameters and species)
    sites_z = np.unique(crystal.get_atomic_numbers())
    param_str, max_cutoff = make_LJ_input(sites_z)

    pot = Potential('IP LJ', param_str=param_str)
    crystal.set_calculator(pot)
    crystal.set_cutoff(max_cutoff, 0.5)

    minimiser = Minim(crystal, relax_positions=True, relax_cell=True, logfile='-', method='fire',
                      external_pressure=pressure_tensor, eps_guess=0.2, fire_dt0=0.1, fire_dt_max=1.0, use_precond=None)

    minimiser.run(fmax=1e-6, steps=1e6)

    crystal.wrap()

    return crystal


def test():
    from input_structure import input2crystal
    from utils import unskewCell
    seed = 2
    vdw_ratio = 1.5
    sites_z = [14]

    crystal, sg, wki = input2crystal(sites_z, seed, vdw_ratio)
    crystal = unskewCell(crystal)

    p = 0.

    cc_qp = LJ_vcrelax_qp_simple( crystal, isotropic_external_pressure=p)

    cc_ase = LJ_vcrelax_ase_simple( crystal, isotropic_external_pressure=p)

    print 'Are the relaxed structures identical ?'
    print np.allclose(cc_qp.get_scaled_positions(), cc_ase.get_scaled_positions(), atol=1e-5)
    print np.allclose(cc_qp.get_cell(), cc_ase.get_cell(), atol=1e-5)

if __name__ == '__main__':
    test()

