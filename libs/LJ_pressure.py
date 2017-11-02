from __future__ import division
import numpy as np
from utils import ase2qp,qp2ase

from raw_data import z2Covalentradius


def LJ_vcrelax(sites_z, crystal, isotropic_external_pressure=1e-2):
    from quippy.potential import Potential, Minim
    from ase.optimize import FIRE
    import numpy as np
    from ase.constraints import UnitCellFilter
    # do a copy and change the object type
    crystal = ase2qp(crystal)

    pressure_tensor = np.eye(3) * isotropic_external_pressure
    # get the string to setup the quippy LJ potential (parameters and species)
    param_str, max_cutoff = make_LJ_input(sites_z)

    pot = Potential('IP LJ', param_str=param_str)
    crystal.set_calculator(pot)
    crystal.set_cutoff(max_cutoff, 0.5)

    # use ASE fire implementation to relax the internal d.o.g. to make sure atoms are not too close to each other
    # when optimizing with quippy's fire implemnetation (it crashes otherwise)
    dyn = FIRE(crystal, logfile=None)
    max_force = np.linalg.norm(crystal.get_forces(), axis=1).max()
    max_stress = np.abs(crystal.get_stress()).max()
    # V = crystal.get_volume()
    # N = crystal.get_number_of_atoms()
    # J = V ** (1 / 3.) * N ** (1 / 6.)
    # ucf = UnitCellFilter(crystal, mask=[1, 1, 1, 1, 1, 1], cell_factor=V / J, hydrostatic_strain=False,
    #                      constant_volume=False)
    # dyn = FIRE(ucf, logfile=None)
    # max_force = np.linalg.norm(crystal.get_forces(), axis=1).max()

    # the threshold here make sure that quippy will not exit with out of memory error
    while max_force > 1e4:
        dyn.run(**{'fmax': 1e-6, 'steps': 1})
        max_force = np.linalg.norm(crystal.get_forces(), axis=1).max()
        max_stress = np.abs(crystal.get_stress()).max()

    # 1st round of vc relax with external isotropic pressure
    minimiser = Minim(crystal, relax_positions=True, relax_cell=True, logfile='-', method='fire',
                      external_pressure=pressure_tensor, eps_guess=0.2, fire_dt0=0.1, fire_dt_max=1.0, use_precond=None)

    minimiser.run(fmax=5e-2, steps=5e4)

    # 2nd round of vc relax without external isotropic pressure
    minimiser = Minim(crystal, relax_positions=True, relax_cell=True, logfile='-', method='fire',
                      external_pressure=None, eps_guess=0.2, fire_dt0=0.1, fire_dt_max=1.0, use_precond=None)

    minimiser.run(fmax=1e-6, steps=5e4)

    crystal.wrap()

    crystal = qp2ase(crystal)

    return crystal



def make_LJ_input(sites_z):

    # TODO: change value with formation energy and mixing rules for QMAT-2
    # https://en.wikipedia.org/wiki/Gas_constant
    # https://en.wikipedia.org/wiki/Lennard-Jones_potential
    # https://en.wikipedia.org/wiki/Combining_rules
    n_types = len(sites_z)
    atomic_nums = sites_z
    types = range(1 ,len(sites_z ) +1)

    fac = 2** (1. / 6.)

    param_str = []
    param_str.append('<LJ_params n_types="{}" label="default">'.format(n_types))
    for tp, atomic_num in zip(types, atomic_nums):
        param_str.append('<per_type_data type="{}" atomic_num="{}" />'.format(tp, atomic_num))
    cutoffs = []
    for tp1, atomic_num1 in zip(types, atomic_nums):
        for tp2, atomic_num2 in zip(types, atomic_nums):
            if tp1 == tp2:
                sigma = 2 * z2Covalentradius[atomic_num1] / fac
                epsilon = 1.0
                cutoff = 3. * sigma
            elif tp1 < tp2:
                continue
            else:
                continue
            cutoffs.append(cutoff)
            ss = '<per_pair_data type1="{}" type2="{}" sigma="{}" eps6="{}" eps12="{}" cutoff="{}" energy_shift="F" linear_force_shift="F" />'.format(
                tp1, tp2, sigma, epsilon, epsilon, cutoff)
            param_str.append(ss)

    param_str.append('</LJ_params>')
    max_cutoff = np.max(cutoffs)
    return ' '.join(param_str),max_cutoff