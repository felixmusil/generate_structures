from __future__ import division
import numpy as np
from utils import ase2qp,qp2ase
from processify import processify
from raw_data import z2Covalentradius
from utils import stdchannel_to_null

timeout = 3

def LJ_vcrelax(crystal,isotropic_external_pressure=1e-2,debug=False):
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
        cc = vc_relax_qp(cc,fmax=7e-5, steps=1e5,
                         isotropic_external_pressure=1e-5)

        if cc is None:
            crystal = vc_relax_ase(crystal, fmax=7e-5, steps=1e5)

            crystal = qp2ase(crystal)
        else:
            crystal = qp2ase(cc)

    return crystal

@processify(timeout=timeout)
def vc_relax_qp( crystal, isotropic_external_pressure=None,fmax=5e-3, steps=5e4):
    from quippy.potential import Minim
    import numpy as np

    if isotropic_external_pressure == 0. or isotropic_external_pressure is None:
        pressure_tensor = None
    else:
        pressure_tensor = np.eye(3) * isotropic_external_pressure


    minimiser = Minim(crystal, relax_positions=True, relax_cell=True, logfile='-', method='fire',
                      external_pressure=pressure_tensor, eps_guess=0.2, fire_dt0=0.1, fire_dt_max=1.0, use_precond=None)

    minimiser.run(fmax=fmax, steps=steps)

    crystal.wrap()

    return crystal

def vc_relax_ase( crystal, isotropic_external_pressure=None,fmax=5e-3, steps=5e4):
    from libs.custom_unitcellfilter import UnitCellFilter
    from ase.optimize import FIRE

    V = crystal.get_volume()
    N = crystal.get_number_of_atoms()
    J = V ** (1 / 3.) * N ** (1 / 6.)
    ucf = UnitCellFilter(crystal, mask=[1, 1, 1, 1, 1, 1], cell_factor=V / J, hydrostatic_strain=False,
                         constant_volume=False, isotropic_pressure=isotropic_external_pressure)
    dyn = FIRE(ucf, logfile=None)

    dyn.run(fmax=fmax, steps=steps)

    crystal.wrap()

    return crystal


def get_LJ_parameters(crystal):
    # TODO: change value with formation energy and mixing rules for QMAT-2
    # https://en.wikipedia.org/wiki/Gas_constant
    # https://en.wikipedia.org/wiki/Lennard-Jones_potential
    # https://en.wikipedia.org/wiki/Combining_rules
    atomic_nums = np.unique(crystal.get_atomic_numbers())
    fac = 2 ** (1. / 6.)
    if len(atomic_nums) == 1:
        sigmas = np.asarray([[2 * z2Covalentradius[atomic_nums[0]] / fac]])
        epsilons = np.asarray([[1.0]])
        cutoffs = 3. * sigmas

    return dict(sigmas=sigmas, epsilons=epsilons, cutoffs=cutoffs)


def make_LJ_input(crystal, LJ_parameters):
    atomic_nums = np.unique(crystal.get_atomic_numbers())
    n_types = len(atomic_nums)
    types = range(1, n_types + 1)
    ids = range(n_types)
    sigmas, epsilons, cutoffs = LJ_parameters['sigmas'], LJ_parameters['epsilons'], LJ_parameters['cutoffs']

    param_str = []
    param_str.append('<LJ_params n_types="{}" label="default">'.format(n_types))
    for tp, atomic_num in zip(types, atomic_nums):
        param_str.append('<per_type_data type="{}" atomic_num="{}" />'.format(tp, atomic_num))

    for it, tp1, atomic_num1 in zip(ids, types, atomic_nums):
        for jt, tp2, atomic_num2 in zip(ids, types, atomic_nums):
            if tp1 > tp2:
                continue
            ss = '<per_pair_data type1="{}" type2="{}" sigma="{}" eps6="{}" eps12="{}" cutoff="{}" energy_shift="F" linear_force_shift="F" />'.format(
                tp1, tp2, sigmas[it, jt], epsilons[it, jt], epsilons[it, jt], cutoffs[it, jt])
            param_str.append(ss)

    param_str.append('</LJ_params>')

    return ' '.join(param_str)


def LJ_vcrelax_ase_simple( crystal, isotropic_external_pressure=1e-2):
    from ase.optimize import FIRE
    import numpy as np
    from quippy.potential import Potential
    # from ase.constraints import UnitCellFilter
    from libs.custom_unitcellfilter import UnitCellFilter
    # do a copy and change the object type
    crystal = ase2qp(crystal)
    sites_z = np.unique(crystal.get_atomic_numbers())
    param_str, max_cutoff = make_LJ_input(sites_z)

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

