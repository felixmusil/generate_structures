import numpy as np
from quippy.potential import Potential

from libs.LJ_pressure import make_LJ_input
from libs.input_structure import input2crystal
from libs.processify import processify
from libs.utils import ase2qp

seed = 14
vdw_ratio = 1.5
sites_z = [14]


crystal,sg,wki = input2crystal(sites_z,seed,vdw_ratio)
pr = 1e-2
pressure = np.eye(3)*pr

param_str,cutoff_max = make_LJ_input(sites_z)
cc_qp = ase2qp(crystal)
pot = Potential('IP LJ',param_str=param_str)

# cc_qp.set_calculator(pot)
# minimiser = Minim(cc_qp, relax_positions=True, relax_cell=True,logfile='-',method='fire',
#                       external_pressure=None,eps_guess=0.2, fire_dt0=0.1, fire_dt_max=1.0,use_precond=None)
#
# try:
#     minimiser.run(fmax=1e0,steps=1e6)
# except:
#     print '######'

convergence_tol = 1.
steps = 10
linminroutine = None

@processify(5)
def tt():
    seed = 14
    vdw_ratio = 1.5
    sites_z = [14]

    crystal, sg, wki = input2crystal(sites_z, seed, vdw_ratio)
    pr = 1e-2
    pressure = np.eye(3) * pr

    param_str, cutoff_max = make_LJ_input(sites_z)
    cc_qp = ase2qp(crystal)
    pot = Potential('IP LJ', param_str=param_str)

    pot.minim(cc_qp, 'fire', convergence_tol, steps,
              linminroutine, do_pos=True, do_lat=True,
              args_str=pot.get_calc_args_str(), eps_guess=0.2,
              fire_minim_dt0=0.1, fire_minim_dt_max=1.0,
              external_pressure=None,
              use_precond=None)


    return cc_qp

print '#########'
cc = tt()
print '#########'
print cc
# thread = Thread(target=pot.minim,args=[cc_qp, 'fire', convergence_tol, steps,
#                linminroutine],kwargs=dict(do_pos=True, do_lat=True,
#                args_str=pot.get_calc_args_str(), eps_guess=0.2,
#                fire_minim_dt0=0.1, fire_minim_dt_max=1.0,
#                external_pressure=None,
#                use_precond=None))
#
# p = Process(target=pot.minim,args=[cc_qp, 'fire', convergence_tol, steps,
#                linminroutine],kwargs=dict(do_pos=True, do_lat=True,
#                args_str=pot.get_calc_args_str(), eps_guess=0.2,
#                fire_minim_dt0=0.1, fire_minim_dt_max=1.0,
#                external_pressure=None,
#                use_precond=None))
#
# p.start()
# try:
#
#     p.run()
#
#     p.join()
# except:
#     print '####33'
#
#
#
# pot.minim(cc_qp, 'fire', convergence_tol, steps,
#                linminroutine, do_pos=True, do_lat=True,
#                args_str=pot.get_calc_args_str(), eps_guess=0.2,
#                fire_minim_dt0=0.1, fire_minim_dt_max=1.0,
#                external_pressure=None,
#                use_precond=None)
