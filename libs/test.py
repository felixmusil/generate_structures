import cPickle as pck
from time import time

import numpy as np
from ase.optimize import FIRE
from quippy.potential import Potential, Minim

from input_structure import input2crystal
from LJ_pressure import make_LJ_input
from utils import ase2qp, s2hms, unskewCell
from utils import stdchannel_to_null
from ase.constraints import UnitCellFilter


infoPath = '../reference_info/'
fileNames = {}
fileNames['wyck'] = infoPath+'SpaceGroup-multiplicity-wickoff-info.pck'
with open(fileNames['wyck'],'rb') as f:
    WyckTable = pck.load(f)

vdw_ratio = 1.5
sites_z = [14]




st = time()
with stdchannel_to_null(disable=False):
    for seed in range(1,100):
        print '#################',seed

        crystal,sg,wki = input2crystal(sites_z,seed,vdw_ratio,WyckTable)
        pr = 1e-0
        pressure = np.eye(3)*pr

        param_str = make_LJ_input(sites_z)
        cc_qp = ase2qp(unskewCell(crystal))
        pot = Potential('IP LJ',param_str=param_str)
        cc_qp.set_calculator(pot)
        cc_qp.set_cutoff(2.,0.5)
        Natom = cc_qp.get_number_of_atoms()

        # V = crystal.get_volume()
        # N = crystal.get_number_of_atoms()
        # J = V ** (1 / 3.) * N ** (1 / 6.)
        # ucf = UnitCellFilter(cc_qp, mask=[1, 1, 1, 1, 1, 1], cell_factor=V / J, hydrostatic_strain=False,
        #                      constant_volume=False)
        # dyn = FIRE(ucf, logfile=None)
        #
        # dyn.run(**{'fmax': 1e-5, 'steps': 1e6})

        dyn = FIRE(cc_qp, logfile=None)
        mf = np.linalg.norm(cc_qp.get_forces(), axis=1).max()
        while mf > 1e5:
            dyn.run(**{'fmax': 3e-2, 'steps': 5})
            mf = np.linalg.norm(cc_qp.get_forces(), axis=1).max()
        print mf
        minimiser = Minim(cc_qp, relax_positions=True, relax_cell=True,logfile='-',method='fire',
                          external_pressure=None,eps_guess=0.2, fire_dt0=0.1, fire_dt_max=1.0,use_precond=None)

        minimiser.run(fmax=1e-5,steps=1e6)




print 'Elapsed time ',s2hms(time()-st)