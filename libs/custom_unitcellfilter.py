from ase.constraints import Filter,voigt_6_to_full_3x3_stress
import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError

description = '''
Small 
'''

class UnitCellFilter(Filter):
    """Modify the supercell and the atom positions. """
    def __init__(self, atoms, mask=None,cell_factor=None,hydrostatic_strain=False,
                 constant_volume=False,isotropic_pressure=None):
        """Create a filter that returns the atomic forces and unit cell
        stresses together, so they can simultaneously be minimized.

        The first argument, atoms, is the atoms object. The optional second
        argument, mask, is a list of booleans, indicating which of the six
        independent components of the strain are relaxed.

        - True = relax to zero
        - False = fixed, ignore this component

        Degrees of freedom are the positions in the original undeformed cell,
        plus the deformation tensor (extra 3 "atoms"). This gives forces
        consistent with numerical derivatives of the potential energy
        with respect to the cell degreees of freedom.

        Helpful conversion table:

        - 0.05 eV/A^3   = 8 GPA
        - 0.003 eV/A^3  = 0.48 GPa
        - 0.0006 eV/A^3 = 0.096 GPa
        - 0.0003 eV/A^3 = 0.048 GPa
        - 0.0001 eV/A^3 = 0.02 GPa

        Additional optional arguments:

        cell_factor: float (default float(len(atoms)))
            Factor by which deformation gradient is multiplied to put
            it on the same scale as the positions when assembling
            the combined position/cell vector. The stress contribution to
            the forces is scaled down by the same factor. This can be thought
            of as a very simple preconditioners. Default is number of atoms
            which gives approximately the correct scaling.

        hydrostatic_strain: bool (default False)
            Constrain the cell by only allowing hydrostatic deformation.
            The virial tensor is replaced by np.diag([np.trace(virial)]*3).

        constant_volume: bool (default False)
            Project out the diagonal elements of the virial tensor to allow
            relaxations at constant volume, e.g. for mapping out an
            energy-volume curve. Note: this only approximately conserves
            the volume and breaks energy/force consistency so can only be
            used with optimizers that do require do a line minimisation
            (e.g. FIRE).
        """

        Filter.__init__(self, atoms, indices=range(len(atoms)))
        self.atoms = atoms
        self.deform_grad = np.eye(3)
        self.atom_positions = atoms.get_positions()
        self.orig_cell = atoms.get_cell()
        self.stress = None
        if isotropic_pressure is None or isotropic_pressure == 0.:
            self.external_pressure = None
        else:
            self.external_pressure = np.diag([isotropic_pressure,]*3)

        if mask is None:
            mask = np.ones(6)
        mask = np.asarray(mask)
        if mask.shape == (6,):
            self.mask = voigt_6_to_full_3x3_stress(mask)
        elif mask.shape == (3, 3):
            self.mask = mask
        else:
            raise ValueError('shape of mask should be (3,3) or (6,)')

        if cell_factor is None:
            cell_factor = float(len(atoms))
        self.hydrostatic_strain = hydrostatic_strain
        self.constant_volume = constant_volume
        self.cell_factor = cell_factor
        self.copy = self.atoms.copy
        self.arrays = self.atoms.arrays

    def get_positions(self):
        '''
        this returns an array with shape (natoms + 3,3).

        the first natoms rows are the positions of the atoms, the last
        three rows are the deformation tensor associated with the unit cell,
        scaled by self.cell_factor.
        '''

        natoms = len(self.atoms)
        pos = np.zeros((natoms + 3, 3))
        pos[:natoms] = self.atom_positions
        pos[natoms:] = self.cell_factor * self.deform_grad
        return pos

    def set_positions(self, new, **kwargs):
        '''
        new is an array with shape (natoms+3,3).

        the first natoms rows are the positions of the atoms, the last
        three rows are the deformation tensor used to change the cell shape.

        the positions are first set with respect to the original
        undeformed cell, and then the cell is transformed by the
        current deformation gradient.
        '''

        natoms = len(self.atoms)
        self.atom_positions[:] = new[:natoms]
        self.deform_grad = new[natoms:] / self.cell_factor
        self.atoms.set_positions(self.atom_positions, **kwargs)
        self.atoms.set_cell(self.orig_cell, scale_atoms=False)
        self.atoms.set_cell(np.dot(self.orig_cell, self.deform_grad.T),
                            scale_atoms=True)

    def get_forces(self, apply_constraint=False):
        '''
        returns an array with shape (natoms+2,3) of the atomic forces
        and unit cell stresses.

        the first natoms rows are the forces on the atoms, the last
        three rows are the forces on the unit cell, which are
        computed from the stress tensor.
        '''

        atoms_forces = self.atoms.get_forces()
        stress = self.atoms.get_stress()
        self.stress = voigt_6_to_full_3x3_stress(stress) * self.mask

        if self.external_pressure is not None:
            self.stress += self.external_pressure

        volume = self.atoms.get_volume()
        virial = -volume * voigt_6_to_full_3x3_stress(stress)
        atoms_forces = np.dot(atoms_forces, self.deform_grad)
        dg_inv = np.linalg.inv(self.deform_grad)
        virial = np.dot(virial, dg_inv.T)

        if self.hydrostatic_strain:
            vtr = virial.trace()
            virial = np.diag([vtr / 3.0, vtr / 3.0, vtr / 3.0])

        # Zero out components corresponding to fixed lattice elements
        if (self.mask != 1.0).any():
            virial *= self.mask

        if self.constant_volume:
            vtr = virial.trace()
            np.fill_diagonal(virial, np.diag(virial) - vtr / 3.0)

        natoms = len(self.atoms)
        forces = np.zeros((natoms + 3, 3))
        forces[:natoms] = atoms_forces
        forces[natoms:] = virial / self.cell_factor
        return forces

    def get_stress(self):
        raise PropertyNotImplementedError

    def has(self, x):
        return self.atoms.has(x)

    def __len__(self):
        return (len(self.atoms) + 3)