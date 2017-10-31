from __future__ import division
import numpy as np
import numpy
from ase.neighborlist import NeighborList
import copy,os
'''
Implementation of a LJ calculator compatible with ase with pressure.
import LJp
'''


def equal(a, b, tol=None):
    """ndarray-enabled comparison function."""
    if isinstance(a, numpy.ndarray):
        b = numpy.array(b)
        if a.shape != b.shape:
            return False
        if tol is None:
            return (a == b).all()
        else:
            return numpy.allclose(a, b, rtol=tol, atol=tol)
    if isinstance(b, numpy.ndarray):
        return equal(b, a, tol)
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(equal(a[key], b[key], tol) for key in a.keys())
    if tol is None:
        return a == b
    return abs(a - b) < tol * abs(b) + tol


class Parameters(dict):
    """Dictionary for parameters.

    Special feature: If param is a Parameters instance, then param.xc
    is a shorthand for param['xc'].
    """

    def __getattr__(self, key):
        if key not in self:
            return dict.__getattribute__(self, key)
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    @classmethod
    def read(cls, filename):
        """Read parameters from file."""
        file = open(os.path.expanduser(filename))
        parameters = cls(eval(file.read()))
        file.close()
        return parameters

    def tostring(self):
        keys = sorted(self)
        return 'dict(' + ',\n     '.join(
            '%s=%r' % (key, self[key]) for key in keys) + ')\n'

    def write(self, filename):
        file = open(filename, 'w')
        file.write(self.tostring())
        file.close()


class PropertyNotImplementedError(NotImplementedError):
    pass


class Calculator(object):
    """Base-class for all ASE calculators.

    A calculator must raise PropertyNotImplementedError if asked for a
    property that it can't calculate.  So, if calculation of the
    stress tensor has not been implemented, get_stress(atoms) should
    raise PropertyNotImplementedError.  This can be achieved simply by not
    including the string 'stress' in the list implemented_properties
    which is a class member.  These are the names of the standard
    properties: 'energy', 'forces', 'stress', 'dipole', 'charges',
    'magmom' and 'magmoms'.
    """

    implemented_properties = []
    'Properties calculator can handle (energy, forces, ...)'

    default_parameters = {}
    'Default parameters'

    all_changes = ['positions', 'numbers', 'cell', 'pbc',
                   'initial_charges', 'initial_magmoms']

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None, **kwargs):
        """Basic calculator implementation.

        restart: str
            Prefix for restart file.  May contain a directory.  Default
            is None: don't restart.
        ignore_bad_restart_file: bool
            Ignore broken or missing restart file.  By default, it is an
            error if the restart file is missing or broken.
        label: str
            Name used for all files.  May contain a directory.
        atoms: Atoms object
            Optional Atoms object to which the calculator will be
            attached.  When restarting, atoms will get its positions and
            unit-cell updated from file.
        """
        self.atoms = None  # copy of atoms object from last calculation
        self.results = {}  # calculated properties (energy, forces, ...)
        self.parameters = None  # calculational parameters

        if restart is not None:
            try:
                self.read(restart)  # read parameters, atoms and results
            except ReadError:
                if ignore_bad_restart_file:
                    self.reset()
                else:
                    raise

        self.label = None
        self.directory = None
        self.prefix = None

        self.set_label(label)

        if self.parameters is None:
            # Use default parameters if they were not read from file:
            self.parameters = self.get_default_parameters()

        if atoms is not None:
            atoms.calc = self
            if self.atoms is not None:
                # Atoms were read from file.  Update atoms:
                if not (equal(atoms.numbers, self.atoms.numbers) and
                            (atoms.pbc == self.atoms.pbc).all()):
                    raise RuntimeError('Atoms not compatible with file')
                atoms.positions = self.atoms.positions
                atoms.cell = self.atoms.cell

        self.set(**kwargs)

        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()

    def set_label(self, label):
        """Set label and convert label to directory and prefix.

        Examples:

        * label='abc': (directory='.', prefix='abc')
        * label='dir1/abc': (directory='dir1', prefix='abc')

        Calculators that must write results to files with fixed names
        can overwrite this method so that the directory is set to all
        of label."""

        self.label = label

        if label is None:
            self.directory = None
            self.prefix = None
        else:
            self.directory, self.prefix = os.path.split(label)
            if self.directory == '':
                self.directory = os.curdir

    def get_default_parameters(self):
        return Parameters(copy.deepcopy(self.default_parameters))

    def todict(self, skip_default=True):
        defaults = self.get_default_parameters()
        dct = {}
        for key, value in self.parameters.items():
            if hasattr(value, 'todict'):
                value = value.todict()
            if skip_default:
                default = defaults.get(key, '_no_default_')
                if default != '_no_default_' and equal(value, default):
                    continue
            dct[key] = value
        return dct

    def reset(self):
        """Clear all information from old calculation."""

        self.atoms = None
        self.results = {}

    def read(self, label):
        """Read atoms, parameters and calculated properties from output file.

        Read result from self.label file.  Raise ReadError if the file
        is not there.  If the file is corrupted or contains an error
        message from the calculation, a ReadError should also be
        raised.  In case of succes, these attributes must set:

        atoms: Atoms object
            The state of the atoms from last calculation.
        parameters: Parameters object
            The parameter dictionary.
        results: dict
            Calculated properties like energy and forces.

        The FileIOCalculator.read() method will typically read atoms
        and parameters and get the results dict by calling the
        read_results() method."""

        self.set_label(label)

    def get_atoms(self):
        if self.atoms is None:
            raise ValueError('Calculator has no atoms')
        atoms = self.atoms.copy()
        atoms.calc = self
        return atoms

    @classmethod
    def read_atoms(cls, restart, **kwargs):
        return cls(restart=restart, label=restart, **kwargs).get_atoms()

    def set(self, **kwargs):
        """Set parameters like set(key1=value1, key2=value2, ...).

        A dictionary containing the parameters that have been changed
        is returned.

        Subclasses must implement a set() method that will look at the
        chaneged parameters and decide if a call to reset() is needed.
        If the changed parameters are harmless, like a change in
        verbosity, then there is no need to call reset().

        The special keyword 'parameters' can be used to read
        parameters from a file."""

        if 'parameters' in kwargs:
            filename = kwargs.pop('parameters')
            parameters = Parameters.read(filename)
            parameters.update(kwargs)
            kwargs = parameters

        changed_parameters = {}

        for key, value in kwargs.items():
            oldvalue = self.parameters.get(key)
            if key not in self.parameters or not equal(value, oldvalue):
                changed_parameters[key] = value
                self.parameters[key] = value

        return changed_parameters

    def check_state(self, atoms, tol=1e-15):
        """Check for system changes since last calculation."""
        all_changes = ['positions', 'numbers', 'cell', 'pbc',
                       'initial_charges', 'initial_magmoms']
        if self.atoms is None:
            system_changes = all_changes[:]
        else:
            system_changes = []
            if not equal(self.atoms.positions, atoms.positions, tol):
                system_changes.append('positions')
            if not equal(self.atoms.numbers, atoms.numbers):
                system_changes.append('numbers')
            if not equal(self.atoms.cell, atoms.cell, tol):
                system_changes.append('cell')
            if not equal(self.atoms.pbc, atoms.pbc):
                system_changes.append('pbc')
            if not equal(self.atoms.get_initial_magnetic_moments(),
                         atoms.get_initial_magnetic_moments(), tol):
                system_changes.append('initial_magmoms')
            if not equal(self.atoms.get_initial_charges(),
                         atoms.get_initial_charges(), tol):
                system_changes.append('initial_charges')

        return system_changes

    def get_potential_energy(self, atoms=None, force_consistent=False):
        energy = self.get_property('energy', atoms)
        if force_consistent:
            if 'free_energy' not in self.results:
                name = self.__class__.__name__
                raise PropertyNotImplementedError(
                    'Force consistent/free energy not provided by {0} '
                    'calculator'.format(name))
            return self.results['free_energy']
        else:
            return energy

    def get_forces(self, atoms=None):
        return self.get_property('forces', atoms)

    def get_stress(self, atoms=None):
        return self.get_property('stress', atoms)

    def get_dipole_moment(self, atoms=None):
        return self.get_property('dipole', atoms)

    def get_charges(self, atoms=None):
        return self.get_property('charges', atoms)

    def get_magnetic_moment(self, atoms=None):
        return self.get_property('magmom', atoms)

    def get_magnetic_moments(self, atoms=None):
        """Calculate magnetic moments projected onto atoms."""
        return self.get_property('magmoms', atoms)

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError

        if atoms is None:
            atoms = self.atoms
            system_changes = []
        else:
            system_changes = self.check_state(atoms)
            if system_changes:
                self.reset()
        if name not in self.results:
            if not allow_calculation:
                return None
            self.calculate(atoms, [name], system_changes)

        if name == 'magmom' and 'magmom' not in self.results:
            return 0.0

        if name == 'magmoms' and 'magmoms' not in self.results:
            return numpy.zeros(len(atoms))

        if name not in self.results:
            # For some reason the calculator was not able to do what we want,
            # and that is OK.
            raise PropertyNotImplementedError

        result = self.results[name]
        if isinstance(result, numpy.ndarray):
            result = result.copy()
        return result

    def calculation_required(self, atoms, properties):
        system_changes = self.check_state(atoms)
        if system_changes:
            return True
        for name in properties:
            if name not in self.results:
                return True
        return False

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        """Do the calculation.

        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces', 'stress', 'dipole', 'charges', 'magmom'
            and 'magmoms'.
        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these six: 'positions', 'numbers', 'cell',
            'pbc', 'initial_charges' and 'initial_magmoms'.

        Subclasses need to implement this, but can ignore properties
        and system_changes if they want.  Calculated properties should
        be inserted into results dictionary like shown in this dummy
        example::

            self.results = {'energy': 0.0,
                            'forces': numpy.zeros((len(atoms), 3)),
                            'stress': numpy.zeros(6),
                            'dipole': numpy.zeros(3),
                            'charges': numpy.zeros(len(atoms)),
                            'magmom': 0.0,
                            'magmoms': numpy.zeros(len(atoms))}

        The subclass implementation should first call this
        implementation to set the atoms attribute.
        """

        if atoms is not None:
            self.atoms = atoms.copy()

    def calculate_numerical_forces(self, atoms, d=0.001):
        """Calculate numerical forces using finite difference.

        All atoms will be displaced by +d and -d in all directions."""

        from ase.calculators.test import numeric_force
        return numpy.array([[numeric_force(atoms, a, i, d)
                             for i in range(3)] for a in range(len(atoms))])

    def calculate_numerical_stress(self, atoms, d=1e-6, voigt=True):
        """Calculate numerical stress using finite difference."""

        stress = numpy.zeros((3, 3), dtype=float)

        cell = atoms.cell.copy()
        V = atoms.get_volume()
        for i in range(3):
            x = numpy.eye(3)
            x[i, i] += d
            atoms.set_cell(numpy.dot(cell, x), scale_atoms=True)
            eplus = atoms.get_potential_energy(force_consistent=True)

            x[i, i] -= 2 * d
            atoms.set_cell(numpy.dot(cell, x), scale_atoms=True)
            eminus = atoms.get_potential_energy(force_consistent=True)

            stress[i, i] = (eplus - eminus) / (2 * d * V)
            x[i, i] += d

            j = i - 2
            x[i, j] = d
            x[j, i] = d
            atoms.set_cell(numpy.dot(cell, x), scale_atoms=True)
            eplus = atoms.get_potential_energy(force_consistent=True)

            x[i, j] = -d
            x[j, i] = -d
            atoms.set_cell(numpy.dot(cell, x), scale_atoms=True)
            eminus = atoms.get_potential_energy(force_consistent=True)

            stress[i, j] = (eplus - eminus) / (4 * d * V)
            stress[j, i] = stress[i, j]
        atoms.set_cell(cell, scale_atoms=True)

        if voigt:
            return stress.flat[[0, 4, 8, 5, 2, 1]]
        else:
            return stress

    def get_spin_polarized(self):
        return False

    def band_structure(self):
        """Create band-structure object for plotting."""
        from ase.dft.band_structure import get_band_structure
        # XXX This calculator is supposed to just have done a band structure
        # calculation, but the calculator may not have the correct Fermi level
        # if it updated the Fermi level after changing k-points.
        # This will be a problem with some calculators (currently GPAW), and
        # the user would have to override this by providing the Fermi level
        # from the selfconsistent calculation.
        return get_band_structure(calc=self)


class LJp(Calculator):
    all_changes = ['positions', 'numbers', 'cell', 'pbc',
                   'initial_charges', 'initial_magmoms']

    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters = {'epsilon': 1.0,
                          'sigma': 1.0,
                          'rc': None,
                          'pressure': 0.}
    nolabel = True

    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        # Calculator.__init__(self, **kwargs)
        # pass

    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        import numpy as np
        super(self.__class__, self).calculate(atoms, properties, system_changes)

        natoms = len(self.atoms)

        sigma = self.parameters.sigma
        epsilon = self.parameters.epsilon
        rc = self.parameters.rc
        pressure = self.parameters.pressure
        if rc is None:
            rc = 3 * sigma

        if 'numbers' in system_changes:
            self.nl = NeighborList([rc / 2] * natoms, self_interaction=False)

        self.nl.update(self.atoms)

        positions = self.atoms.positions
        cell = self.atoms.cell

        e0 = 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)

        energy = 0.0
        forces = np.zeros((natoms, 3))
        stress = np.zeros((3, 3))

        for a1 in range(natoms):
            neighbors, offsets = self.nl.get_neighbors(a1)
            cells = np.dot(offsets, cell)
            d = positions[neighbors] + cells - positions[a1]
            r2 = (d ** 2).sum(1)
            c6 = (sigma ** 2 / r2) ** 3
            c6[r2 > rc ** 2] = 0.0
            energy -= e0 * (c6 != 0.0).sum()
            c12 = c6 ** 2
            energy += 4 * epsilon * (c12 - c6).sum()
            f = (24 * epsilon * (2 * c12 - c6) / r2)[:, np.newaxis] * d
            forces[a1] -= f.sum(axis=0)
            for a2, f2 in zip(neighbors, f):
                forces[a2] += f2
            stress += np.dot(f.T, d)

        if 'stress' in properties:
            if self.atoms.number_of_lattice_vectors == 3:
                stress += stress.T.copy()
                stress *= -0.5 / self.atoms.get_volume()

                stress += np.diag([pressure, ] * 3)
                self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]
                # print stress
            else:
                raise PropertyNotImplementedError

        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces