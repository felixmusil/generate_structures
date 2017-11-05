import h5py
import numpy as np
from time import ctime
import os

class Frame_Dataset_h5(object):
    def __init__(self,fname,mode='a',swmr_mode=True ,bname="frame"):
        super(Frame_Dataset_h5,self).__init__()
        if mode == 'r':
            fname = fname
            self.f = h5py.File(fname, 'r', libver='latest')
            self.swmr_mode = self.f.swmr_mode
            self.f.close()
        else:
            fname = check_suffix(fname)
            self.f = h5py.File(fname, 'a', libver='latest')
            self.f.swmr_mode = swmr_mode
            self.swmr_mode = swmr_mode

            self.f.close()

        self.fname = fname

        self.counter = 0
        self.frame_fields = ["cell" ,"positions" ,"numbers" ,"pbc"]

        self.bname = bname
        self.names = self.get_names()

    def get_names(self):
        with h5py.File(self.fname, 'r', libver='latest') as f:
            names = f.keys()
        return names

    def dump_frame(self ,f ,crystal ,inp_dict=None):
        name = self.bname + '_{}'.format(self.counter)
        self.names.append(name)
        grp = f.create_group(name)
        grp.attrs['created'] = ctime()
        grp.create_dataset("cell", data=crystal.get_cell())
        grp.create_dataset("positions" ,data=crystal.get_positions())
        grp.create_dataset("numbers" ,data=crystal.get_atomic_numbers())
        grp.create_dataset("pbc" ,data=crystal.get_pbc())

        if inp_dict is not None:
            sgrp = grp.create_group("inputs")
            for name ,val in inp_dict.iteritems():
                sgrp.create_dataset(name, data=np.array(val))
        self.counter += 1


    def dump_frames(self ,crystals ,inputs=None):
        if inputs is None:
            inputs = [None for crystal in crystals]

        with h5py.File(self.fname, 'a' ,libver='latest') as f:
            for crystal ,inp_dict in zip(crystals ,inputs):
                try:
                    self.dump_frame(f ,crystal ,inp_dict)
                except:
                    print 'frame {} with input was not saved'.format(crystal,inp_dict)
                    pass

    def load_frame(self  ,name ,frame_type='quippy'):
        data = {}
        with h5py.File(self.fname, "r" ,libver='latest', swmr=self.swmr_mode) as f:
            for field in self.frame_fields:
                data[field] = f[name][field].value
        if frame_type == 'quippy':
            from quippy import Atoms as qpAtoms
            frame = qpAtoms(**data)
        elif frame_type == 'ase':
            from ase import Atoms as aseAtoms
            frame = aseAtoms(**data)
        return frame

    def load_frames(self  ,names=None ,frame_type='quippy'):
        if names is None:
            names = self.get_names()

        data = {}
        with h5py.File(self.fname, "r" ,libver='latest', swmr=self.swmr_mode) as f:
            for name in names:
                data[name] = {}
                for field in self.frame_fields:
                    data[name][field] = f[name][field].value
        frames = {}
        if frame_type == 'quippy':
            from quippy import Atoms as qpAtoms
            for name in names:
                frames[name] = qpAtoms(**data[name])
        elif frame_type == 'ase':
            from ase import Atoms as aseAtoms
            for name in names:
                frames[name] = aseAtoms(**data[name])

        return frames

class DescriptorWriter(object):
    def __init__(self, fname, mode='a', swmr_mode=True, bname="soap"):
        super(DescriptorWriter, self).__init__()
        if mode == 'a':
            self.fname = fname
            with h5py.File(self.fname, 'r', libver='latest') as f:
                self.swmr_mode = f.swmr_mode
                self.frame_names = f.keys()

        elif mode == 'w':
            self.fname = check_suffix(fname)
            with h5py.File(self.fname, 'w', libver='latest') as f:
                f.swmr_mode = swmr_mode
                self.swmr_mode = swmr_mode
            self.counter = 0
        self.mode = mode
        self.bname = bname

    def dump_descriptor(self,desc,frame_name,f):
        pass


def check_suffix(fileName):
    fname = os.path.abspath(fileName)
    a = fname.split('.')
    end = a[-1]
    fname = ''.join(a[:-1])
    suffix = 0
    while os.path.isfile(fname + '-{}'.format(suffix) +'.'+ end ):
        suffix += 1
    new_file_path = fname +'-{}'.format(suffix) +'.'+ end
    return new_file_path