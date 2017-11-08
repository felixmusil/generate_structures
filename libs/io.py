import h5py
import numpy as np
from time import ctime
import os
import itertools
from quippy import Atoms as qpAtoms
from ase import Atoms as aseAtoms
from multiprocessing import Pool
from utils import is_notebook
if is_notebook():
    from tqdm import tqdm_notebook as tqdm_cs
else:
    from tqdm import tqdm as tqdm_cs

class Frame_Dataset_h5(object):
    def __init__(self,fname,mode='a',swmr_mode=True ,bname="frame",debug=False,disable_pbar=False):
        # super(Frame_Dataset_h5, self).__init__()
        if mode == 'r':
            self.fname = fname
            self.f = h5py.File(self.fname, 'r', libver='latest')
            self.swmr_mode = self.f.swmr_mode
            self.f.close()
        else:
            self.fname = check_suffix(fname)
            self.f = h5py.File(self.fname, 'a', libver='latest')
            self.f.swmr_mode = swmr_mode
            self.swmr_mode = swmr_mode

            self.f.close()

        self.isOpen = False
        self.debug = debug

        self.counter = 0
        self.frame_fields = ["cell" ,"positions" ,"numbers" ,"pbc"]
        self.disable_pbar = disable_pbar
        self.bname = bname
        self.names = self.get_names()

    def open(self,mode):
        self.f = h5py.File(self.fname, mode ,libver='latest')
        if self.debug:
            print 'Opening {}'.format(self.fname)
        self.isOpen = True
    def close(self):
        self.f.close()
        if self.debug:
            print 'Closing {}'.format(self.fname)
        self.isOpen = False

    def get_names(self):
        with h5py.File(self.fname, 'r', libver='latest') as f:
            names = f.keys()
        return names

    def dump_frame(self  ,crystal ,inp_dict=None,f = None):
        to_close = False
        if f is None and self.isOpen is True:
            f = self.f

        elif f is None and self.isOpen is False:
            self.open(mode='a')
            f = self.f
            to_close = True

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

        if to_close:
            self.close()

    def dump_frames(self ,crystals ,inputs=None):
        if inputs is None:
            inputs = [None for crystal in crystals]

        with h5py.File(self.fname, 'a' ,libver='latest') as f:
            for crystal ,inp_dict in zip(crystals ,inputs):
                try:
                    self.dump_frame(crystal ,inp_dict,f)
                except:
                    print 'frame {} with input was not saved'.format(crystal,inp_dict)
                    pass

    def load_frame(self  ,name ,frame_type='quippy',f = None):
        to_close = False
        if f is None and self.isOpen is True:
            f = self.f
        elif f is None and self.isOpen is False:
            self.open(mode='r')
            f = self.f
            to_close = True

        data = {}
        for field in self.frame_fields:
            data[field] = f[name][field].value

        if frame_type == 'quippy':
            frame = qpAtoms(**data)
        elif frame_type == 'ase':
            frame = aseAtoms(**data)

        if to_close:
            self.close()

        return frame

    def load_frames(self  ,names=None ,frame_type='quippy',nprocess=1):
        if names is None:
            names = self.get_names()
        frames = {}
        with h5py.File(self.fname, "r" ,libver='latest', swmr=self.swmr_mode) as f:
            for name in tqdm_cs(names,desc='Load frames',disable=self.disable_pbar):
                frames[name] = self.load_frame(name,frame_type=frame_type,f=f)

        return frames


descriptor_parameters = {
    'soap':['nmax','lmax','cutoff','gaussian_width',
            'centerweight','cutoff_transition_width','nocenters','exclude','chem_channels']
}

class DescriptorWriter(object):
    def __init__(self, fname, mode='a', swmr_mode=True, bname="soap",debug=False,disable_pbar=False):
        super(self.__class__, self).__init__()
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
        self.disable_pbar = disable_pbar
        self.isOpen = False
        self.debug = debug

    def open(self,mode='a'):
        self.f = h5py.File(self.fname, mode ,libver='latest')
        if self.debug:
            print 'Opening {}'.format(self.fname)
        self.isOpen = True
    def close(self):
        self.f.close()
        if self.debug:
            print 'Closing {}'.format(self.fname)
        self.isOpen = False

    def get_names(self):
        with h5py.File(self.fname, 'r', libver='latest') as f:
            names = f.keys()
        return names

    def dump(self,desc,frame_name,input_param,f=None):
        to_close = False
        if f is None and self.isOpen is True:
            f = self.f
        elif f is None and self.isOpen is False:
            self.open(mode=self.mode)
            f = self.f
            to_close = True
        if self.mode == 'a':
            grp = f[frame_name]
        elif self.mode == 'w':
            grp = f.create_group(frame_name)

        iis = [-1]
        for name in grp.keys():
            if self.bname in name:
                ii = int(name.split('_')[-1])
                iis.append(ii)
        desc_name = self.bname + '_{}'.format(int(np.max(iis)+1))

        desc_grp = grp.create_group(desc_name)
        desc_grp.attrs['created'] = ctime()
        for name,val in input_param.iteritems():
            desc_grp.attrs[name] = val

        if self.bname == 'soap':
            if input_param['chem_channels'] is False:
                for name,val in desc.iteritems():
                    desc_grp.create_dataset(name,data=val)
            elif input_param['chem_channels'] is True:
                for name,center in desc.iteritems():
                    subgrp = desc_grp.create_group(name)
                    for ab,pA in center.iteritems():
                        subgrp.create_dataset(ab,data=pA)
        else:
            raise NotImplementedError('Descriptor {} is not implemented'.format(self.bname))

        if to_close:
            self.close()

    def dumps(self,descs,frame_names,input_params):
        if isinstance(input_params,dict):
            input_params_ = [input_params for it in range(len(descs))]
        else:
            input_params_ = input_params
        with h5py.File(self.fname,mode=self.mode,libver='latest') as f:
            for desc,frame_name,input_param in tqdm_cs(zip(descs,frame_names,input_params_),
                                                             desc='Dump desc',disable=self.disable_pbar):
                self.dump(desc,frame_name,input_param,f)

class DescriptorReader(object):
    def __init__(self, fname, bname="soap",debug=False,disable_pbar=False):
        super(self.__class__, self).__init__()
        self.mode = 'r'
        self.fname = fname
        with h5py.File(self.fname, self.mode, libver='latest') as f:
            self.swmr_mode = f.swmr_mode
            self.frame_names = f.keys()
        self.debug = debug
        self.bname = bname
        self.isOpen = False
        self.disable_pbar = disable_pbar

    def open(self,mode='r'):
        self.f = h5py.File(self.fname, mode ,libver='latest')
        if self.debug:
            print 'Opening {}'.format(self.fname)
        self.isOpen = True
    def close(self):
        self.f.close()
        if self.debug:
            print 'Closing {}'.format(self.fname)
        self.isOpen = False

    def load(self,frame_name,input_param=None,f=None,get_dataset=False,check_param=True,desc_name=None):
        to_close = False
        if f is None and self.isOpen is True:
            f = self.f
        elif f is None and self.isOpen is False:
            self.open(mode=self.mode)
            f = self.f
            to_close = True

        grp = f[frame_name]

        # there are no descriptor of the bname kind in this frame

        aaa = []
        for k in grp.keys():
            aaa.append(self.bname in k)
        if not np.any(aaa):
            return None

        if check_param is True:
            desc_name = None
            for name in grp:
                if self.bname in name:
                    aaa = []
                    for inp_name,val in input_param.iteritems():
                        if inp_name in descriptor_parameters[self.bname]:
                            try:
                                if isinstance(val,list) or isinstance(val,np.ndarray):
                                    if np.allclose(val,grp[name].attrs[inp_name]):
                                        aaa.append(True)
                                    else:
                                        aaa.append(False)
                                elif isinstance(val,int) or isinstance(val,float) or isinstance(val,str):
                                    if val == grp[name].attrs[inp_name]:
                                        aaa.append(True)
                                    else:
                                        aaa.append(False)
                                else:
                                    raise ValueError(''.format(val))
                            except KeyError:
                                pass

                    if np.all(aaa):
                        desc_name = name
                        break

        try:
            desc_h5 = grp[desc_name]
        except TypeError:
            raise ValueError('Input parameters {} do not match descriptors in {}'.format(input_param,frame_name))

        desc = {}

        if self.bname == 'soap':
            if input_param['chem_channels'] is False:
                for name,p in desc_h5.iteritems():
                    if not get_dataset:
                        desc[name] = p.value
                    else:
                        desc[name] = p
            elif input_param['chem_channels'] is True:
                for name,center in desc_h5.iteritems():
                    desc[name] = {}
                    for ab,pA in center.iteritems():
                        if not get_dataset:
                            desc[name][ab] = pA.value
                        else:
                            desc[name][ab] = pA
        else:
            raise NotImplementedError('Descriptor {} is not implemented'.format(self.bname))

        if to_close:
            self.close()

        return desc

    def loads(self, input_params,frame_names=None,get_dataset=False):
        if frame_names is None:
            frame_names = self.frame_names
        if isinstance(input_params,dict):
            input_params_ = [input_params for it in range(len(frame_names))]
        else:
            input_params_ = input_params



        descs = {}
        with h5py.File(self.fname, mode=self.mode, libver='latest',swmr=self.swmr_mode) as f:
            for frame_name,input_param in tqdm_cs(zip(frame_names,input_params_),
                                                             desc='Load desc',disable=self.disable_pbar):
                descs[frame_name] = self.load(frame_name,input_param,f,get_dataset)


        return descs

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