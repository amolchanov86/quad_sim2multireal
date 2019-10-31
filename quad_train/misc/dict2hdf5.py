#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import h5py
import os
import numbers
import inspect



def h5_is_simple_type(obj):
    return (
        isinstance(obj, numbers.Number) or
        type(obj) == str or 
        isinstance(obj, np.bool_)
    )

class dict2h5(object):
    """
    The class can save and load python dictionary into a hdf5 file
    """
    indx_format = "%04d"
    @classmethod
    def indx2str(cls, indx):
        return cls.indx_format % indx

    @classmethod
    def strkey_val(cls, obj):
        if type(obj) == list or type(obj) == tuple:
            for i,v in enumerate(obj):
                yield cls.indx2str(i), v
        elif type(obj) == dict:
            for i,v in obj.items():
                yield str(i), v
        else:
            raise NotImplementedError("ERROR: %s: Unsupported data type: %s" % (inspect.stack()[0][3], type(obj)))

    ## Saving a dictionary
    @classmethod
    def save(cls, dic, filename=None, h5file=None, float_nptype=None):
        ## The following check is not necessary
        # if os.path.exists(filename):
        #     raise ValueError('File %s exists, will not overwrite.' % filename)
        if filename is None:
            cls.__recursively_save_dict_contents_to_group__(h5file, '/', dic, float_nptype=float_nptype)
        else:
            with h5py.File(filename, 'a') as h5file:
                cls.__recursively_save_dict_contents_to_group__(h5file, '/', dic, float_nptype=float_nptype)



    @classmethod
    def __recursively_save_dict_contents_to_group__(cls, h5file, path, dic, float_nptype=None):
        """
        @param: float_nptype: allows reduction in float precision. 
        If size of float_nptype < original size then data is converted
        """
        # argument type checking
        # if not isinstance(dic, dict):
        #     raise ValueError("must provide a dictionary")
        if not isinstance(path, str):
            raise ValueError("path must be a string")
        if not isinstance(h5file, h5py._hl.files.File):
            raise ValueError("must be an open h5py file")
        # save items to the hdf5 file
        # for key, item in dic.items():
        for key, item in cls.strkey_val(dic):
            # save strings, numpy.intX, and numpy.floatX types
            if h5_is_simple_type(item):
                h5file[path + key] = item
                if not h5file[path + key].value == item:
                    raise ValueError('The data representation in the HDF5 file does not match the original dict.')
            # save numpy arrays
            elif isinstance(item, np.ndarray):
                if float_nptype is not None and \
                    item.dtype in [np.float, np.float16, np.float32, np.float64] and \
                    item.dtype.itemsize > np.zeros(1,dtype=float_nptype).itemsize:
                    item = item.astype(float_nptype)
                h5file[path + key] = item
                if not np.array_equal(h5file[path + key].value, item):
                    raise ValueError('The data representation in the HDF5 file does not match the original dict.')
            # save dictionaries
            elif type(item) in [dict, list, tuple]:
                cls.__recursively_save_dict_contents_to_group__(h5file, path + key + '/', item, float_nptype=float_nptype)
            # attempt to convert to string
            else:
                print("WARNING: %s: Unknown HDF5 data type %s. Attempting to convert to a string" % (inspect.stack()[0][3], type(item)))
                h5file[path + key] = str(item)
                if not h5file[path + key].value == str(item):
                    raise ValueError('The data representation in the HDF5 file does not match the original dict.')
            # other types cannot be saved and will result in an error
            # else:
            #     raise ValueError('Cannot save %s type.' % type(item))

    ## Load a hdf5 file
    @classmethod
    def load(cls, filename, pack2list=False):
        """
        Loads HDF5 content into a dictionary
        """
        with h5py.File(filename, 'r') as h5file:
            return cls.__recursively_load_dict_contents_from_group__(h5file, '/', pack2list=pack2list)

    @classmethod
    def __recursively_load_dict_contents_from_group__(cls, h5file, path, pack2list=False):
        """
        A helper function to for recursive loading into a dictionary
        If pack2list == True then if the keys in the group are indices (i.e. int > 0 and sequential starting at 0)
        Then it will pack them into a list instead of a dict()
        """
        ## Check if group val is a simple type
        keys = list(h5file[path].keys())
        # isdigit() checks if it is a positive integer
        keys_are_digits = [key.isdigit() for key in keys]

        ## Checking if subgroups names are just indices
        ## In that case we should create a list of values
        pack2list_here = pack2list and len(keys) > 0 and np.all(keys_are_digits)
        if pack2list_here:
            keys_val = np.array([int(key) for key in keys])
            keys_indx_sorted = np.argsort(keys_val)
            keys_val = keys_val[keys_indx_sorted]
            # keys_val = np.sort([int(key) for key in keys]) #This is correct. Excluded to avoid double sorting
            key_indices = np.arange(keys_val[0], keys_val[-1]+1)
            pack2list_here = pack2list_here and np.all(keys_val == key_indices) and len(keys_val) > 0

        if pack2list_here:
            ans = []
            # re-sorting, but now the strings
            # it is done to avoid situations when, say your indx is "0000" which is just 0
            # keys_val = np.sort(keys) #This may have problems, when you have "11", "2". It will be sorted incorrectly
            keys = np.array(keys)
            keys_sort = keys[keys_indx_sorted] #same sorting that is for keys numerical values
            for key in keys_sort:
                if isinstance(h5file[path + key], h5py._hl.group.Group):
                    ans.append(cls.__recursively_load_dict_contents_from_group__(h5file, path + key + '/', pack2list=pack2list))
                elif isinstance(h5file[path + key], h5py._hl.dataset.Dataset):
                    ans.append(h5file[path + key].value)
        else:
            ans = {}
            for key, item in h5file[path].items():
                if isinstance(item, h5py._hl.dataset.Dataset):
                    ans[key] = item.value
                elif isinstance(item, h5py._hl.group.Group):
                    ans[key] = cls.__recursively_load_dict_contents_from_group__(h5file, path + key + '/', pack2list=pack2list)
        return ans

    @classmethod
    def append_train_iter_data(cls, h5file, data, data_group="traj_data/", teacher_indx=0, itr=None, float_nptype=np.float32):
        """
        Appends data of training iteration [itr] to the existing list in hdf5
        If [itr] is None - tries to find max index and append
        If [itr] is provided writes data under the group with the name "itr".
        Assumed structure: 
            data_group/teacher_indx/itr = data 
        If you would like to avoid adding teacher_indx then set it to None.
        It will result in:
            data_group/itr = data
        """
        data_group = data_group.rstrip("/") + "/"
        if teacher_indx is None:
            teacher_group = data_group
        else:
            teacher_group = data_group + cls.indx2str(teacher_indx) + "/"
        if itr is None:
            iter_indices = [int(i) for i in iter(h5file[teacher_group])]
            if not iter_indices:
                itr = 0
            else:
                max_indx = np.max(iter_indices)
                itr = max_indx + 1
        cls.__recursively_save_dict_contents_to_group__(h5file, teacher_group + cls.indx2str(itr) + '/', data, float_nptype=float_nptype)
    
    @classmethod
    def add_dict(cls, h5file, dic, groupname="/", float_nptype=None):
        groupname = groupname.rstrip("/") + "/"
        cls.__recursively_save_dict_contents_to_group__(h5file, groupname, dic, float_nptype=float_nptype)
    
    @classmethod
    def add_cmd(h5file):
        # Saving command line to hdf
        h5file["cmd"] = " ".join(sys.argv)
    



## Test
if __name__ == "__main__":

    filename = 'foo.hdf5'
    if os.path.exists(filename):
        os.remove(filename)
    ex = {
        'name': 'stefan',
        'age':  np.int64(24),
        'fav_numbers': np.array([2,4,4.3]),
        'fav_tensors': {
            'levi_civita3d': np.array([
                [[0,0,0],[0,0,1],[0,-1,0]],
                [[0,0,-1],[0,0,0],[1,0,0]],
                [[0,1,0],[-1,0,0],[0,0,0]]
            ]),
            'kronecker2d': np.identity(3)
        },
        'lst': [1.1, 1.2, 1.3]
    }
    print(ex)
    dict2h5.save(ex, filename)
    loaded = dict2h5.load('foo.hdf5')
    print(loaded)
    np.testing.assert_equal(loaded, ex)
    print('check passed!')
    ex2 = {
        'name2': 'stefan',
        'age2':  np.int64(24),
        'fav_numbers2': np.array([2,4,4.3]),
        'fav_tensors2': {
            'levi_civita3d': np.array([
                [[0,0,0],[0,0,1],[0,-1,0]],
                [[0,0,-1],[0,0,0],[1,0,0]],
                [[0,1,0],[-1,0,0],[0,0,0]]
            ]),
            'kronecker2d': np.identity(3)
        },
        'lst2': [1.1, 1.2, 1.3]
    }
    dict2h5.save(ex2, filename)
