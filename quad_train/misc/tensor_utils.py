#!/usr/bin/env python
import numpy as np
import inspect

import h5py
# import os,sys

import garage_metadist.misc.dict2hdf5 as h5u

def stack_tensor_list(tensor_list):
    if tensor_list is None or len(tensor_list) == 0:
        # print('WARINING: Sampler: Empty tensor list provided')
        return tensor_list
    # In case we have tuple observations - we will have a list of tuples
    # Thus we have to re-pack it to tuple of lists and then stack them to np array
    if isinstance(tensor_list[0], tuple):
        lists = list(zip(*tensor_list))
        arrays = []
        for lst in lists:
            arrays.append(np.stack(lst))
            # print('!stack_tensor_list: array shape ', arrays[-1].shape)
        return tuple(arrays)
    else:
        stacked_array = np.array(tensor_list)
        # print('!!stack_tensor_list: array shape ', np.array(tensor_list).shape, ' Tensor shape = ', tensor_list[0].shape)
        return np.stack(tensor_list)

def truncate_tensor_list(tensor_list, truncated_len):
    # In case we have tuple observations - we will have a list of tuples
    if isinstance(tensor_list[0], tuple):
        tensor_list = list(tensor_list)
        for lst_i in range(len(tensor_list)):
            tensor_list[lst_i] = tensor_list[lst_i][:truncated_len]
        return tuple(tensor_list)
    else:
        return tensor_list[:truncated_len]


def concat_tensor_list(tensor_list):
    if isinstance(tensor_list[0], tuple):
        obs_lists = list(zip(*tensor_list))
        # print('concat_tensor_list: obs = ', [len(obs) for obs in obs_lists])
        return tuple([np.concatenate(obs, axis=0) for obs in obs_lists])
    else:
        return np.concatenate(tensor_list, axis=0)


def dict_make_list_leafs(dic_in, orig_dic):
    for key in orig_dic:
        if not isinstance(orig_dic[key], dict):
            dic_in[key] = []
        else:
            dic_in[key] = {}
            dict_make_list_leafs(dic_in[key], orig_dic[key])

def append2leafs(dic_append, orig_dic):
    for key in orig_dic:
        if not isinstance(orig_dic[key], dict):
            dic_append[key].append(orig_dic[key])
        else:
            append2leafs(dic_append[key], orig_dic[key])         

def stack_leafs(dic_stack):
    for key in dic_stack:
        if not isinstance(dic_stack[key], dict):
            dic_stack[key] = stack_tensor_list(dic_stack[key])
        else:
            stack_leafs(dic_stack[key])   

def dic_print_shapes(dic, indent=""):
    for key in dic:
        if not isinstance(dic[key], dict):
            if isinstance(dic[key], np.ndarray):
                print(indent, key, ":", dic[key].shape)
            else:
                print(indent, key, ":", type(dic[key]))
        else:
            dic_print_shapes(dic[key], indent + "  ") 

## Let's assign nested list to each dict el
def repack2arrays(list_of_dic, path="/"):
    """
    Assuming that you have a dictionary of lists with the structure 
    (that is used for keeping trajectories)
        [runs_num][iter_num][traj_per_iter]{dict_of_data}
    Repack it into:
        {dict_of_data} x np.array([runs_num x  iter_num x traj_per_iter x data_dim])
    Assumes uniform sizes and structure (including dictionary names) through all lists
    """

    if not isinstance(list_of_dic[0], dict):
        new_list_of_dic = []
        for eli, el in enumerate(list_of_dic):
            new_list_of_dic.append(repack2arrays(el, path=path + str(eli) + "/"))
        list_of_dic = new_list_of_dic
    
    # Replicating dict structure to a dict of lists
    dic_of_lists = {}
    dict_make_list_leafs(dic_of_lists, list_of_dic[0])

    # Crawling each of dicts and copying data to the dict of lists
    for el in list_of_dic:
        append2leafs(dic_of_lists, el)

    # Crawling dictionary to stack arrays
    stack_leafs(dic_of_lists)

    return dic_of_lists

def test():
    data = h5u.dict2h5.load("_results_temp/ppo_precred/joined.h5", pack2list=True)
    dic_of_np = repack2arrays(data["traj_data"])
    dic_print_shapes(dic_of_np)

if __name__ == "__main__":
    test()