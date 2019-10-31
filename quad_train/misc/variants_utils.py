"""
Different utils helping with experiment variants generation
"""
import argparse
import sys
import os
import datetime, time
import itertools
import os.path as osp
import uuid
import copy

import numpy as np

import dateutil.tz
import yaml

def set_dictval_by_name(dic, fullkey, val):
    """
    Given a name like: param.subparam.subsubparam
    sets dict[param][subparam][subsubparam] = val
    """
    keys = fullkey.split(".", 1)
    if len(keys) == 1:
        dic[keys[0]] = val
    else:
        if keys[0] not in dic:
            dic[keys[0]] = {}
        set_dictval_by_name(dic[keys[0]], keys[1], val)

def str2num(s):
    try:
        return int(s)
    except:
        try: 
            return float(s)
        except:
            return s

def seed_str(seed):
    return "%03d" % seed

def grid_of_variants(args, param_def=None, remove_extra_folders=False):
    """
    Grid parameter generation into a list of variants.
    @param: args: argument provided into a command line
    @param: param: default parameters provided (could be None)
    @param: remove_extra_folders: if a single experiment is - just dump in the root folder 
        i.e. remove seed subfolder and subfolders named with param values
    """
    ## Some auxiliary stuff if needed
    # rand_id = str(uuid.uuid4())[:5]
    # now = datetime.datetime.now(dateutil.tz.tzlocal())
    # timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')
    # exp_name = 'experiment_%s_%s' % (timestamp, rand_id)

    args.log_dir = args.log_dir.rstrip(os.sep) + os.sep

    if args.seed is None:
        seeds = [param_def["seed"]]
    else:
        seeds = [int(x) for x in args.seed.split(',')]

    if args.param_name is None:
        # Just running with  default settings
        param_name = None
        param_values = []
    else:
        param_name = [x for x in args.param_name.split(',')]
        if args.param_val is None:
            raise ValueError('No values provided for param %s' % param_name)
        else:
            param_values = [[str2num(y) for y in x.split(',')] for x in args.param_val.split(',,')]

    ## GRID of parameters to run: 
    param_values.append(seeds)
    single_seed = len(seeds) == 1
    param_cart_product = itertools.product(*param_values)

    variants_list = []

    for param_tuple in param_cart_product:
        variant = copy.deepcopy(param_def)
        params = copy.deepcopy(param_def["variant"])
        seed = param_tuple[-1]
        param_tuple = param_tuple[:-1]
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('PARAMETERS TUPLE: ', param_name, param_tuple, ' SEED: ', seed)

        if param_name is not None:
            log_dir = args.log_dir
            # In case we have a single parameter then we create subdirectories with values of this parameter
            if len(param_name) == 1:
                set_dictval_by_name(params, fullkey=param_name[0], val=param_tuple[0])
                log_dir += (param_name[0] + '/' + str(param_tuple[0]) + '/')
            else:
                for par_i, par in enumerate(param_name):
                    set_dictval_by_name(params, fullkey=par, val=param_tuple[par_i])
                    if par_i == 0:
                        log_dir += (par + '_' + str(param_tuple[par_i]))
                    else:
                        log_dir += ('__' + par + '_' + str(param_tuple[par_i]))
                log_dir += '/'
            ## In case I want to eliminate unnecessary folders
            if not single_seed or not remove_extra_folders:
                log_dir += ("seed_" + seed_str(seed) + '/')
        else:
            log_dir = args.log_dir + "seed_" + seed_str(seed) + '/'
        log_dir_errors = log_dir + 'errors/'
        if not os.path.isdir(log_dir_errors):
            os.makedirs(log_dir_errors)

        variant['seed'] = seed
        variant['log_dir'] = log_dir
        variant["variant"] = params #Confusing, but it is just a task variant (i.e. task parameters)

        # These are just copied from the args
        variant["n_parallel"] = args.n_parallel 
        variant["snapshot_mode"] = args.snapshot_mode 
        variant["plot"] = args.plot 

        variants_list.append(variant)

    if len(variants_list) == 1 and remove_extra_folders:
        variants_list[0]["log_dir"] = args.log_dir
    
    return variants_list