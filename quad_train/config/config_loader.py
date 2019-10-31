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

def trpo_ppo_default_params():
    path2conf = os.path.realpath(__file__).rsplit(os.sep, 1)[0]
    yaml_stream = open(path2conf + os.sep + "trpo_ppo_default.yml", 'r')
    params = yaml.load(yaml_stream)
    return params