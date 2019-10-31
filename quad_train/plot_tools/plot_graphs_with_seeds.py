#!/usr/bin/env python
import numpy as np
import os, sys
import matplotlib.pyplot as plt

import argparse
from argparse import RawTextHelpFormatter, ArgumentDefaultsHelpFormatter

import time
import datetime
from dateutil.relativedelta import relativedelta

from quad_dynalearn.plot_tools import plot_tools as pt

def runtime_str(t_diff):
    days = int(t_diff / (3600 * 24))
    hours = int((t_diff % (3600 * 24)) / 3600)
    minutes = int( (t_diff % 3600) / 60 )
    seconds = int(t_diff % 60)
    return '{d}d {h}h {m}m {s}s'.format(d=days, h=hours, m=minutes, s=seconds)

def main(argv):
    #Parsing command line arguments
    parser = argparse.ArgumentParser(
        description="Argument description: ",
        formatter_class=ArgumentDefaultsHelpFormatter)
        # formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "dir",
        help="Input directory"
    )
    parser.add_argument(
        "-g","--graph",
        default="rewards/rew_main_avg",
        help="Name of data column in a csv file"
    )
    parser.add_argument(
        "-s","--seeds",
        action="store_true",
        help="Show individual seeds"
    )
    parser.add_argument(
        "-sns","--seaborn",
        action="store_true",
        help="Use seaborn for visualiztion"
    )
    parser.add_argument(
        "-top","--top_n",
        type=int,
        help="Show top highest N results only. If N < 0 then takes top lowest N results"
    )
    parser.add_argument(
        "-ti","--top_n_interval",
        type=int,
        default=5,
        help="Averaging interval for the top performing agents"
    )
    args = parser.parse_args()

    ###################################
    ### Main code
    print('Arguments: ')
    [print(arg, ': ',val) for arg,val in args.__dict__.items()]

    ## Printing runtime
    time_start = time.time()

    if args.seaborn:
        try:
            import seaborn as sns
            sns.set()
        except:
            print("WARN: No seaborn found. Continuing with classic theme ...")

    # Reading subdirectories
    folders = pt.subdir(args.dir)

    # Forming labels
    labels = [sd.rsplit(os.sep,1)[1] for sd in folders]

    # Reading and ploting
    pt.read_and_plot_seeds(folders=folders, 
                        graph_names=[args.graph]*len(folders), 
                        labels=labels, 
                        ylabel_str=args.graph, 
                        show_seeds=args.seeds,
                        top_n=args.top_n,
                        top_n_interval=args.top_n_interval)
    
    time_end = time.time()
    print("RUNTIME: ", runtime_str(time_end-time_start))

if __name__ == '__main__':
    main(sys.argv)