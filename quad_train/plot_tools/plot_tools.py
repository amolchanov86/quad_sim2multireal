#!/usr/bin/env python
import numpy as np
import os, sys
import pandas as pd
from scipy.signal import lfilter
import matplotlib.pyplot as plt



def subdir(folder):
    return [f.path for f in os.scandir(folder) if f.is_dir() ]

def get_opt_score(data, interval):
    """
    Optimization score or fitness for the overal run
    """
    return np.mean(data[-interval:])

def plot_seeds_stats(data_list, labels, fig_id=0, max_len=None, ylim_range=None, shade_alpha=0.2, 
                    xlabel_str="Iteration", ylabel_str=None, 
                    width=6, height=4, linewidth = 2,
                    top_n=None, top_n_interval=None):
    """
    @param: data: list of numpy arrays with data
    @param: labels: how to call each numpy array of data
    """

    # Preprocessing
    data_mean_list = []
    data_err_list = []
    data_len = []
    for di, data in enumerate(data_list):
        data_mean_list.append(np.mean(data, axis=0))
        data_err_list.append(np.std(data, axis=0))
        data_len.append(data_mean_list[-1].size)
    
    # Filtering out data that we should not plot
    if top_n is not None and top_n != 0:
        ## Calculating the score
        scores = []
        for di, data in enumerate(data_mean_list):
            scores.append(get_opt_score(data, top_n_interval))
        
        indx_sorted = np.argsort(scores)
        if top_n > 0:
            indices_retain = indx_sorted[-top_n:]
        elif top_n < 0:
            indices_retain = indx_sorted[:-top_n]
        
        labels_tmp , data_list_tmp, data_err_tmp, data_mean_tmp = [], [], [], []
        for i in indices_retain:
            labels_tmp.append(labels[i])
            data_list_tmp.append(data_list[i])
            data_err_tmp.append(data_err_list[i])
            data_mean_tmp.append(data_mean_list[i])

        labels, data_list, data_err_list, data_mean_list = labels_tmp , data_list_tmp, data_err_tmp, data_mean_tmp 
        # import pdb; pdb.set_trace()

    # Getting default colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # Plotting (shades come first to avoid overlap with mean curves)
    plt.figure(fig_id, figsize=(width, height))
    for di, data in enumerate(data_list):
        error = data_err_list[di]
        mean  = data_mean_list[di]
        x = np.arange(mean.shape[0])
        plt.fill_between(x, mean-error, mean+error,
            alpha=shade_alpha, facecolor=colors[di % len(colors)], antialiased=True)
    
    # Plotting Mean curves
    for di, data in enumerate(data_mean_list):
        x = np.arange(data.shape[0])
        plt.plot(data, linewidth=linewidth, color=colors[di % len(colors)], antialiased=True)

    # Setting properties
    plt.legend(labels)
    if max_len is not None:
        max_len = min(max_len, np.max(data_len))
    else:
        max_len = np.max(data_len)
    plt.xlim([0, max_len])
    if ylim_range is not None:
        plt.ylim(ylim_range)
    plt.xlabel(xlabel_str)
    plt.ylabel(ylabel_str)

    # TODO: Set font size if necessary

    # Remove white margins
    plt.tight_layout()

    plt.show(block=False)
    input("Enter to continue ...")

def plot_seeds(data_list, labels, fig_id=0, max_len=None, ylim_range=None, shade_alpha=0.2, xlabel_str="Iteration", ylabel_str=None, width=6, height=4, linewidth = 2):
    """
    @param: data: list of numpy arrays with data
    @param: labels: how to call each numpy array of data
    """

    # Preprocessing
    data_mean_list = []
    data_err_list = []
    data_len = []
    for di, data in enumerate(data_list):
        data_mean_list.append(np.mean(data, axis=0))
        data_err_list.append(np.std(data, axis=0))
        data_len.append(data_mean_list[-1].size)

    # Getting default colors
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']

    # Plotting (shades come first to avoid overlap with mean curves)
    for di, data in enumerate(data_list):
        plt.figure(fig_id + di, figsize=(width, height))
        error = data_err_list[di]
        mean  = data_mean_list[di]
        x = np.arange(mean.shape[0])
        plt.fill_between(x, mean-error, mean+error,
            alpha=shade_alpha, facecolor="blue", antialiased=True)
    
    # Plotting Mean curves
    for di, data in enumerate(data_mean_list):
        plt.figure(fig_id + di, figsize=(width, height))
        x = np.arange(mean.shape[0])
        plt.plot(data, linewidth=linewidth, color="blue", antialiased=True)
        for ds in range(data_list[di].shape[0]):
            plt.plot(data_list[di][ds, :], linewidth=1, antialiased=True)

        # Setting properties
        plt.title(labels[di])
        max_len = data.shape[0]
        plt.xlim([0, max_len])
        if ylim_range is not None:
            plt.ylim(ylim_range)
        plt.xlabel(xlabel_str)
        plt.ylabel(ylabel_str)

        # TODO: Set font size if necessary

        # Remove white margins
        plt.tight_layout()

    plt.show(block=False)
    input("Enter to continue ...")



def read_graph_seeds(folders, graph_names, filter_width=1):
    """
    Reads data in numpy arrays with the first dimension being a seed.
    Assumes that every specified folder contains "seed_X" folders over which it averages the graphs.
    @param: graph_names: one graph name for each folder (covers scenarios when different approaches may document same graph under different names)
    @param: filter_width: FIR filter width for data smoothing
    """

    ###################################
    ## PARAMETERS
    progress_filename = "progress.csv"

    # Parameters of window filtering of graphs
    filt_a = 1.
    filt_b = np.ones(filter_width) * 1./filter_width

    graph_num = len(folders)

    data_all = []
    data_filtered_all = []

    for folder_j, folder in enumerate(folders):
        print('Searching folder %s ...' % folder)
        seed_folders = subdir(folder)
        seeds_num = len(seed_folders)
        data = []
        data_filtered = []
        data_lengths = []
        # Indices of corresponding graphs in csv files
        indices  = []

        for seed_i in range(seeds_num):
            file_cur = os.path.join(seed_folders[seed_i], progress_filename)
            print('File %d = %s ' % (seed_i, file_cur))
            
            # Reading the csv file
            csv_data = pd.read_csv(file_cur, sep=',')
            header = list(csv_data.dtypes.index)
            data.append(csv_data.values)
            # alternative: csv_data[graph_names[folder_j]]

            # Finding the column corresponding to our data
            indices.append(header.index(graph_names[folder_j]))
            print('Field index = %d ' % indices[-1])
            
            # Filtering the data (for example if reward is too jerky)
            data_filtered.append(lfilter(filt_b, filt_a, data[-1][:, indices[-1]]))
            data_lengths.append(data_filtered[-1].size)
            print('Data length %d ' % data_lengths[-1])
        
        # Finding minimal data lenght
        mindata_len = np.min(data_lengths)
        maxdata_len = np.max(data_lengths)
        print('Min data length %d' % mindata_len)

        # Giving warning in case some of the seeds are longer than the minimal
        if mindata_len != maxdata_len: print("!!! WARN: Min (%d) != Max(%d) len: " % (mindata_len, maxdata_len))

        # Trimming all seeds data to this length
        data_filtered = [data_i[:mindata_len] for data_i in data_filtered]

        # Converting to numpy array
        print('Converting to np.array with the frist dimension being a seed ...')
        data_filtered = np.array(data_filtered)
        data = np.array(data)
        
        # Storing
        data_all.append(data)
        data_filtered_all.append(data_filtered) 

    return data_all, data_filtered_all


def read_and_plot_seeds(folders, graph_names, labels, filter_width=1, show_seeds=False, **kwargs):
    # Reading
    data, data_filtered = read_graph_seeds(folders=folders, graph_names=graph_names, filter_width=filter_width)
    # import pdb; pdb.set_trace()
    # Plotting
    if show_seeds:
        # Plotting individual seeds
        plot_seeds(data_list=data_filtered, labels=labels, **kwargs)
    else:
        # Plotting average + std among seeds
        plot_seeds_stats(data_list=data_filtered, labels=labels, **kwargs)
    return data, data_filtered


################################################################################
## Plotting a single
import argparse
from argparse import RawTextHelpFormatter, ArgumentDefaultsHelpFormatter
import sys, os

import time
import datetime
from dateutil.relativedelta import relativedelta

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
        default="rew_main_avg",
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

    # Reading and ploting
    read_and_plot_seeds(folders=[args.dir], graph_names=[args.graph], labels=[""], ylabel_str=args.graph, show_seeds=args.seeds)

    
    time_end = time.time()
    print("RUNTIME: ", runtime_str(time_end-time_start))

if __name__ == '__main__':
    main(sys.argv)