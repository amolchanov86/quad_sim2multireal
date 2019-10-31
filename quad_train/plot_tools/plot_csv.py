#!/usr/bin/env python
import pandas as pd
from time import sleep
from matplotlib import pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filename", help='Name of the csv file')
    parser.add_argument("--time", "-t", type=float, default=10, help='Sleep period')
    args = parser.parse_args()

    sleep_period = 10

    # MaxReturn,LossAfter,
    # BNN_DynModelSqLossAfter,
    # BNN_DynModelSqLossBefore,
    # AverageReturn,
    # Expl_MaxKL,
    # Iteration,
    # AverageDiscountedReturn,
    # MinReturn,
    # Expl_MinKL,
    # dLoss,Entropy,
    # AveragePolicyStd,
    # StdReturn,
    # Perplexity,
    # MeanKL,
    # ExplainedVariance,
    # Expl_MeanKL,
    # NumTrajs,
    # Expl_StdKL
    namelist = ['LossAfter',
                'BNN_DynModelSqLossAfter',
                'AverageReturn',
                'AverageDiscountedReturn',
                'Entropy',
                'Perplexity',
                'MeanKL',
                'Expl_MeanKL',
                'Expl_StdKL']

    plt.figure(1)
    subplot_indx = -1
    cols = 3
    rows = 3
    figures = {}
    for col_i in range(0, cols):
        for row_i in range(0, rows):
            subplot_indx += 1
            name = namelist[subplot_indx]
            figures[name] = plt.subplot('%d%d%d' % (cols, rows, subplot_indx + 1))

    plt.ion()
    plt.show()


    while True:
        data = pd.read_csv(args.filename, sep=',')
        namelist_cur = data.dtypes.index
        data = data.as_matrix()
        # print 'data= ', data

        # print 'namelist = ', namelist_cur
        # namelist_cur = data[0,:]
        # data = data[1:,:]

        # namelist_cur = namelist
        names_num = len(namelist_cur)

        subplot_indx = -1
        name_indx = {}
        for n_i in range(0,names_num):
            name_indx[namelist_cur[n_i]] = n_i
        # print 'dtype = ', type(data)

        for col_i in range(0, cols):
            for row_i in range(0, rows):
                subplot_indx += 1
                name = namelist[subplot_indx]
                n_i = name_indx[name]
                figures[name].clear()
                figures[name].plot(data[:, n_i])
                figures[name].set_title(name)
                plt.draw()
                plt.tight_layout()
                plt.pause(0.001)

        print('waiting for the next read ...')
        sleep(sleep_period)



if __name__ == "__main__":
    main()