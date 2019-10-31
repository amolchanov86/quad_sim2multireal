#!/usr/bin/env python
import pandas as pd
from time import sleep
from matplotlib import pyplot as plt
import argparse
import os

def read_graph_names(fname):
    with open(fname) as f:
        content = f.readlines()
    return [x.strip() for x in content]

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filename", nargs='*', help='Names of the csv files')
    parser.add_argument("--legend", "-l", nargs='*', help='Short names for the legend (must be as many as num of files provided)')
    parser.add_argument("--time", "-t", type=float, default=10, help='Sleep period')
    parser.add_argument("--graphs", "-g", nargs='*', help='Graphs to plot')
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
    # namelist = ['LossAfter',
    #             'BNN_DynModelSqLossAfter',
    #             'AverageReturn',
    #             'AverageDiscountedReturn',
    #             'Entropy',
    #             'Perplexity',
    #             'MeanKL',
    #             'Expl_MeanKL',
    #             'Expl_StdKL']

    # hide_rew_action
    # hide_rew_digit_entropy
    # hide_rew_digit_correct
    # seek_accuracy
    # seek_ep_len
    # seek_rew_digit_correct
    # seek_rew_digit_entropy
    # seek_digit_entropy
    # hide_RewAvg_Main
    # hide_AverageDiscountedReturn
    # hide_AverageReturn
    # hide_ExplainedVariance
    # hide_NumTrajs
    # hide_Entropy
    # hide_Perplexity
    # hide_StdReturn
    # hide_MaxReturn
    # hide_MinReturn
    # seek_RewAvg_Main
    # seek_AverageDiscountedReturn
    # seek_AverageReturn
    # seek_ExplainedVariance
    # seek_NumTrajs
    # seek_Entropy
    # seek_Perplexity
    # seek_StdReturn
    # seek_MaxReturn
    # seek_MinReturn
    # seek_vf_LossBefore
    # seek_vf_LossAfter
    # seek_vf_dLoss
    # AveragePolicyStd
    # AveragePolicyStd
    # hide_LossAfter
    # hide_MeanKL
    # hide_dLoss
    # seek_LossAfter
    # seek_MeanKL
    # seek_dLoss


    # hide_ep_len
    # hide_rew_action
    # hide_rew_digit_entropy
    # hide_rew_digit_correct
    # seek_accuracy
    # seek_ep_len
    # seek_rew_digit_correct
    # seek_rew_digit_entropy
    # seek_digit_entropy
    # hide_AverageReturn
    # hide_MaxReturn
    # seek_AverageReturn
    # seek_MaxReturn

    namelist = ['LossAfter',
                'BNN_DynModelSqLossAfter',
                'AverageReturn',
                'Entropy',
                'RewAvg_PredEntrChange',
                'RewAvg_PredEntrGlobChange',
                'RewAvg_Main',
                'RewAvg_Dyn',
                'RewAvg_PredEntr']

    # namelist = ['hide_ep_len',
    #             'seek_ep_len',
    #             'hide_AverageReturn',
    #             'seek_AverageReturn',
    #             'hide_rew_action',
    #             'hide_rew_digit_correct',
    #             'seek_rew_digit_entropy',
    #             'seek_digit_entropy',
    #             'seek_accuracy']

    # namelist = [
    #     'hide_ep_len',
    #     'hide_rew_seek_time',
    #     'hide_rew_action',
    #     'hide_rew_digit_entropy',
    #     'hide_rew_digit_correct',
    #     'hide_RewAvg_Main',
    #     'hide_AverageReturn',
    #     'seek_rew_digit_correct',
    #     'hide_MaxReturn'
    # ]

    namelist = [
        'seek_ep_len',
        'seek_AverageReturn',
        'hide_rew_action',
        'hide_rew_digit_correct_final',
        'seek_rew_digit_entropy_final',
        'seek_digit_entropy_final',
        'seek_accuracy_final',
        'seek_accuracy_samplewise',
        'seek_rew_digit_correct_sum'
    ]

    if args.graphs is not None:
        namelist = args.graphs
    print('Graphs to plot = ', namelist)
    if len(namelist) < 9:
        diff = 9 - len(namelist)
        namelist.extend([namelist[-1]] * diff)

    fig = plt.figure(1)
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

    colors_all = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    legend_lbl = args.legend
    if legend_lbl is None:
        legend_lbl = args.filename


    while True:
        lines = []
        file_i = -1
        for filename in args.filename:
            file_i+=1
            print('Plotting file = ', filename)

            if os.stat(filename).st_size == 0:
                print('File %s is empty. Skipping ...' % filename)
                sleep(2)
                break

            data = pd.read_csv(filename, sep=',')
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
                    figures[name].clear()

            for col_i in range(0, cols):
                for row_i in range(0, rows):
                    subplot_indx += 1
                    name = namelist[subplot_indx]
                    if name in name_indx:
                        n_i = name_indx[name]
                        line, = figures[name].plot(data[:, n_i], colors_all[file_i % len(colors_all)], label=legend_lbl[file_i])
                        figures[name].set_title(name)

            lines.append(line)
        # self.axis[name_i].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        #                          ncol=3, mode="expand", borderaxespad=0.)
        # OBS: use this  solution  AFTER you use  fig.set_size_inches() and BEFORE  you  use   fig.tight_layout()
        plt.draw()
        # plt.legend(lines, legend_lbl, loc = 'upper center', bbox_to_anchor = (0.5, 0),
        #            bbox_transform=plt.gcf().transFigure)
        # plt.legend(lines, legend_lbl, loc='lower center', bbox_to_anchor=(0, -0.1, 1, 1),
        #            bbox_transform=plt.gcf().transFigure, ncol=2)
        # plt.figlegend(lines, legend_lbl, loc='upper center', bbox_to_anchor=(0.5, -0.05),
        #           fancybox=True, shadow=True, ncol=1)
        # fig.legend(lines, legend_lbl, loc=(0.5, 0), ncol=1, bbox_to_anchor=(0, -0.1, 1, 1))
        plt.figlegend(lines, legend_lbl, loc='lower center', ncol=2, labelspacing=0.)
        # plt.show()
        plt.tight_layout()
        plt.pause(0.005)

        print('waiting for the next read ...')
        sleep(sleep_period)



if __name__ == "__main__":
    main()