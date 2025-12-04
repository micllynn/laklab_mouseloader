#!/usr/bin/env python3
"""
Python script to generate a learning figure from a
Pavlovian behavioral experiment in mice
"""

import laklab_mouseloader.plt_beh as lml_plt
import argparse


def plot_behaviour(fname, noise=False):
    if noise is False:
        vp = lml_plt.VisualPavlovAnalysis(fname, lick_type='normal')
    elif noise is True:
        vp = lml_plt.VisualPavlovAnalysis(fname, lick_type='noise')

    vp.plt()


def main():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', type=str,
                        help='path to behaviour folder')
    parser.add_argument('-n', '--noise',
                        help='flag for noisy lick signal',
                        action='store_true')
    args = parser.parse_args()

    # run plotting function
    plot_behaviour(args.fname, args.noise)


if __name__ == "__main__":
    main()
