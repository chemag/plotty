#!/usr/bin/env python3

"""histogram.py: simple histogram plotter.

# http://matplotlib.org/examples/statistics/histogram_demo_cumulative.html

# runme
# $ echo -e "1\n2\n1\n2\n3\n1\n4\n" | ./plotty-histogram.py - /tmp/bar.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
import sys


default_values = {
    'debug': 0,
    'title': 'Plot Title',
    'label': '',
    'col': 0,
    # number of bins for the histogram
    'nbins': 50,
    'sigma': None,
    'fmt': 'ro',
    'sep': None,
    'xlabel': 'x axis',
    'ylabel': 'y axis',
    'add_mean': False,
    'add_median': False,
    'infile': 'histogram_file.txt',
    'outfile': None,
}


def read_data(infile, options):
    # open infile
    if infile != sys.stdin:
        try:
            fin = open(infile, 'rb+')
        except IOError:
            print('Error: cannot open file "%s":', infile)
    else:
        fin = sys.stdin.buffer

    # read data
    data = fin.read()

    # split the input in lines
    data = data.decode('ascii').split('\n')
    # remove comment lines
    data = [line for line in data if not line.startswith('#')]
    # break up each line in fields
    if options.sep is None:
        # use space and tab
        data = [item.replace('\t', ' ') for item in data]
    sep = options.sep if options.sep is not None else ' '
    if options.col == -1:
        x = range(len([row for row in data if row]))
    else:
        x = [float(row.split(sep)[options.col]) for row in data if row]
    # remove values way over the average
    if options.sigma is not None:
        total_x = sum(x)
        mean_x = total_x / len(x)
        stddev_x = math.sqrt(sum((i - mean_x) ** 2 for i in x) / (len(x) - 1))
        min_value = mean_x - (stddev_x * options.sigma)
        max_value = mean_x + (stddev_x * options.sigma)
        new_x = [i for i in x if i > min_value and i < max_value]
        if (len(x) - len(new_x)) / len(x) > 0.1:
            print('Ignoring sigma removal of outliers (at sigma: %f would be '
                  'dropping %f%% of the values' % (
                      options.sigma, 100. * (len(x) - len(new_x)) / len(x)))
        else:
            if options.debug > 0:
                print('Removing %i of %i values sigma: %f stddev: %f '
                      'range: [%f, %f]' % (
                          len(x) - len(new_x), len(x), options.sigma,
                          stddev_x, min_value, max_value))
            x = new_x
    return x


def create_graph(x, options):
    # create figure
    fig, ax = plt.subplots(figsize=(8, 4))

    # plot the non-cumulative histogram
    n, bins, patches = ax.hist(x, options.nbins, histtype='step')

    # add median and average
    if options.add_median:
        plt.axvline(np.median(x), color='r', linestyle='dotted', linewidth=1)
    if options.add_mean:
        plt.axvline(np.mean(x), color='k', linestyle='dashed', linewidth=1)

    # tidy up the figure
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title(options.title)
    ax.set_xlabel(options.xlabel)
    ax.set_ylabel('Frequency of the value')

    # plt.show()
    print('output is %s' % options.outfile)
    plt.savefig('%s' % options.outfile)


def get_options(argv):
    # parse opts
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--debug', action='count',
                        dest='debug', default=default_values['debug'],
                        help='Increase verbosity (multiple times for more)',)
    parser.add_argument('--quiet', action='store_const',
                        dest='debug', const=-1,
                        help='Zero verbosity',)
    parser.add_argument('--title', action='store',
                        dest='title', default=default_values['title'],
                        metavar='PLOTTITLE',
                        help='use PLOTTITLE plot title',)
    parser.add_argument('-c', '--col', action='store', type=int,
                        dest='col', default=default_values['col'],
                        metavar='XCOL',
                        help='use COL col',)
    parser.add_argument('-b', '--nbins', action='store', type=int,
                        dest='nbins', default=default_values['nbins'],
                        metavar='NBINS',
                        help='use NBINS bins',)
    parser.add_argument('--sigma', action='store', type=float,
                        dest='sigma', default=default_values['sigma'],
                        metavar='SIGMA',
                        help='use avg += (SIGMA * stddev) to remove outliers',)
    parser.add_argument('--sep', action='store', type=str,
                        dest='sep', default=default_values['sep'],
                        metavar='SEP',
                        help='use SEP as separator',)
    parser.add_argument('--xlabel', action='store',
                        dest='xlabel', default=default_values['xlabel'],
                        metavar='XLABEL',
                        help='use XLABEL x label',)
    parser.add_argument('--ylabel', action='store',
                        dest='ylabel', default=default_values['ylabel'],
                        metavar='YLABEL',
                        help='use YLABEL x label',)
    parser.add_argument('--label', action='store',
                        dest='label', default=default_values['label'],
                        metavar='LABEL',
                        help='use YLABEL label',)
    parser.add_argument('--add-mean', action='store_const',
                        dest='add_mean', const=True,
                        default=default_values['add_mean'],
                        help='Add a line at the mean',)
    parser.add_argument('--add-median', action='store_const',
                        dest='add_median', const=True,
                        default=default_values['add_median'],
                        help='Add a line at the median',)
    parser.add_argument('infile', type=str,
                        default=default_values['infile'],
                        metavar='input-file',
                        help='input file',)
    parser.add_argument('outfile', type=str,
                        default=default_values['outfile'],
                        metavar='output-file',
                        help='output file',)
    # do the parsing
    options = parser.parse_args(argv[1:])
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    # get infile/outfile
    if options.infile == '-':
        options.infile = sys.stdin
    if options.outfile == '-':
        options.outfile = sys.stdout
    # print results
    if options.debug > 2:
        print(options)
    # create the graph
    x = read_data(options.infile, options)
    create_graph(x, options)


if __name__ == '__main__':
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
