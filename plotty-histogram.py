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
    'title': '--title',
    'label': '',
    'col': 0,
    # number of bins for the histogram
    'nbins': 50,
    'sigma': None,
    'filter': None,
    'fmt': 'ro',
    'sep': None,
    'xlabel': '--xlabel',
    'ylabel': '--ylabel',
    'add_mean': False,
    'add_median': False,
    'xlim': ['-', '-'],
    'ylim': ['-', '-'],
    'infile': None,
    'outfile': None,
}

# filter ops
VALID_OPS = 'eq', 'ne'

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

    # filter lines
    if options.filter:
        new_data = []
        for row in data:
            if not row:
                continue
            for col, op, val in options.filter:
                field = row.split(sep)[int(col)]
                if ((op == 'eq' and field == val) or
                        (op == 'ne' and field != val)):
                    new_data.append(row)
        data = new_data

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

    # set xlim/ylim
    if options.xlim[0] != '-':
        plt.xlim(left=float(options.xlim[0]))
    if options.xlim[1] != '-':
        plt.xlim(right=float(options.xlim[1]))
    if options.ylim[0] != '-':
        plt.ylim(bottom=float(options.ylim[0]))
    if options.ylim[1] != '-':
        plt.ylim(top=float(options.ylim[1]))

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
                        metavar='COL',
                        help='use COL col',)
    parser.add_argument('-b', '--nbins', action='store', type=int,
                        dest='nbins', default=default_values['nbins'],
                        metavar='NBINS',
                        help='use NBINS bins',)
    parser.add_argument('--sigma', action='store', type=float,
                        dest='sigma', default=default_values['sigma'],
                        metavar='SIGMA',
                        help='use avg += (SIGMA * stddev) to remove outliers',)
    parser.add_argument('--filter', action='append', type=str, nargs=3,
                        dest='filter', default=default_values['filter'],
                        metavar=('COL', 'OP', 'VAL'),
                        help='select only rows where COL OP VAL is true',)
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
                        help='use LABEL label',)
    parser.add_argument('--add-mean', action='store_const',
                        dest='add_mean', const=True,
                        default=default_values['add_mean'],
                        help='Add a line at the mean',)
    parser.add_argument('--add-median', action='store_const',
                        dest='add_median', const=True,
                        default=default_values['add_median'],
                        help='Add a line at the median',)
    parser.add_argument('--xlim', action='store', type=str, nargs=2,
                        dest='xlim', default=default_values['xlim'],
                        metavar=('left', 'right'),)
    parser.add_argument('--ylim', action='store', type=str, nargs=2,
                        dest='ylim', default=default_values['ylim'],
                        metavar=('bottom', 'top'),)
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

    # check the filter
    if options.filter:

        def is_int(s):
            if s[0] in ('-', '+'):
                return s[1:].isdigit()
            return s.isdigit()

        def is_op(s):
            return s in VALID_OPS
        for col, op, val in options.filter:
            assert is_int(col) and is_op(op), 'invalid filter: %s %s %s' % (
                col, op, val)
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
