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


DEFAULT_COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
DEFAULT_FMT = DEFAULT_COLORS

default_values = {
    'debug': 0,
    'title': '--title',
    'col': 0,
    # number of bins for the histogram
    'nbins': 50,
    'sigma': None,
    'filter': None,
    'sep': None,
    'xlabel': '--xlabel',
    'ylabel': '--ylabel',
    'add_mean': False,
    'add_median': False,
    'label': [],
    'fmt': [],
    'xlim': ['-', '-'],
    'ylim': ['-', '-'],
    'infile': [],
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

    xlist = []
    for i, row in enumerate(data):
        if not row:
            # empty row
            continue
        # get x component
        if options.col == -1:
            # use line number
            x = i
        else:
            # use column value
            x = float(row.split(sep)[options.col])
        # append values
        xlist.append(x)

    # remove values way over the average
    if options.sigma is not None:
        total_x = sum(xlist)
        mean_x = total_x / len(xlist)
        stddev_x = math.sqrt(sum((i - mean_x) ** 2 for i in xlist) /
                             (len(xlist) - 1))
        min_value = mean_x - (stddev_x * options.sigma)
        max_value = mean_x + (stddev_x * options.sigma)
        new_xlist = [i for i in xlist if i > min_value and i < max_value]
        if (len(xlist) - len(new_xlist)) / len(xlist) > 0.1:
            dropped_pct = 100. * (len(xlist) - len(new_xlist)) / len(xlist)
            print('Ignoring sigma removal of outliers (at sigma: %f would be '
                  'dropping %f%% of the values' % (
                      options.sigma, dropped_pct))
        else:
            if options.debug > 0:
                print('Removing %i of %i values sigma: %f stddev: %f '
                      'range: [%f, %f]' % (
                          len(xlist) - len(new_xlist),
                          len(xlist), options.sigma,
                          stddev_x, min_value, max_value))
            xlist = new_xlist
    return xlist


def create_graph_begin(options):
    # create figure
    fig, ax1 = plt.subplots(figsize=(8, 4))
    # tidy up the figure
    ax1.grid(True)
    ax1.legend(loc='right')
    ax1.set_title(options.title)
    ax1.set_xlabel(options.xlabel)
    ax1.set_ylabel('Frequency of the value')
    return ax1


def create_graph_draw(ax1, xlist, fmt, label, options):
    # plot the non-cumulative histogram
    n, bins, patches = ax1.hist(xlist, options.nbins, histtype='step',
                                color=fmt)

    # add median and average
    if options.add_median:
        plt.axvline(np.median(xlist), color=fmt, linestyle='dotted',
                    linewidth=1)
    if options.add_mean:
        plt.axvline(np.mean(xlist), color=fmt, linestyle='dashed', linewidth=1)


def create_graph_end(ax1, options):
    _ = ax1.legend(loc='upper right')

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
    parser.add_argument('--fmt', action='append',
                        dest='fmt', default=default_values['fmt'],
                        metavar='FMT',
                        help='use FMT format(s)',)
    parser.add_argument('--label', action='append',
                        dest='label', default=default_values['label'],
                        metavar='LABEL',
                        help='use LABEL label(s)',)
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
    parser.add_argument('-i', '--infile', action='append',
                        default=default_values['infile'],
                        metavar='input-file',
                        help='input file(s)',)
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
    # get infile(s)/outfile
    options.infile = [(sys.stdin if name == '-' else name) for name in
                      options.infile]
    if options.outfile == '-':
        options.outfile = sys.stdout
    # print results
    if options.debug > 0:
        print(options)

    # create the graph
    x_data = []
    for infile in options.infile:
        x_data.append(read_data(infile, options))

    ax1 = create_graph_begin(options)
    for index, xlist in enumerate(x_data):
        fmt = (options.fmt[index] if index < len(options.fmt) else
               DEFAULT_FMT[index])
        label = options.label[index] if index < len(options.label) else ''
        create_graph_draw(ax1, xlist, fmt, label, options)
    create_graph_end(ax1, options)


if __name__ == '__main__':
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
