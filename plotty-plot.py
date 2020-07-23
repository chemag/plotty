#!/usr/bin/env python3

"""plot.py: simple data plotter.

# http://stackoverflow.com/a/11249340

# runme
# $ echo -e "1 2\n2 2\n3 5\n4 7\n5 1\n" | ./plotty-plot.py -i - /tmp/foo.png
"""

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import sys


DEFAULT_MARKER = '.'
DEFAULT_COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
DEFAULT_FMT = ['%s%s' % (color, DEFAULT_MARKER) for color in DEFAULT_COLORS]

YSCALE_VALUES = ('linear', 'log', 'symlog', 'logit')

default_values = {
    'debug': 0,
    'title': '--title',
    'xcol': 0,
    'xcol2': None,
    'ycol': 1,
    'ycol2': None,
    'ydelta': False,
    'ycumulative': False,
    'filter': None,
    'sep': None,
    'sep2': None,
    # histogram information
    'histogram': False,
    # number of bins for the histogram
    'histogram_bins': 50,
    # filter outliers
    'histogram_sigma': None,
    # use relative values
    'histogram_ratio': False,
    'xlabel': '--xlabel',
    'ylabel': '--ylabel',
    'xlim': ['-', '-'],
    'ylim': ['-', '-'],
    'yscale': 'linear',
    'add_mean': False,
    'add_median': False,
    'add_stddev': False,
    # per-line parameters
    'xshift': [],
    'yshift': [],
    'label': [],
    'fmt': [],
    'infile': [],
    'outfile': None,
}

# filter ops
VALID_OPS = 'eq', 'ne'


def remove_outliers(xlist, sigma):
    # remove values way over the average
    total_x = sum(xlist)
    mean_x = total_x / len(xlist)
    stddev_x = math.sqrt(sum((i - mean_x) ** 2 for i in xlist) /
                         (len(xlist) - 1))
    min_value = mean_x - (stddev_x * sigma)
    max_value = mean_x + (stddev_x * sigma)
    in_xlist = [i for i in xlist if i > min_value and i < max_value]
    out_xlist = [i for i in xlist if i < min_value or i > max_value]
    return in_xlist, out_xlist, [min_value, max_value]


def get_histogram(xlist, bins, ratio, sigma, debug):
    if sigma is not None:
        in_xlist, out_xlist, in_range = (
            remove_outliers(xlist, sigma))
        if (len(xlist) - len(in_xlist)) / len(xlist) > 0.1:
            dropped_pct = 100. * (len(xlist) - len(in_xlist)) / len(xlist)
            print('Ignoring sigma removal of outliers (at sigma: %f would be '
                  'dropping %f%% of the values' % (
                      sigma, dropped_pct))
        else:
            if debug > 0:
                print('Removing %i of %i values sigma: %f range: [%f, %f]' % (
                          len(xlist) - len(in_xlist),
                          len(xlist), sigma,
                          in_range[0], in_range[1]))
            xlist = in_xlist

    # get the extreme points
    minx = min(xlist)
    maxx = max(xlist)
    # do not bin more than needed
    nbins = min(bins, len(set(xlist)))
    bin_size = (maxx - minx) / (nbins - 1)
    real_xlist = [minx + i * bin_size for i in range(nbins)]
    border_values = [minx + (i + 0.5) * bin_size for i in range(nbins - 1)]
    # get the number of values in the histogram
    ylist = [0] * nbins
    for x in xlist:
        i = 0
        for border_value in border_values:
            if x < border_value:
                ylist[i] += 1
                break
            i += 1
        else:
            ylist[-1] += 1

    # support for ratio histograms
    if ratio:
        ylist = [(1.0 * y) / sum(ylist) for y in ylist]

    return real_xlist, ylist


def read_data(infile):
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
    return data.decode('ascii')


def parse_data(data, xshift_local, yshift_local, options):
    return parse_data_internal(data, xshift_local, yshift_local,
                               **vars(options))


def parse_data_internal(data, xshift_local=None, yshift_local=None, **kwargs):
    debug = kwargs.get('debug', default_values['debug'])

    # split the input in lines
    data = data.split('\n')
    # remove comment lines
    data = [line for line in data if not line.strip().startswith('#')]
    # break up each line in fields
    sep = kwargs.get('sep', default_values['sep'])
    if sep is None:
        # use space and tab
        data = [item.replace('\t', ' ') for item in data]
    sep = sep if sep is not None else ' '

    # pre-filter lines
    prefilter = kwargs.get('filter', default_values['filter'])
    if prefilter:
        new_data = []
        for row in data:
            if not row:
                continue
            for col, op, val in prefilter:
                field = row.split(sep)[int(col)]
                if ((op == 'eq' and field == val) or
                        (op == 'ne' and field != val)):
                    new_data.append(row)
        data = new_data

    xlist = []
    ylist = []
    sep2 = kwargs.get('sep2', default_values['sep2'])
    sep2 = sep2 if sep2 is not None else ' '

    statistics = {}
    xcol = kwargs.get('xcol', default_values['xcol'])
    xcol2 = kwargs.get('xcol2', default_values['xcol2'])
    histogram = kwargs.get('histogram', default_values['histogram'])
    ycol = kwargs.get('ycol', default_values['ycol'])
    ycol2 = kwargs.get('ycol2', default_values['ycol2'])
    for i, row in enumerate(data):
        if not row:
            # empty row
            continue
        # get x component
        if xcol == -1:
            # use line number
            x = i
        elif xcol2 is None:
            # use column value
            x = float(row.split(sep)[xcol])
        else:
            # get column value
            value = row.split(sep)[xcol]
            # parse column value
            if not value:
                # empty column value
                continue
            # parse value
            x = float(value.split(sep2)[xcol2])
        # get y component
        if histogram:
            # will replace value later
            y = 0
        elif ycol == -1:
            # use line number
            y = i
        elif ycol2 is None:
            # use column value
            y = float(row.split(sep)[ycol])
        else:
            # get column value
            value = row.split(sep)[ycol]
            # parse column value
            if not value:
                # empty column value
                continue
            # parse value
            y = float(value.split(sep2)[ycol2])
        # append values
        xlist.append(x)
        ylist.append(y)

    # support for shift modes
    if xshift_local is not None:
        xlist = [(x + xshift_local) for x in xlist]
    if yshift_local is not None:
        ylist = [(y + yshift_local) for y in ylist]

    # support for histogram mode
    histogram_bins = kwargs.get('histogram_bins',
                                default_values['histogram_bins'])
    histogram_ratio = kwargs.get('histogram_ratio',
                                 default_values['histogram_ratio'])
    histogram_sigma = kwargs.get('histogram_sigma',
                                 default_values['histogram_sigma'])
    if histogram:
        statistics['median'] = np.median(xlist)
        statistics['mean'] = np.mean(xlist)
        statistics['stddev'] = np.std(xlist)
        xlist, ylist = get_histogram(xlist, histogram_bins, histogram_ratio,
                                     histogram_sigma, debug)
    else:
        statistics['median'] = np.median(ylist)
        statistics['mean'] = np.mean(ylist)
        statistics['stddev'] = np.std(ylist)

    # support for plotting `y[k] - y[k-1]` instead of `y[k]`
    ydelta = kwargs.get('ydelta', default_values['ydelta'])
    ycumulative = kwargs.get('ycumulative', default_values['ycumulative'])
    if ydelta:
        ylist = [y1 - y0 for y0, y1 in zip([ylist[0]] + ylist[:-1], ylist)]
    elif ycumulative:
        new_ylist = []
        for y in ylist:
            prev_y = new_ylist[-1] if new_ylist else 0
            new_ylist.append(y + prev_y)
        ylist = new_ylist

    return xlist, ylist, statistics


def create_graph_begin(options):
    # create figure
    fig = plt.figure()
    # plt.gca().set_xlim([xstart, xstart + 1100000000])
    # plt.gca().set_ylim([0, 100])
    # # add a vertical line
    # plt.axvline(x=game_start)
    # # add a horizontal line
    # plt.axhline(y=1)
    ax1 = fig.add_subplot(111)
    ax1.set_title(options.title)
    ax1.set_xlabel(options.xlabel)
    ax1.set_ylabel(options.ylabel)
    return ax1


def create_graph_draw(ax1, xlist, ylist, statistics, fmt, label, options):
    ax1.plot(xlist, ylist, fmt, label=label)
    if options.debug > 1:
        print('ax1.plot(%r, %r, \'%s\', label=%r)' % (
            list(xlist), ylist, fmt, label))
    if options.histogram:
        if options.add_median:
            plt.axvline(statistics['median'], color=fmt[0], linestyle='dotted',
                        linewidth=1)
        if options.add_mean:
            plt.axvline(statistics['mean'], color=fmt[0], linestyle='dotted',
                        linewidth=1)
        if options.add_stddev:
            plt.axvline(statistics['mean'] + statistics['stddev'],
                        color=fmt[0], linestyle='dotted', linewidth=1)
            plt.axvline(statistics['mean'] - statistics['stddev'],
                        color=fmt[0], linestyle='dotted', linewidth=1)
    else:
        if options.add_median:
            plt.axhline(statistics['median'], color=fmt[0], linestyle='dotted',
                        linewidth=1)
        if options.add_mean:
            plt.axhline(statistics['mean'], color=fmt[0], linestyle='dotted',
                        linewidth=1)
        if options.add_stddev:
            plt.axhline(statistics['mean'] + statistics['stddev'],
                        color=fmt[0], linestyle='dotted', linewidth=1)
            plt.axhline(statistics['mean'] - statistics['stddev'],
                        color=fmt[0], linestyle='dotted', linewidth=1)


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

    # set yscale
    plt.yscale(options.yscale)

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
    parser.add_argument('--xcol', action='store', type=int,
                        dest='xcol', default=default_values['xcol'],
                        metavar='XCOL',
                        help='use XCOL x col',)
    parser.add_argument('--xcol2', action='store', type=int,
                        dest='xcol2', default=default_values['xcol2'],
                        metavar='XCOL2',
                        help='use XCOL2 for refining x col',)
    parser.add_argument('--ycol', action='store', type=int,
                        dest='ycol', default=default_values['ycol'],
                        metavar='YCOL',
                        help='use YCOL y col',)
    parser.add_argument('--ycol2', action='store', type=int,
                        dest='ycol2', default=default_values['ycol2'],
                        metavar='YCOL2',
                        help='use YCOL2 for refining y col',)
    parser.add_argument('--ydelta', action='store_const', const=True,
                        dest='ydelta', default=default_values['ydelta'],
                        help='use $y[k] = (y[k] - y[k-1])$',)
    parser.add_argument('--ycumulative', action='store_const', const=True,
                        dest='ycumulative',
                        default=default_values['ycumulative'],
                        help='use $y[k] = \\sum_i=0^k y[i]$',)
    parser.add_argument('--filter', action='append', type=str, nargs=3,
                        dest='filter', default=default_values['filter'],
                        metavar=('COL', 'OP', 'VAL'),
                        help='select only rows where COL OP VAL is true',)
    parser.add_argument('--sep', action='store', type=str,
                        dest='sep', default=default_values['sep'],
                        metavar='SEP',
                        help='use SEP as separator',)
    parser.add_argument('--sep2', action='store', type=str,
                        dest='sep2', default=default_values['sep2'],
                        metavar='SEP2',
                        help='use SEP2 as alternate separator',)
    parser.add_argument('--histogram', action='store_const', const=True,
                        dest='histogram', default=default_values['histogram'],
                        help='sort and bin xlist, get ylist as histogram',)
    parser.add_argument('--histogram-bins', action='store', type=int,
                        dest='histogram_bins',
                        default=default_values['histogram_bins'],
                        metavar='NBINS',
                        help='use NBINS bins',)
    parser.add_argument('--histogram-sigma', action='store', type=float,
                        dest='histogram_sigma',
                        default=default_values['histogram_sigma'],
                        metavar='SIGMA',
                        help='use avg += (SIGMA * stddev) to remove outliers',)
    parser.add_argument('--histogram-ratio', action='store_const', const=True,
                        dest='histogram_ratio',
                        default=default_values['histogram_ratio'],
                        help='use ratio for ylist instead of total number',)
    parser.add_argument('--xlabel', action='store',
                        dest='xlabel', default=default_values['xlabel'],
                        metavar='XLABEL',
                        help='use XLABEL x label',)
    parser.add_argument('--ylabel', action='store',
                        dest='ylabel', default=default_values['ylabel'],
                        metavar='YLABEL',
                        help='use YLABEL x label',)
    parser.add_argument('--xshift', action='append',
                        dest='xshift', default=default_values['xshift'],
                        metavar='XSHIFT',
                        help='use XSHIFT x shift(s)',)
    parser.add_argument('--yshift', action='append',
                        dest='yshift', default=default_values['yshift'],
                        metavar='YSHIFT',
                        help='use YSHIFT y shift(s)',)
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
    parser.add_argument('--add-stddev', action='store_const',
                        dest='add_stddev', const=True,
                        default=default_values['add_stddev'],
                        help='Add 2 lines at mean +- stddev',)
    parser.add_argument('--xlim', action='store', type=str, nargs=2,
                        dest='xlim', default=default_values['xlim'],
                        metavar=('left', 'right'),)
    parser.add_argument('--ylim', action='store', type=str, nargs=2,
                        dest='ylim', default=default_values['ylim'],
                        metavar=('bottom', 'top'),)
    parser.add_argument('--yscale', action='store', type=str,
                        dest='yscale', default=default_values['yscale'],
                        choices=YSCALE_VALUES,
                        metavar='[%s]' % (' | '.join(YSCALE_VALUES,)),
                             help='yscale values',)
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
    xy_data = []
    for index, infile in enumerate(options.infile):
        xshift = (float(options.xshift[index]) if index < len(options.xshift)
                  else None)
        if xshift is not None:
            print('shifting x by %f' % xshift)
        yshift = (float(options.yshift[index]) if index < len(options.yshift)
                  else None)
        if yshift is not None:
            print('shifting y by %f' % yshift)
        xy_data.append(parse_data(read_data(infile), xshift, yshift, options))

    ax1 = create_graph_begin(options)
    for index, (xlist, ylist, statistics) in enumerate(xy_data):
        fmt = (options.fmt[index] if index < len(options.fmt) else
               DEFAULT_FMT[index])
        label = options.label[index] if index < len(options.label) else ''
        create_graph_draw(ax1, xlist, ylist, statistics, fmt, label, options)
    create_graph_end(ax1, options)


if __name__ == '__main__':
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
