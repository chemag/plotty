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


YSCALE_VALUES = ('linear', 'log', 'symlog', 'logit')

default_values = {
    'debug': 0,
    'marker': '.',
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
    'legend_loc': 'upper right',
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
    # output parameter
    'outfile': None,
}

# filter ops
VALID_OPS = 'eq', 'ne', 'gt', 'ge', 'lt', 'le'


def is_valid_op(s):
    return s in VALID_OPS


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


def get_histogram(xlist, nbins, ratio, sigma, debug):
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
    nbins = min(nbins, len(xlist))
    bin_size = (maxx - minx) / nbins
    border_values = [minx + (i + 1) * bin_size for i in range(nbins - 1)]
    bin_list = [minx, ] + border_values + [maxx, ]
    real_xlist = [(r + l) / 2.0 for (l, r) in zip(bin_list[:-1], bin_list[1:])]

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
    raw_data = fin.read()
    return raw_data.decode('ascii')


def parse_csv(raw_data, sep):
    # split the input in lines
    lines = raw_data.split('\n')
    # look for named columns in line 0
    column_names = []
    if lines[0].strip().startswith('#'):
        column_names = lines[0].strip()[1:].strip().split(sep)
    # remove comment lines
    lines = [line for line in lines if not line.strip().startswith('#')]
    return column_names, lines


def filter_lines(lines, sep, prefilter, column_names):
    new_lines = []
    for line in lines:
        if not line:
            continue
        for fcol, fop, fval in prefilter:
            if is_int(fcol):
                fcol = int(fcol)
            else:
                # look for named columns
                assert fcol in column_names, (
                    'error: invalid fcol name: "%s"' % fcol)
                fcol = column_names.index(fcol)
            lval = line.split(sep)[int(fcol)]
            # implement eq and ne
            if ((fop == 'eq' and lval == fval) or
                    (fop == 'ne' and lval != fval)):
                new_lines.append(line)
            # implement gt, ge, lt, le
            elif fop in ('gt', 'ge', 'lt', 'le'):
                # make sure line val and filter val are numbers
                lval = float(lval)
                fval = float(fval)
                if ((fop == 'ge' and lval >= fval) or
                        (fop == 'gt' and lval > fval) or
                        (fop == 'le' and lval <= fval) or
                        (fop == 'lt' and lval < fval)):
                    new_lines.append(line)
    return new_lines


def get_column(line, sep, col, sep2, col2):
    if not line:
        # empty line
        return None
    # get component
    val = line.split(sep)[col]
    if col2 is not None:
        # use sep1, then sep2
        if not val:
            # empty column value
            return None
        # parse value
        val = val.split(sep2)[int(col2)]
    return val


def parse_line(line, i, sep, xcol, ycol, sep2, xcol2, ycol2):
    if not line:
        # empty line
        return None, None

    # get x component
    x = i if xcol == -1 else get_column(line, sep, xcol, sep2, xcol2)

    # get y component
    y = i if ycol == -1 else get_column(line, sep, ycol, sep2, ycol2)

    return x, y


def parse_data(raw_data, xshift_local, yshift_local, options):
    prefilter = options.filter
    sep = options.sep if options.sep != '' else None
    xcol = options.xcol
    xcol2 = options.xcol2
    sep2 = options.sep2 if options.sep2 != '' else None
    # histograms do not need a ycol
    ycol = options.ycol if not options.histogram else xcol
    ycol2 = options.ycol2

    # get starting data
    xlist, ylist = parse_data_internal(raw_data, prefilter, sep, xcol, ycol,
                                       sep2, xcol2, ycol2)

    # support for shift modes
    if xshift_local is not None:
        xlist = [(x + xshift_local) for x in xlist]
    if yshift_local is not None:
        ylist = [(y + yshift_local) for y in ylist]

    # support for histogram mode
    if options.histogram:
        xlist, ylist = get_histogram(xlist,
                                     options.histogram_bins,
                                     options.histogram_ratio,
                                     options.histogram_sigma,
                                     options.debug)

    # support for ydelta (plotting `y[k] - y[k-1]` instead of `y[k]`)
    if options.ydelta:
        ylist = [y1 - y0 for y0, y1 in zip([ylist[0]] + ylist[:-1], ylist)]

    # support for ycumulative (plotting `\Sum y[k]` instead of `y[k]`)
    if options.ycumulative:
        new_ylist = []
        for y in ylist:
            prev_y = new_ylist[-1] if new_ylist else 0
            new_ylist.append(y + prev_y)
        ylist = new_ylist

    return xlist, ylist


def is_int(s):
    if isinstance(s, int):
        return True
    return (s[1:].isdigit() if s[0] in ('-', '+') else s.isdigit())


def parse_data_internal(raw_data, prefilter, sep, xcol, ycol,
                        sep2, xcol2, ycol2):
    # convert the raw data into lines
    column_names, lines = parse_csv(raw_data, sep)

    # pre-filter lines
    if prefilter:
        lines = filter_lines(lines, sep, prefilter, column_names)

    xlist = []
    ylist = []

    # get the column IDs
    if is_int(xcol):
        xcol = int(xcol)
    else:
        # look for named columns
        assert xcol in column_names, 'error: invalid xcol name: "%s"' % xcol
        xcol = column_names.index(xcol)
    if is_int(ycol):
        ycol = int(ycol)
    else:
        # look for named columns
        assert ycol in column_names, 'error: invalid ycol name: "%s"' % ycol
        ycol = column_names.index(ycol)

    # parse all the lines
    for i, line in enumerate(lines):
        x, y = parse_line(line, i, sep, xcol, ycol, sep2, xcol2, ycol2)
        if x is not None and y is not None:
            # append values
            xlist.append(float(x))
            ylist.append(float(y))

    return xlist, ylist


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
    _ = ax1.legend(loc=options.legend_loc)

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
    parser.add_argument('--marker', action='store',
                        dest='marker', default=default_values['marker'],
                        metavar='MARKER',
                        help='use MARKER as plot marker',)
    parser.add_argument('--title', action='store',
                        dest='title', default=default_values['title'],
                        metavar='PLOTTITLE',
                        help='use PLOTTITLE plot title',)
    parser.add_argument('--xcol', action='store',
                        dest='xcol', default=default_values['xcol'],
                        metavar='XCOL',
                        help='use XCOL x col',)
    parser.add_argument('--xcol2', action='store', type=int,
                        dest='xcol2', default=default_values['xcol2'],
                        metavar='XCOL2',
                        help='use XCOL2 for refining x col',)
    parser.add_argument('--ycol', action='store',
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
    parser.add_argument('--legend-loc', action='store', type=str,
                        dest='legend_loc',
                        default=default_values['legend_loc'],
                        help='Legend location',)
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
    # per-line arguments
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
    parser.add_argument('-i', '--infile', action='append',
                        default=default_values['infile'],
                        metavar='input-file',
                        help='input file(s)',)
    # output
    parser.add_argument('outfile', type=str,
                        default=default_values['outfile'],
                        metavar='output-file',
                        help='output file',)
    # do the parsing
    options = parser.parse_args(argv[1:])
    # check the filter
    if options.filter:
        for fcol, fop, fval in options.filter:
            assert is_valid_op(fop), 'invalid filter: %s %s %s' % (
                fcol, fop, fval)
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

    # get all the input data into xy_data, a list of (xlist, ylist) tuples,
    # where `xlist` contains the x-axis values, `ylist` contains the y-axis
    # values
    fmt_list = ('C%i%s' % (i, options.marker) for i in range(10))
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
        xlist, ylist = parse_data(read_data(infile), xshift, yshift, options)
        label = options.label[index] if index < len(options.label) else ''
        fmt = (options.fmt[index] if index < len(options.fmt) else
               fmt_list[index])
        xy_data.append([xlist, ylist, label, fmt])

    # create the graph, adding each of the entries in xy_data
    ax1 = create_graph_begin(options)

    for xlist, ylist, label, fmt in xy_data:
        # `statistics` is a dictionary containing some statistics about the
        # distribution ('median', 'mean', 'stddev')
        statistics = {}
        if options.histogram:
            statistics['median'] = np.median(xlist)
            statistics['mean'] = np.mean(xlist)
            statistics['stddev'] = np.std(xlist)
        else:
            statistics['median'] = np.median(ylist)
            statistics['mean'] = np.mean(ylist)
            statistics['stddev'] = np.std(ylist)
        create_graph_draw(ax1, xlist, ylist, statistics, fmt, label, options)
    create_graph_end(ax1, options)


if __name__ == '__main__':
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
