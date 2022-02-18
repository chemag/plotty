#!/usr/bin/env python3

"""plot.py: simple data plotter.

# http://stackoverflow.com/a/11249340

# runme
# $ echo -e "1,2\n2,2\n3,5\n4,7\n5,1\n" | ./plotty-plot.py -i - /tmp/plot.png
"""

import argparse
import datetime
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import os
from scipy.optimize import curve_fit
import sys


__version__ = '0.2'

MAX_INFILE_LIST_LEN = 20
SCALE_VALUES = (None, 'linear', 'log', 'symlog', 'logit')
VALID_LEGEND_LOCS = (
    'none',
    'best',
    'upper right',
    'upper left',
    'lower left',
    'lower right',
    'right',
    'center left',
    'center right',
    'lower center',
    'upper center',
    'center',
)


# https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
VALID_MATPLOTLIB_LINESTYLES = {
    # make sure 2-character linestyles go before 1-character ones
    '--': 'dashed',
    '-.': 'dashdot',
    '-': 'solid',
    ':': 'dotted',
}


# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
# https://matplotlib.org/stable/api/markers_api.html
VALID_MATPLOTLIB_MARKERS = (
    '.',
    ',',
    'o',
    'v',
    '^',
    '<',
    '>',
    '1',
    '2',
    '3',
    '4',
    '8',
    's',
    'p',
    'P',
    '*',
    'h',
    'H',
    '+',
    'x',
    'X',
    'D',
    'd',
    '|',
    '_',
)


# https://matplotlib.org/2.0.2/api/colors_api.html
VALID_MATPLOTLIB_COLORS = (
    'b',
    'g',
    'r',
    'c',
    'm',
    'y',
    'k',
    'w',
)


VALID_COLUMN_FMTS = ('float', 'int', 'unix')


VALID_HISTOGRAM_TYPES = ('raw', 'pdf', 'cdf')


# parameter notes
# There are 3 type of parameters
# * 1. single: same value for every line (e.g. --xfmt)
# * 2. per-axis: need a different value per axis (e.g. --ylim)
#   We use --twinx to create multiple axis
# * 3. per-line: need a different value per line (e.g. --ycol, --fmt)
#   We attach lines to axis based on the exact parameter location.
#   * subtypes depending on what to do if there are not as many occurrences
#     as '-i' elements.
#     * 3.1. keep the last one if not enough
#     * 3.2. use default value if not enough
#     * 3.3. positional parameters: attach to the previous '-i'
# * 4. multiple applications:
#   * e.g. filter (prefilter): allows multiple filtering
#
# List of per-axis and per-line parameters:
# * used in parse_data()
#   * ycol [v]
#   * yfmt [ ]
#   * ydelta [ ]
#   * ycumulative [ ]
#   * xshift [v]
#   * yshift [v]
#   * label [v]
#   * fmt [v]
#   * color [v]
#   * infile [v]
# * used in create_graph_begin()
#   * ylabel [v]
# * used in create_graph_draw()
#   * xfmt [ ]
#   * histogram [ ]
#   * add_median [ ]
#   * add_mean [ ]
#   * add_stddev [ ]
#   * add_regression [ ]
# * used in create_graph_end()
#   * legend_loc [v]
#   * xlim [ ]
#   * ylim [v]
#   * xscale [ ]
#   * yscale [ ]

#
default_values = {
    'debug': 0,
    'marker': '.',
    'figsize': matplotlib.rcParams['figure.figsize'],
    'title': '--title',
    'header': False,
    'xcol': 0,
    'xcol2': None,
    'ycol': [],
    'ycol2': None,
    'xfmt': 'float',
    'yfmt': 'float',
    'fmtdate': '%Y-%m-%d\n%H:%M:%S',
    'ydelta': False,
    'ycumulative': False,
    'filter': None,
    # use '' to separate using None (any number of spaces/tabs)
    'sep': ',',
    'sep2': ',',
    'legend_loc': 'best',
    # histogram information
    'histogram': False,
    # number of bins for the histogram
    'histogram_bins': 50,
    # filter out zeroes
    'histogram_nozeroes': False,
    # filter outliers
    'histogram_sigma': None,
    # histogram type
    'histogram_type': 'raw',
    'xlabel': '--xlabel',
    'ylabel': [],
    'xlim': ['-', '-'],
    'ylim': [],
    'xscale': None,
    'yscale': None,
    'add_mean': False,
    'add_median': False,
    'add_stddev': False,
    'add_regression': False,
    # per-line parameters
    'xshift': [],
    'yshift': [],
    'label': [],
    'fmt': [],
    'color': [],
    'infile': [],
    # batch conf parameters
    'batch_infile': None,
    'batch_sep': ',',
    'batch_col': None,
    'batch_label_col': None,
    'batch_filter': None,
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


def get_histogram(xlist, nbins, htype, nozeroes, sigma, debug):
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
    # remove nan values
    xlist = [x for x in xlist if x is not np.nan]
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

    # support for raw, pdf (ratio), and cdf histograms
    if htype == 'raw':
        pass
    elif htype == 'pdf':
        ylist = [(1.0 * y) / sum(ylist) for y in ylist]
    elif htype == 'cdf':
        # start with the pdf
        ylist = [(1.0 * y) / sum(ylist) for y in ylist]
        # add the pdf up
        new_ylist = []
        accum = 0.0
        for y in ylist:
            accum += y
            new_ylist.append(accum)
        ylist = new_ylist

    if nozeroes:
        real_xlist, ylist = zip(*[(x, y) for x, y in zip(real_xlist, ylist)
                                if y != 0])

    return real_xlist, ylist


def read_file(infile):
    # open infile
    if infile == '-':
        infile = '/dev/fd/0'
    with open(infile, 'r') as fin:
        # read data
        raw_data = fin.read()
    return raw_data


def parse_csv(raw_data, sep, header):
    # split the input in lines
    lines = raw_data.split('\n')
    # look for named columns in line 0
    column_names = []
    if lines[0].strip().startswith('#') or header:
        column_names = lines[0].strip().split(sep)
        if column_names[0].startswith('#'):
            column_names[0] = column_names[0][1:]
        # remove extra spaces
        column_names = [colname.strip() for colname in column_names]
        lines = lines[1:]
    # remove comment lines
    lines = [line for line in lines if not line.strip().startswith('#')]
    return column_names, lines


def filter_lines(lines, sep, prefilter, column_names):
    new_lines = []
    for line in lines:
        if not line:
            continue
        must_keep_line = True
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
            if fop in ('eq', 'ne'):
                if ((fop == 'eq' and lval != fval) or
                        (fop == 'ne' and lval == fval)):
                    must_keep_line = False
            # implement gt, ge, lt, le
            elif fop in ('gt', 'ge', 'lt', 'le'):
                # make sure line val and filter val are numbers
                lval = float(lval)
                fval = float(fval)
                if ((fop == 'ge' and lval < fval) or
                        (fop == 'gt' and lval <= fval) or
                        (fop == 'le' and lval > fval) or
                        (fop == 'lt' and lval >= fval)):
                    must_keep_line = False
        if must_keep_line:
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


def parse_data(raw_data, ycol, xshift_local, yshift_local, options):
    prefilter = options.filter
    sep = options.sep if options.sep != '' else None
    xcol = options.xcol
    xcol2 = options.xcol2
    sep2 = options.sep2 if options.sep2 != '' else None
    ycol2 = options.ycol2
    xfmt = options.xfmt
    yfmt = options.yfmt

    # get starting data
    xlist, ylist = parse_data_internal(raw_data, prefilter, sep, xcol, ycol,
                                       sep2, xcol2, ycol2, xfmt, yfmt,
                                       options.header)

    # support for shift modes
    if xshift_local is not None:
        xlist = [(x + xshift_local) for x in xlist]
    if yshift_local is not None:
        ylist = [(y + yshift_local) for y in ylist]

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

    # `statistics` is a dictionary containing some statistics about the
    # distribution ('median', 'mean', 'stddev')
    statistics = {}

    # support for histogram mode
    if options.histogram:
        # calculate the histogram distro before binning the data
        statistics['median'] = np.median(xlist)
        statistics['mean'] = np.mean(xlist)
        statistics['stddev'] = np.std(xlist)
        xlist, ylist = get_histogram(xlist,
                                     options.histogram_bins,
                                     options.histogram_type,
                                     options.histogram_nozeroes,
                                     options.histogram_sigma,
                                     options.debug)
    else:
        statistics['median'] = np.median(ylist)
        statistics['mean'] = np.mean(ylist)
        statistics['stddev'] = np.std(ylist)

    return xlist, ylist, statistics


def is_int(s):
    if isinstance(s, int):
        return True
    return (s[1:].isdigit() if s[0] in ('-', '+') else s.isdigit())


# convert axis format
def fmt_convert(item, fmt):
    if fmt == 'int':
        return int(float(item))
    elif fmt == 'float':
        if not item:
            return np.nan
        return float(item)
    elif fmt == 'unix':
        # convert unix timestamp to matplotlib datenum
        return md.date2num(datetime.datetime.fromtimestamp(float(item)))
    raise Exception('Error: invalid fmt (%s)' % fmt)


def parse_data_internal(raw_data, prefilter, sep, xcol, ycol,
                        sep2, xcol2, ycol2, xfmt, yfmt, header):
    # convert the raw data into lines
    column_names, lines = parse_csv(raw_data, sep, header)

    # pre-filter lines
    if prefilter:
        lines = filter_lines(lines, sep, prefilter, column_names)
        if not lines:
            raise Exception('Error: no data left after filtering')

    xlist = []
    ylist = []

    # get the column IDs
    if is_int(xcol):
        xcol = int(xcol)
    else:
        # look for named columns
        assert xcol in column_names, 'error: invalid xcol name: "%s"' % xcol
        xcol = column_names.index(xcol)
    if ycol is None:
        # used in histograms: value will be discarded
        ycol = 0
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
            xlist.append(fmt_convert(x, xfmt))
            ylist.append(fmt_convert(y, yfmt))
    return xlist, ylist


def create_graph_begin(options):
    # create figure
    figsize = tuple(float(v) for v in options.figsize)
    fig = plt.figure(figsize=figsize)
    # plt.gca().set_xlim([xstart, xstart + 1100000000])
    # plt.gca().set_ylim([0, 100])
    # # add a vertical line
    # plt.axvline(x=game_start)
    # # add a horizontal line
    # plt.axhline(y=1)
    ax1 = fig.add_subplot(111)
    ax1.set_title(options.title)
    ax1.set_xlabel(options.xlabel)
    if options.xfmt == 'unix':
        xfmt = md.DateFormatter(options.fmtdate)
        ax1.xaxis.set_major_formatter(xfmt)
    return ax1


def matplotlib_fmt_parse(fmt, in_color):
    # https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.plot.html
    # A format string consists of a part for color, marker and line:
    # fmt = '[marker][line][color]'
    # Each of them is optional. If not provided, the value from the style
    # cycle is used. Exception: If line is given, but no marker, the data
    # will be a line without markers.
    # Other combinations such as [color][marker][line] are also supported,
    # but note that their parsing may be ambiguous.
    marker, linestyle, color = None, None, None

    # let's start with color in '[color][marker][line]'
    # single letter (e.g. 'b'),
    if len(fmt) >= 1 and fmt[0] in VALID_MATPLOTLIB_COLORS:
        color = fmt[0]
        fmt = fmt[1:]
    # predefined colors (e.g. 'C0')
    if len(fmt) >= 2 and fmt[0] == 'C' and fmt[1].isdigit():
        color = fmt[:2]
        fmt = fmt[2:]

    # marker
    if len(fmt) >= 1 and fmt[0] in VALID_MATPLOTLIB_MARKERS:
        marker = fmt[0]
        fmt = fmt[1:]
    else:
        marker = None

    # linestyle
    for ls in VALID_MATPLOTLIB_LINESTYLES.keys():
        if fmt.startswith(ls):
            linestyle = VALID_MATPLOTLIB_LINESTYLES[ls]
            fmt = fmt[len(ls):]
            break
    else:
        linestyle = None

    # color
    if color is None:
        color = fmt
    if in_color is not None:
        color = in_color
    return marker, linestyle, color


# define a simple fitting function
def fit_function(x, a, b):
    return a * x + b


def create_graph_draw(ax, xlist, ylist, statistics, fmt, color, label,
                      options):
    marker, linestyle, color = matplotlib_fmt_parse(fmt, color)

    ax.plot(xlist, ylist, color=color,
            linestyle=linestyle, marker=marker,
            label=label)

    if options.debug > 1:
        print('ax.plot(%r, %r, \'%s\', color=%s, label=%r)' % (
            list(xlist), ylist, fmt, color, label))

    if options.xfmt == 'int':
        # make sure the ticks are all integers
        num_values = len(ax.get_xticks())
        step = int(math.ceil((int(math.ceil(ax.get_xticks()[-1])) -
                              int(math.floor(ax.get_xticks()[0]))) /
                             num_values))
        ax.set_xticks(range(int(ax.get_xticks()[0]),
                            int(ax.get_xticks()[-1]),
                            step))
    if options.histogram:
        if options.add_median:
            plt.axvline(statistics['median'], color=color, linestyle='dotted',
                        linewidth=1)
        if options.add_mean:
            plt.axvline(statistics['mean'], color=color, linestyle='dotted',
                        linewidth=1)
        if options.add_stddev:
            plt.axvline(statistics['mean'] + statistics['stddev'],
                        color=color, linestyle='dotted', linewidth=1)
            plt.axvline(statistics['mean'] - statistics['stddev'],
                        color=color, linestyle='dotted', linewidth=1)
    else:
        if options.add_median:
            plt.axhline(statistics['median'], color=color, linestyle='dotted',
                        linewidth=1)
        if options.add_mean:
            plt.axhline(statistics['mean'], color=color, linestyle='dotted',
                        linewidth=1)
        if options.add_stddev:
            plt.axhline(statistics['mean'] + statistics['stddev'],
                        color=color, linestyle='dotted', linewidth=1)
            plt.axhline(statistics['mean'] - statistics['stddev'],
                        color=color, linestyle='dotted', linewidth=1)
        if options.add_regression:
            # curve fit
            (a, b), _ = curve_fit(fit_function, xlist, ylist)
            print('fit curve: y = %.5f * x + %.5f' % (a, b))
            # define sequence of inputs
            x_line = np.arange(min(xlist), max(xlist), 1)
            y_line = fit_function(x_line, a, b)
            plt.plot(x_line, y_line, '--', color='red')


def create_graph_end(ax, ylabel, xlim, ylim, xscale, yscale):
    ax.set_ylabel(ylabel)

    # set xlim/ylim
    if xlim[0] != '-':
        ax.set_xlim(left=float(xlim[0]))
    if xlim[1] != '-':
        ax.set_xlim(right=float(xlim[1]))

    if ylim is not None:
        if ylim[0] != '-':
            ax.set_ylim(bottom=float(ylim[0]))
        if ylim[1] != '-':
            ax.set_ylim(top=float(ylim[1]))

    # set xscale/yscale
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)


def get_options(argv):
    # parse opts
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-v', '--version', action='store_true',
                        dest='version', default=False,
                        help='Print version',)
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
    parser.add_argument('--figsize', action='store', type=str, nargs=2,
                        dest='figsize', default=default_values['figsize'],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='set figsize to WIDTH x HEIGHT (default: %s)' % (
                            default_values['figsize'],))
    parser.add_argument('--title', action='store',
                        dest='title', default=default_values['title'],
                        metavar='PLOTTITLE',
                        help='use PLOTTITLE plot title',)
    parser.add_argument('--header', action='store_const', const=True,
                        dest='header', default=default_values['header'],
                        help='Read CSV header from first row (even if no #)',)
    parser.add_argument('--xcol', action='store',
                        dest='xcol', default=default_values['xcol'],
                        metavar='XCOL',
                        help='use XCOL x col',)
    parser.add_argument('--xcol2', action='store', type=int,
                        dest='xcol2', default=default_values['xcol2'],
                        metavar='XCOL2',
                        help='use XCOL2 for refining x col',)
    parser.add_argument('--ycol', action='append',
                        dest='ycol', default=default_values['ycol'],
                        metavar='YCOL',
                        help='use YCOL y col',)
    parser.add_argument('--ycol2', action='store', type=int,
                        dest='ycol2', default=default_values['ycol2'],
                        metavar='YCOL2',
                        help='use YCOL2 for refining y col',)
    parser.add_argument('--xfmt', action='store', type=str,
                        dest='xfmt', default=default_values['xfmt'],
                        choices=VALID_COLUMN_FMTS,
                        metavar='[%s]' % (' | '.join(VALID_COLUMN_FMTS,)),
                        help='use XFMT format for x column',)
    parser.add_argument('--yfmt', action='store', type=str,
                        dest='yfmt', default=default_values['yfmt'],
                        choices=VALID_COLUMN_FMTS,
                        metavar='[%s]' % (' | '.join(VALID_COLUMN_FMTS,)),
                        help='use YFMT format for y column',)
    parser.add_argument('--fmtdate', action='store', type=str,
                        dest='fmtdate', default=default_values['fmtdate'],
                        metavar='FMTDATE',
                        help='use FMTDATE for formatting unix dates',)
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
                        choices=VALID_LEGEND_LOCS,
                        metavar='[%s]' % (' | '.join(VALID_LEGEND_LOCS,)),
                        help='Legend location',)
    parser.add_argument('--histogram', action='store_const', const=True,
                        dest='histogram', default=default_values['histogram'],
                        help='sort and bin xlist, get ylist as histogram',)
    parser.add_argument('--histogram-bins', action='store', type=int,
                        dest='histogram_bins',
                        default=default_values['histogram_bins'],
                        metavar='NBINS',
                        help='use NBINS bins',)
    parser.add_argument('--histogram-nozeroes', action='store_const',
                        const=True,
                        dest='histogram_nozeroes',
                        default=default_values['histogram_nozeroes'],
                        help='remove zeroes on the histogram',)
    parser.add_argument('--histogram-sigma', action='store', type=float,
                        dest='histogram_sigma',
                        default=default_values['histogram_sigma'],
                        metavar='SIGMA',
                        help='use avg += (SIGMA * stddev) to remove outliers',)
    parser.add_argument('--histogram-type', action='store', type=str,
                        dest='histogram_type',
                        default=default_values['histogram_type'],
                        choices=VALID_HISTOGRAM_TYPES,
                        metavar='[%s]' % (' | '.join(VALID_HISTOGRAM_TYPES,)),
                        help='Histogram type',)
    parser.add_argument('--xlabel', action='store',
                        dest='xlabel', default=default_values['xlabel'],
                        metavar='XLABEL',
                        help='use XLABEL x label',)
    parser.add_argument('--ylabel', action='append',
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
    parser.add_argument('--add-regression', action='store_const',
                        dest='add_regression', const=True,
                        default=default_values['add_regression'],
                        help='Add a line at the linear regression',)
    parser.add_argument('--xlim', action='store', type=str, nargs=2,
                        dest='xlim', default=default_values['xlim'],
                        metavar=('left', 'right'),)
    parser.add_argument('--ylim', action='append', type=str, nargs=2,
                        dest='ylim', default=default_values['ylim'],
                        metavar=('bottom', 'top'),)
    scale_values_str = [str(item) for item in SCALE_VALUES]
    parser.add_argument('--xscale', action='store', type=str,
                        dest='xscale', default=default_values['xscale'],
                        choices=SCALE_VALUES,
                        metavar='[%s]' % (' | '.join(scale_values_str,)),
                             help='yscale values',)
    parser.add_argument('--yscale', action='store', type=str,
                        dest='yscale', default=default_values['yscale'],
                        choices=SCALE_VALUES,
                        metavar='[%s]' % (' | '.join(scale_values_str,)),
                             help='yscale values',)
    parser.add_argument('--twinx', action='count',
                        dest='twinx', default=0,
                        help='use twin y axes',)
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
                        help='use FMT format(s) for plotting',)
    parser.add_argument('--color', action='append',
                        dest='color', default=default_values['color'],
                        metavar='COLOR',
                        help='use COLOR color(s) for plotting',)
    parser.add_argument('--label', action='append',
                        dest='label', default=default_values['label'],
                        metavar='LABEL',
                        help='use LABEL label(s)',)
    parser.add_argument('-i', '--infile', action='append',
                        default=default_values['infile'],
                        metavar='input-file',
                        help='input file(s)',)
    # batch conf arguments
    parser.add_argument('--batch-infile', type=str,
                        default=default_values['batch_infile'],
                        metavar='batch_infile',
                        help='conf input file',)
    parser.add_argument('--batch-sep', action='store', type=str,
                        dest='batch_sep', default=default_values['batch_sep'],
                        metavar='SEP',
                        help='use SEP as separator in the batch file',)
    parser.add_argument('--batch-col', action='store',
                        dest='batch_col', default=default_values['batch_col'],
                        metavar='BATCHCOL',
                        help='use BATCHCOL batch col',)
    parser.add_argument('--batch-label-col', action='store',
                        dest='batch_label_col',
                        default=default_values['batch_label_col'],
                        metavar='BATCHLABELCOL',
                        help='use BATCHLABELCOL batch for label col',)
    parser.add_argument('--batch-filter', action='append', type=str, nargs=3,
                        dest='batch_filter',
                        default=default_values['batch_filter'],
                        metavar=('COL', 'OP', 'VAL'),
                        help='select only batch rows where COL OP VAL '
                        'is true',)
    # output
    parser.add_argument('outfile', type=str, nargs='?',
                        default=default_values['outfile'],
                        metavar='output-file',
                        help='output file',)
    # do the parsing
    options = parser.parse_args(argv[1:])
    if options.version:
        return options

    # check the filters
    for f in (options.filter, options.batch_filter):
        if f is None:
            continue
        for fcol, fop, fval in f:
            assert is_valid_op(fop), (
                'invalid op ("%s") in filter: "%s %s %s". Options: %r' % (
                    fop, fcol, fop, fval, VALID_OPS))
    # check there is an input file
    assert options.infile or options.batch_infile, (
       'error: must provide valid input file')
    # check twinx
    if options.twinx > 0:
        # count the number of lines before and after the twinx
        lines_before_twinx = argv[:argv.index('--twinx')].count('-i')
        lines_after_twinx = argv[argv.index('--twinx'):].count('-i')
        assert lines_before_twinx > 0, 'need at least 1 line before twinx'
        assert lines_after_twinx > 0, 'need at least 1 line after twinx'
        options.twinx = lines_before_twinx
    # check positional parameters
    options.color = build_positional_parameter(argv, '--color')

    return options


def build_positional_parameter(argv, parid):
    parameter = []
    # note that every parameter must follow a '-i' (or '--infile')
    init = True
    prev_i = False
    for i, par in enumerate(argv):
        if par not in (parid, '-i', '--infile'):
            # ignore other parameters
            continue
        assert not ((par == parid) and init), 'cannot have %s before "-i"'
        init = False
        if par in ('-i', '--infile'):
            if prev_i:
                # hanging '-i'
                parameter.append(None)
            else:
                prev_i = True
        else:  # par == parid
            parameter.append(argv[i+1])
            prev_i = False
    if prev_i:
        # hanging '-i'
        parameter.append(None)
    return parameter


def batch_process_file(infile, sep, col, f):
    flist = batch_process_data(read_file(infile), sep, col, f, header=False)
    # make sure to include infile path
    dirname = os.path.dirname(infile)
    if dirname:
        flist = [os.path.join(dirname, f) for f in flist]
    return flist


def batch_parse_line(line, sep, xcol):
    if not line:
        # empty line
        return None, None

    # get x component
    x = get_column(line, sep, xcol, None, None)

    return x


def batch_process_data(raw_data, sep, col, f, header):
    sep = sep if sep != '' else None

    # convert the raw data into lines
    column_names, lines = parse_csv(raw_data, sep, header)

    # pre-filter lines
    if f:
        lines = filter_lines(lines, sep, f, column_names)

    flist = []

    # get the column IDs
    if is_int(col):
        col = int(col)
    else:
        # look for named columns
        assert col in column_names, 'error: invalid col name: "%s"' % col
        col = column_names.index(col)

    # parse all the lines
    for line in lines:
        f = batch_parse_line(line, sep, col)
        if f is not None:
            # append values
            flist.append(f)

    return flist


def get_line_info(index, infile, options, batch_label_list):
    # 1. parameters that keep the last one if not enough
    # ycol
    if len(options.ycol) == 0:
        # no ycol
        if options.histogram:
            ycol = options.xcol
        else:
            raise Exception('Error: need a ycol value')
    elif index < len(options.ycol):
        ycol = options.ycol[index]
    else:
        ycol = options.ycol[-1]
    # look for named columns
    if is_int(ycol):
        ycol = int(ycol)

    # 2. parameters that use a default if not enough
    # shifts
    xshift = None
    if index < len(options.xshift):
        xshift = float(options.xshift[index])
        print('shifting x by %f' % xshift)
    yshift = None
    if index < len(options.yshift):
        yshift = float(options.yshift[index])
        print('shifting y by %f' % yshift)

    # 3. parameters that are derived automatically if not enough
    # label
    if index < len(options.label):
        label = options.label[index]
        if label.lower() == 'none':
            label = None
    elif index < len(batch_label_list):
        label = batch_label_list[index]
    else:
        label = os.path.basename(infile) if infile != '/dev/fd/0' else 'stdin'

    # fmt
    default_fmt_list = ['C%i%s' % (i % 10, options.marker) for i in
                        range(MAX_INFILE_LIST_LEN)]
    if index < len(options.fmt):
        fmt = options.fmt[index]
    else:
        fmt = default_fmt_list[index]

    # color
    color = options.color[index]

    return ycol, xshift, yshift, label, fmt, color


def main(argv):
    # parse options
    options = get_options(argv)
    if options.version:
        print('version: %s' % __version__)
        sys.exit(0)

    # 1. get all the per-line info into xy_data
    # 1.1. get infile(s)/outfile
    batch_label_list = []
    if options.batch_infile is not None:
        infile_list = batch_process_file(
            options.batch_infile, options.batch_sep, options.batch_col,
            options.batch_filter, options.header)
        batch_label_list = batch_process_data(
            read_file(options.batch_infile), options.batch_sep,
            options.batch_label_col, options.batch_filter)
    else:
        infile_list = [('/dev/fd/0' if name == '-' else name) for name in
                       options.infile]
    if options.outfile == '-':
        options.outfile = '/dev/fd/1'

    if options.debug > 0:
        print(options)

    # 1.2. get all the per-line info into xy_data
    # Includes `ycol`, `xlist` (x-axis values), `ylist` (y-axis values),
    # `statistics`, `label`, `fmt`, `color`.
    xy_data = []
    for index, infile in enumerate(infile_list):
        # get all the info from the current line
        ycol, xshift, yshift, label, fmt, color = (
                get_line_info(index, infile, options, batch_label_list))
        xlist, ylist, statistics = parse_data(
                read_file(infile), ycol, xshift, yshift, options)
        xy_data.append([xlist, ylist, statistics, label, fmt, color])

    # 2. get all the per-axes info
    # create the axes
    ax = []
    ax.append(create_graph_begin(options))
    if options.twinx > 0:
        ax.append(ax[0].twinx())

    axinfo = []
    for axid in range(len(ax)):
        ylabel = options.ylabel[axid] if axid < len(options.ylabel) else None
        ylim = options.ylim[axid] if axid < len(options.ylim) else None
        axinfo.append([ylabel, ylim])

    # 3. create the graph
    # add each of the lines in xy_data
    axid = 0
    for index, (xlist, ylist, statistics, label, fmt, color
                ) in enumerate(xy_data):
        if options.twinx > 0 and index == options.twinx:
            axid += 1
        create_graph_draw(ax[axid], xlist, ylist, statistics, fmt, color,
                          label, options)

    # set final graph details
    for axid, (ylabel, ylim) in enumerate(axinfo):
        # set the values
        create_graph_end(ax[axid], ylabel, options.xlim,
                         ylim, options.xscale, options.yscale)

    # set common legend
    if options.legend_loc != 'none':
        # https://stackoverflow.com/a/14344146
        lin_list = []
        leg_list = []
        for axid in range(len(ax)):
            lin, leg = ax[axid].get_legend_handles_labels()
            lin_list += lin
            leg_list += leg
        _ = ax[0].legend(lin_list, leg_list, loc=options.legend_loc)

    # save graph
    if options.debug > 0:
        print('output is %s' % options.outfile)
    plt.savefig('%s' % options.outfile)


if __name__ == '__main__':
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
