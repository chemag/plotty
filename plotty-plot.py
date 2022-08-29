#!/usr/bin/env python3

"""plot.py: simple data plotter.

# http://stackoverflow.com/a/11249340

# runme
# $ echo -e "1,2\n2,2\n3,5\n4,7\n5,1\n" | ./plotty-plot.py -i - /tmp/plot.png
"""

import datetime
import importlib
import math
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import os
import scipy.optimize
import sys

config_lib = importlib.import_module('plotty-config')
prefilter_lib = importlib.import_module('prefilter')
utils = importlib.import_module('utils')


__version__ = '0.2'

MAX_INFILE_LIST_LEN = 20


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


def filter_lines(lines, sep, prefilter):
    new_lines = []
    for line in lines:
        if not line:
            continue
        if filter_line(line, sep, prefilter):
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


def parse_line(line, i, sep, xcol, ycol, sep2, xcol2, ycol2, xfactor, yfactor):
    # get x component
    x = i if xcol == -1 else get_column(line, sep, xcol, sep2, xcol2)
    x = x if xfactor is None else float(x) * xfactor

    # get y component
    y = i if ycol == -1 else get_column(line, sep, ycol, sep2, ycol2)
    y = y if yfactor is None else float(y) * yfactor

    return x, y


def get_data(raw_data, ycol, xshift_local, yshift_local, prefilter, options):
    prefilter = prefilter_lib.Prefilter(prefilter)
    sep = options.sep if options.sep != '' else None
    xcol = options.xcol
    xcol2 = options.xcol2
    sep2 = options.sep2 if options.sep2 != '' else None
    ycol2 = options.ycol2
    xfmt = options.xfmt
    yfmt = options.yfmt

    # 1. convert the raw data into lines
    column_names, lines = parse_csv(raw_data, sep, options.header)

    # 2. fix column names in the prefilter
    prefilter.fix_columns(column_names)

    # 3. get column ids
    xcol, ycol = get_column_ids(xcol, ycol, column_names)

    # 4. parse all the lines into (xlist, ylist)
    xlist = []
    ylist = []
    for i, line in enumerate(lines):
        if not line:
            # empty line
            continue
        # prefilter lines
        if not prefilter.match_line(line, sep):
            continue
        x, y = parse_line(line, i, sep, xcol, ycol, sep2, xcol2, ycol2,
                          options.xfactor, options.yfactor)
        if x is not None and y is not None:
            # append values
            xlist.append(fmt_convert(x, xfmt))
            ylist.append(fmt_convert(y, yfmt))

    # 5. postfilter values
    # 5.1. support for shift modes
    if xshift_local is not None:
        xlist = [(x + xshift_local) for x in xlist]
    if yshift_local is not None:
        ylist = [(y + yshift_local) for y in ylist]

    # 5.2. support for ydelta (plotting `y[k] - y[k-1]` instead of `y[k]`)
    if options.ydelta:
        ylist = [y1 - y0 for y0, y1 in zip([ylist[0]] + ylist[:-1], ylist)]

    # 5.3. support for ycumulative (plotting `\Sum y[k]` instead of `y[k]`)
    if options.ycumulative:
        new_ylist = []
        for y in ylist:
            prev_y = new_ylist[-1] if new_ylist else 0
            new_ylist.append(y + prev_y)
        ylist = new_ylist

    # 5.4. support for replacements
    if options.use_median:
        median = np.median(ylist)
        ylist = [median for _ in ylist]
    if options.use_mean:
        mean = np.mean(ylist)
        ylist = [mean for _ in ylist]
    if options.use_stddev:
        stddev = np.std(ylist)
        ylist = [stddev for _ in ylist]
    if options.use_regression:
        # curve fit (linear regression)
        (a, b), _ = scipy.optimize.curve_fit(fit_function, xlist, ylist)
        ylist = [fit_function(x, a, b) for x in xlist]
    if options.use_moving_average is not None:
        # Moving Average fit (linear regression)
        ylist = get_moving_average(ylist, options.use_moving_average)
    if options.use_ewma is not None:
        # Exponentially-Weighted Moving Average
        ylist = get_ewma(ylist, options.use_ewma)

    # 5.5. support for histograms
    if options.histogram:
        xlist, ylist = get_histogram(xlist,
                                     options.histogram_bins,
                                     options.histogram_type,
                                     options.histogram_nozeroes,
                                     options.histogram_sigma,
                                     options.debug)

    return xlist, ylist


def get_moving_average(ylist, n):
    new_ylist = []
    for i in range(len(ylist)):
        max_index = min(i, len(ylist) - 1)
        min_index = max(0, i - n + 1)
        used_values = ylist[min_index:max_index + 1]
        new_val = sum(used_values) / len(used_values)
        new_ylist.append(new_val)
    return new_ylist


def get_ewma(ylist, alpha):
    new_ylist = [ylist[0]]
    for y in ylist[1:]:
        new_val = alpha * y + (1.0 - alpha) * new_ylist[-1]
        new_ylist.append(new_val)
    return new_ylist


# convert axis format
def fmt_convert(item, fmt):
    if fmt == 'int':
        return int(float(item))
    elif fmt == 'float':
        if item is None:
            return np.nan
        return float(item)
    elif fmt == 'unix':
        # convert unix timestamp to matplotlib datenum
        return md.date2num(datetime.datetime.fromtimestamp(float(item)))
    raise Exception('Error: invalid fmt (%s)' % fmt)


def get_column_ids(xcol, ycol, column_names):
    # get the column IDs
    if utils.is_int(xcol):
        xcol = int(xcol)
    else:
        # look for named columns
        assert xcol in column_names, 'error: invalid xcol name: "%s"' % xcol
        xcol = column_names.index(xcol)
    if ycol is None:
        # used in histograms: value will be discarded
        ycol = 0
    if utils.is_int(ycol):
        ycol = int(ycol)
    else:
        # look for named columns
        assert ycol in column_names, 'error: invalid ycol name: "%s"' % ycol
        ycol = column_names.index(ycol)

    return xcol, ycol


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
    if len(fmt) >= 1 and fmt[0] in config_lib.VALID_MATPLOTLIB_COLORS:
        color = fmt[0]
        fmt = fmt[1:]
    # predefined colors (e.g. 'C0')
    if len(fmt) >= 2 and fmt[0] == 'C' and fmt[1].isdigit():
        color = fmt[:2]
        fmt = fmt[2:]

    # marker
    if len(fmt) >= 1 and fmt[0] in config_lib.VALID_MATPLOTLIB_MARKERS:
        marker = fmt[0]
        fmt = fmt[1:]
    else:
        marker = None

    # linestyle
    for ls in config_lib.VALID_MATPLOTLIB_LINESTYLES.keys():
        if fmt.startswith(ls):
            linestyle = config_lib.VALID_MATPLOTLIB_LINESTYLES[ls]
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


def create_graph_draw(ax, xlist, ylist, fmt, color, label, options):
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


def create_graph_end(ax, ylabel, xlim, ylim, xscale, yscale):
    ax.set_ylabel(ylabel)

    # set xlim/ylim
    if xlim[0] != '-':
        ax.set_xlim(left=float(xlim[0]))
    if xlim[1] != '-':
        ax.set_xlim(right=float(xlim[1]))

    if ylim[0] != '-':
        ax.set_ylim(bottom=float(ylim[0]))
    if ylim[1] != '-':
        ax.set_ylim(top=float(ylim[1]))

    # set xscale/yscale
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)


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

    # prefilter lines
    if f:
        lines = filter_lines(lines, sep, f)

    flist = []

    # get the column IDs
    if utils.is_int(col):
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
    # 1.1. ycol
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
    if utils.is_int(ycol):
        ycol = int(ycol)

    # 2. parameters that use a default if not enough
    # 2.1. shifts
    xshift = None
    if index < len(options.xshift):
        xshift = float(options.xshift[index])
        print('shifting x by %f' % xshift)
    yshift = None
    if index < len(options.yshift):
        yshift = float(options.yshift[index])
        print('shifting y by %f' % yshift)

    # 2.2. prefilter
    prefilter = None
    if index < len(options.prefilter):
        prefilter = options.prefilter[index]
        print('filtering input by %r' % prefilter)

    # 3. parameters that are derived automatically if not enough
    # 3.1. label
    if index < len(options.label):
        label = options.label[index]
        if label.lower() == 'none':
            label = None
    elif index < len(batch_label_list):
        label = batch_label_list[index]
    else:
        label = os.path.basename(infile) if infile != '/dev/fd/0' else 'stdin'

    # 3.2. fmt
    default_fmt_list = ['C%i%s' % (i % 10, options.marker) for i in
                        range(MAX_INFILE_LIST_LEN)]
    if index < len(options.fmt):
        fmt = options.fmt[index]
    else:
        fmt = default_fmt_list[index]

    # 3.3. color
    color = options.color[index]

    return ycol, xshift, yshift, label, fmt, color, prefilter


def main(argv):
    # parse options
    options = config_lib.get_options(argv)
    if options.version:
        print('version: %s' % __version__)
        sys.exit(0)

    # 1. get all the per-line info into xy_data
    # 1.1. get infile(s)/outfile
    batch_label_list = []
    if options.batch_infile is not None:
        infile_list = batch_process_file(
            options.batch_infile, options.batch_sep, options.batch_col,
            options.batch_prefilter, options.header)
        batch_label_list = batch_process_data(
            read_file(options.batch_infile), options.batch_sep,
            options.batch_label_col, options.batch_prefilter)
    else:
        infile_list = [('/dev/fd/0' if name == '-' else name) for name in
                       options.infile]
    if options.outfile == '-':
        options.outfile = '/dev/fd/1'

    if options.debug > 0:
        print(options)

    # 1.2. get all the per-line info into xy_data
    # Includes `ycol`, `xlist` (x-axis values), `ylist` (y-axis values),
    # `label`, `fmt`, `color`, and `prefilter`
    xy_data = []
    for index, infile in enumerate(infile_list):
        # get all the info from the current line
        ycol, xshift, yshift, label, fmt, color, prefilter = (
            get_line_info(index, infile, options, batch_label_list))
        xlist, ylist = get_data(
            read_file(infile), ycol, xshift, yshift, prefilter, options)
        xy_data.append([xlist, ylist, label, fmt, color])

    if options.dry_run:
        return xy_data

    # 2. get all the per-axes info
    # create the main axis
    ax = []
    axinfo = []
    ax.append(create_graph_begin(options))
    ylabel = options.ylabel
    ylim = options.ylim
    axinfo.append([ylabel, ylim])
    # create the twin axis
    if options.twinx > 0:
        ax.append(ax[0].twinx())
        ylabel = options.ylabel2
        ylim = options.ylim2
        axinfo.append([ylabel, ylim])

    # 3. create the graph
    # add each of the lines in xy_data
    axid = 0
    for index, (xlist, ylist, label, fmt, color) in enumerate(xy_data):
        if options.twinx > 0 and index == options.twinx:
            axid += 1
        create_graph_draw(ax[axid], xlist, ylist, fmt, color, label, options)

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
