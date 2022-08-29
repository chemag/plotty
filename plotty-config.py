#!/usr/bin/env python3

"""plot-config.py: plotty config management.
"""

import argparse
import importlib
import matplotlib


prefilter_lib = importlib.import_module('prefilter')


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
    ' ': 'None',
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
# * 1. single: same value for every line (e.g. --xfmt, --xcol, --ycol)
# * 2. per-axis: need a different value per axis (e.g. --ylim, --ylim2,
#   --ylabel, --ylabel2)
#   We use --twinx to create multiple axis
# * 3. per-line: need a different value per line (e.g. --i, --fmt, --label,
#   --prefilter))
#   We attach lines to axis based on the exact parameter location.
#   * subtypes depending on what to do if there are not as many occurrences
#     as '-i' elements.
#     * 3.1. keep the last one if not enough
#     * 3.2. use default value if not enough
#     * 3.3. positional parameters: attach to the previous '-i'
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
#   * prefilter [v]
#   * infile [v]
# * used in create_graph_begin()
#   * ylabel [v]
#   * ylabel2 [v]
# * used in create_graph_draw()
#   * xfmt [ ]
#   * histogram [ ]
#   * use_median [ ]
#   * use_mean [ ]
#   * use_stddev [ ]
#   * use_regression [ ]
#   * use_moving_average [ ]
#   * use_ewma [ ]
# * used in create_graph_end()
#   * legend_loc [v]
#   * xlim [ ]
#   * ylim [v]
#   * ylim2 [v]
#   * xscale [ ]
#   * yscale [ ]

#
default_values = {
    'debug': 0,
    'dry_run': False,
    'marker': '.',
    'figsize': matplotlib.rcParams['figure.figsize'],
    'title': '--title',
    'header': False,
    'xcol': 0,
    'xcol2': None,
    'ycol': 0,
    'ycol2': None,
    'xfmt': 'float',
    'yfmt': 'float',
    'fmtdate': '%Y-%m-%d\n%H:%M:%S',
    'ydelta': False,
    'ycumulative': False,
    # use '' to separate using None (any number of spaces/tabs)
    'sep': ',',
    'sep2': ':',
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
    'ylabel': '--ylabel',
    'ylabel2': '--ylabel2',
    'xlim': ['-', '-'],
    'ylim': ['-', '-'],
    'ylim2': ['-', '-'],
    'xscale': None,
    'yscale': None,
    'xfactor': None,
    'yfactor': None,
    'use_mean': False,
    'use_median': False,
    'use_stddev': False,
    'use_regression': False,
    'use_moving_average': None,
    'use_ewma': None,
    # per-line parameters
    'xshift': [],
    'yshift': [],
    'label': [],
    'prefilter': [],
    'fmt': [],
    'color': [],
    'infile': [],
    # batch conf parameters
    'batch_infile': None,
    'batch_sep': ',',
    'batch_col': None,
    'batch_label_col': None,
    'batch_prefilter': None,
    # output parameter
    'outfile': None,
}


def get_options(argv):
    # parse opts
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-v', '--version', action='store_true',
        dest='version', default=False,
        help='Print version',)
    parser.add_argument(
        '-d', '--debug', action='count',
        dest='debug', default=default_values['debug'],
        help='Increase verbosity (multiple times for more)',)
    parser.add_argument(
        '--quiet', action='store_const',
        dest='debug', const=-1,
        help='Zero verbosity',)
    parser.add_argument(
        '--dry-run', action='store_true',
        dest='dry_run', default=default_values['dry_run'],
        help='Dry run',)
    parser.add_argument(
        '--marker', action='store',
        dest='marker', default=default_values['marker'],
        metavar='MARKER',
        help='use MARKER as plot marker',)
    parser.add_argument(
        '--figsize', action='store', type=str, nargs=2,
        dest='figsize', default=default_values['figsize'],
        metavar=('WIDTH', 'HEIGHT'),
        help='set figsize to WIDTH x HEIGHT (default: %s)' % (
            default_values['figsize'],))
    parser.add_argument(
        '--title', action='store',
        dest='title', default=default_values['title'],
        metavar='PLOTTITLE',
        help='use PLOTTITLE plot title',)
    parser.add_argument(
        '--header', action='store_const', const=True,
        dest='header', default=default_values['header'],
        help='Read CSV header from first row (even if no #)',)
    parser.add_argument(
        '--xcol', action='store',
        dest='xcol', default=default_values['xcol'],
        metavar='XCOL',
        help='use XCOL x col',)
    parser.add_argument(
        '--xcol2', action='store', type=int,
        dest='xcol2', default=default_values['xcol2'],
        metavar='XCOL2',
        help='use XCOL2 for refining x col',)
    parser.add_argument(
        '--ycol', action='store',
        dest='ycol', default=default_values['ycol'],
        metavar='YCOL',
        help='use YCOL y col',)
    parser.add_argument(
        '--ycol2', action='store', type=int,
        dest='ycol2', default=default_values['ycol2'],
        metavar='YCOL2',
        help='use YCOL2 for refining y col',)
    parser.add_argument(
        '--xfmt', action='store', type=str,
        dest='xfmt', default=default_values['xfmt'],
        choices=VALID_COLUMN_FMTS,
        metavar='[%s]' % (' | '.join(VALID_COLUMN_FMTS,)),
        help='use XFMT format for x column',)
    parser.add_argument(
        '--yfmt', action='store', type=str,
        dest='yfmt', default=default_values['yfmt'],
        choices=VALID_COLUMN_FMTS,
        metavar='[%s]' % (' | '.join(VALID_COLUMN_FMTS,)),
        help='use YFMT format for y column',)
    parser.add_argument(
        '--fmtdate', action='store', type=str,
        dest='fmtdate', default=default_values['fmtdate'],
        metavar='FMTDATE',
        help='use FMTDATE for formatting unix dates',)
    parser.add_argument(
        '--ydelta', action='store_const', const=True,
        dest='ydelta', default=default_values['ydelta'],
        help='use $y[k] = (y[k] - y[k-1])$',)
    parser.add_argument(
        '--ycumulative', action='store_const', const=True,
        dest='ycumulative',
        default=default_values['ycumulative'],
        help='use $y[k] = \\sum_i=0^k y[i]$',)
    parser.add_argument(
        '--sep', action='store', type=str,
        dest='sep', default=default_values['sep'],
        metavar='SEP',
        help='use SEP as separator',)
    parser.add_argument(
        '--sep2', action='store', type=str,
        dest='sep2', default=default_values['sep2'],
        metavar='SEP2',
        help='use SEP2 as alternate separator',)
    parser.add_argument(
        '--legend-loc', action='store', type=str,
        dest='legend_loc',
        default=default_values['legend_loc'],
        choices=VALID_LEGEND_LOCS,
        metavar='[%s]' % (' | '.join(VALID_LEGEND_LOCS,)),
        help='Legend location',)
    parser.add_argument(
        '--histogram', action='store_const', const=True,
        dest='histogram', default=default_values['histogram'],
        help='sort and bin xlist, get ylist as histogram',)
    parser.add_argument(
        '--histogram-bins', action='store', type=int,
        dest='histogram_bins',
        default=default_values['histogram_bins'],
        metavar='NBINS',
        help='use NBINS bins',)
    parser.add_argument(
        '--histogram-nozeroes', action='store_const',
        const=True,
        dest='histogram_nozeroes',
        default=default_values['histogram_nozeroes'],
        help='remove zeroes on the histogram',)
    parser.add_argument(
        '--histogram-sigma', action='store', type=float,
        dest='histogram_sigma',
        default=default_values['histogram_sigma'],
        metavar='SIGMA',
        help='use avg += (SIGMA * stddev) to remove outliers',)
    parser.add_argument(
        '--histogram-type', action='store', type=str,
        dest='histogram_type',
        default=default_values['histogram_type'],
        choices=VALID_HISTOGRAM_TYPES,
        metavar='[%s]' % (' | '.join(VALID_HISTOGRAM_TYPES,)),
        help='Histogram type',)
    parser.add_argument(
        '--xlabel', action='store',
        dest='xlabel', default=default_values['xlabel'],
        metavar='XLABEL',
        help='use XLABEL x label',)
    parser.add_argument(
        '--ylabel', action='store',
        dest='ylabel', default=default_values['ylabel'],
        metavar='YLABEL',
        help='use YLABEL y label',)
    parser.add_argument(
        '--ylabel2', action='store',
        dest='ylabel2', default=default_values['ylabel2'],
        metavar='YLABEL2',
        help='use YLABEL2 y label 2 (right axis)',)
    parser.add_argument(
        '--use-mean', action='store_const',
        dest='use_mean', const=True,
        default=default_values['use_mean'],
        help='Use a line at the mean',)
    parser.add_argument(
        '--use-median', action='store_const',
        dest='use_median', const=True,
        default=default_values['use_median'],
        help='Use a line at the median',)
    parser.add_argument(
        '--use-stddev', action='store_const',
        dest='use_stddev', const=True,
        default=default_values['use_stddev'],
        help='Use 2 lines at mean +- stddev',)
    parser.add_argument(
        '--use-regression', action='store_const',
        dest='use_regression', const=True,
        default=default_values['use_regression'],
        help='Use a line at the linear regression',)
    parser.add_argument(
        '--use-moving-average', action='store', type=int,
        dest='use_moving_average',
        default=default_values['use_moving_average'],
        help='Use a line at the moving average',)
    parser.add_argument(
        '--use-ewma', action='store', type=float,
        dest='use_ewma',
        default=default_values['use_ewma'],
        help='Use a line at the ewma',)
    parser.add_argument(
        '--xlim', action='store', type=str, nargs=2,
        dest='xlim', default=default_values['xlim'],
        metavar=('left', 'right'),)
    parser.add_argument(
        '--ylim', action='store', type=str, nargs=2,
        dest='ylim', default=default_values['ylim'],
        metavar=('bottom', 'top'),)
    parser.add_argument(
        '--ylim2', action='store', type=str, nargs=2,
        dest='ylim2', default=default_values['ylim2'],
        metavar=('bottom', 'top'),)
    scale_values_str = [str(item) for item in SCALE_VALUES]
    parser.add_argument(
        '--xscale', action='store', type=str,
        dest='xscale', default=default_values['xscale'],
        choices=SCALE_VALUES,
        metavar='[%s]' % (' | '.join(scale_values_str,)),
        help='yscale values',)
    parser.add_argument(
        '--xfactor', action='store', type=float,
        dest='xfactor', default=default_values['xfactor'],
        metavar='XFACTOR',
        help='use XFACTOR factor for the x-axis',)
    parser.add_argument(
        '--yfactor', action='store', type=float,
        dest='yfactor', default=default_values['yfactor'],
        metavar='YFACTOR',
        help='use YFACTOR factor for the y-axis',)
    parser.add_argument(
        '--yscale', action='store', type=str,
        dest='yscale', default=default_values['yscale'],
        choices=SCALE_VALUES,
        metavar='[%s]' % (' | '.join(scale_values_str,)),
        help='yscale values',)
    parser.add_argument(
        '--twinx', action='count',
        dest='twinx', default=0,
        help='use twin y axes',)
    # per-line arguments
    parser.add_argument(
        '--xshift', action='append',
        dest='xshift', default=default_values['xshift'],
        metavar='XSHIFT',
        help='use XSHIFT x shift(s)',)
    parser.add_argument(
        '--yshift', action='append',
        dest='yshift', default=default_values['yshift'],
        metavar='YSHIFT',
        help='use YSHIFT y shift(s)',)
    parser.add_argument(
        '--fmt', action='append',
        dest='fmt', default=default_values['fmt'],
        metavar='FMT',
        help='use FMT format(s) for plotting ([marker][line][color])',)
    parser.add_argument(
        '--color', action='append',
        dest='color', default=default_values['color'],
        metavar='COLOR',
        help='use COLOR color(s) for plotting',)
    parser.add_argument(
        '--label', action='append',
        dest='label', default=default_values['label'],
        metavar='LABEL',
        help='use LABEL label(s)',)
    parser.add_argument(
        '--prefilter', action='append',
        dest='prefilter', default=default_values['prefilter'],
        metavar='PREFILTER-SPEC',
        help='select only rows where PREFILTER-SPEC is true',)
    parser.add_argument(
        '-i', '--infile', action='append',
        default=default_values['infile'],
        metavar='input-file',
        help='input file(s)',)
    # batch conf arguments
    parser.add_argument(
        '--batch-infile', type=str,
        default=default_values['batch_infile'],
        metavar='batch_infile',
        help='conf input file',)
    parser.add_argument(
        '--batch-sep', action='store', type=str,
        dest='batch_sep', default=default_values['batch_sep'],
        metavar='SEP',
        help='use SEP as separator in the batch file',)
    parser.add_argument(
        '--batch-col', action='store',
        dest='batch_col', default=default_values['batch_col'],
        metavar='BATCHCOL',
        help='use BATCHCOL batch col',)
    parser.add_argument(
        '--batch-label-col', action='store',
        dest='batch_label_col',
        default=default_values['batch_label_col'],
        metavar='BATCHLABELCOL',
        help='use BATCHLABELCOL batch for label col',)
    parser.add_argument(
        '--batch-prefilter', action='append', type=str, nargs=3,
        dest='batch_prefilter',
        default=default_values['batch_prefilter'],
        metavar=('COL', 'OP', 'VAL'),
        help='select only batch rows where COL OP VAL '
        'is true',)
    # output
    parser.add_argument(
        'outfile', type=str, nargs='?',
        default=default_values['outfile'],
        metavar='output-file',
        help='output file',)
    # do the parsing
    options = parser.parse_args(argv[1:])
    if options.version:
        return options
    # check the filters
    if options.prefilter is not None:
        for prefilter in options.prefilter:
            prefilter_lib.Prefilter(prefilter)
    if options.batch_prefilter is not None:
        for prefilter in options.batch_prefilter:
            prefilter_lib.Prefilter(prefilter)
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
