#!/usr/bin/env python3

"""plot-config.py: plotty config management.
"""

import argparse
import importlib
import matplotlib
import os
import sys

from google.protobuf import text_format

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
SCRIPT_ROOT_DIR = os.path.join(SCRIPT_DIR, '.')
sys.path.append(SCRIPT_ROOT_DIR)
import proto.plotty_pb2  # noqa: E402

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


MAX_INFILE_LIST_LEN = 20
DEFAULT_FMT_LIST = ['C%i%s' % (i % 10, '.') for i in
                    range(MAX_INFILE_LIST_LEN)]

# parameter notes
# There are 3 type of parameters
# * 1. single: same value for every line (e.g. --xfmt)
# * 2. per-axis: need a different value per axis (e.g. --ylim, --ylim2,
#   --ylabel, --ylabel2, --xcol, --ycol)
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
    'xfmt': 'float',
    'yfmt': 'float',
    'fmtdate': '%Y-%m-%d\n%H:%M:%S',
    'ydelta': False,
    'ycumulative': False,
    # use '' to separate using None (any number of spaces/tabs)
    'sep': ',',
    'sep2': '',
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
    'xcol': [],
    'xcol2': [],
    'ycol': [],
    'ycol2': [],
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
    # config-based parameters
    'inplot': [],
    'lineid': [],
    # output parameter
    'outfile': None,
}


def plotty_config_read(infile):
    print(f'read plotty config: {infile}')
    plots_pb = proto.plotty_pb2.Plots()
    with open(infile, 'rb') as fd:
        text_format.Merge(fd.read(), plots_pb)
    return plots_pb


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
        '--figsize', action='store', type=float, nargs=2,
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
        '--xcol', action='append',
        dest='xcol', default=default_values['xcol'],
        metavar='XCOL',
        help='use XCOL x col',)
    parser.add_argument(
        '--ycol', action='append',
        dest='ycol', default=default_values['ycol'],
        metavar='YCOL',
        help='use YCOL y col',)
    parser.add_argument(
        '--xcol2', action='append',
        dest='xcol2', default=default_values['xcol2'],
        metavar='XCOL2',
        help='use XCOL2 for refining x col',)
    parser.add_argument(
        '--ycol2', action='append',
        dest='ycol2', default=default_values['ycol2'],
        metavar='YCOL2',
        help='use YCOL2 for refining y col',)
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
    # config-based parameters
    parser.add_argument(
        '--inplot', action='append',
        default=default_values['inplot'],
        metavar='input-plot-file',
        help='input file(s)',)
    parser.add_argument(
        '--lineid', action='append',
        default=default_values['lineid'],
        metavar='line id',
        help='line id(s) (<plotid>/<lineid>)',)

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
    # check there is (at least) an input or inplot file
    assert options.infile or options.batch_infile or (
        options.inplot and options.lineid), (
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

    # line management
    options.num_lines = len(options.infile)
    options.cur_line = 0

    gen_options, plot_line_list = convert_namespace_to_config(options)

    # config-based parameters
    if options.inplot and options.lineid:
        plot_line_list += plots_pb_read(options.inplot, options.lineid)

    return gen_options, plot_line_list


def plots_pb_read(inplot, lineid):
    plot_line_list = []
    plots_list = []
    # read the inplots
    for inplot_file in inplot:
        plots_list.append([inplot_file, plotty_config_read(inplot_file)])
    # make sure the lines exist
    for plot_line_id in lineid:
        plot_id, line_id = plot_line_id.split('/')
        # search for the line in plots_list
        line_found = False
        for inplot_file, plots_config in plots_list:
            for plot_pb in plots_config.plot:
                plot_pb = plot_pb_add_defaults(plot_pb)
                if plot_pb.id == plot_id:
                    # found plot_id
                    line_pb = get_line_pb(plot_pb, line_id)
                    if line_pb is not None:
                        print(f'found {plot_line_id} in plot '
                              f'{plot_pb.id} and line {line_pb.id} at '
                              f'{inplot_file}')
                        plot_line_list.append([plot_pb, plot_line_id])
                        line_found = True
                        break
        assert line_found, (f'cannot find {plot_line_id} in {inplot}')
    return plot_line_list


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


DEFAULT_PLOT_PB_PARAMETER_LIST = [
    'marker',
    # 'figsize',
    'title',
    'legend_loc',
    'xlabel',
    'ylabel',
    'ylabel2',
    # 'xlim',
    # 'ylim',
    # 'ylim2',
    # 'xscale',
    # 'yscale',
    # 'xfmt',
    # 'yfmt',
    'fmtdate',
]

DEFAULT_LINE_PB_PARAMETER_LIST = [
    'header',
    'xcol',
    # 'xcol2',
    'ycol',
    # 'ycol2',
    'sep',
    'sep2',
]


def plot_pb_add_defaults(plot_pb):
    for par in DEFAULT_PLOT_PB_PARAMETER_LIST:
        if not plot_pb.HasField(par):
            plot_pb.__setattr__(par, default_values[par])
    # 'figsize'
    par = 'figsize'
    if not plot_pb.HasField(par):
        plot_pb.figsize.__setattr__('width', default_values[par][0])
        plot_pb.figsize.__setattr__('height', default_values[par][1])
    # 'xlim'
    par = 'xlim'
    if not plot_pb.HasField(par):
        plot_pb.xlim.__setattr__('min', default_values[par][0])
        plot_pb.xlim.__setattr__('max', default_values[par][1])
    # 'ylim'
    par = 'ylim'
    if not plot_pb.HasField(par):
        plot_pb.ylim.__setattr__('min', default_values[par][0])
        plot_pb.ylim.__setattr__('max', default_values[par][1])
    # 'ylim2'
    par = 'ylim2'
    if not plot_pb.HasField(par):
        plot_pb.ylim2.__setattr__('min', default_values[par][0])
        plot_pb.ylim2.__setattr__('max', default_values[par][1])
    for par in DEFAULT_LINE_PB_PARAMETER_LIST:
        if not plot_pb.default_line.HasField(par):
            plot_pb.default_line.__setattr__(par, default_values[par])
    return plot_pb


PLOT_PARAMETERS = [
    'marker',
    # 'figsize',
    'title',
    'xfmt',
    'yfmt',
    'fmtdate',
    'xlabel',
    'ylabel',
    'ylabel2',
    'legend_loc',
    # 'xscale',
    # 'yscale',
]

LINE_PARAMETERS = [
    'xcol',
    'ycol',
    'xcol2',
    'ycol2',
    'xshift',
    'yshift',
    'fmt',
    'color',
    'label',
    'prefilter',
    'infile',
]

DEFAULT_LINE_PARAMETERS = [
    'header',
    'sep',
    'sep2',
    # 'xlim',
    # 'ylim',
]


GENERIC_PARAMETERS = [
    'version',
    'debug',
    'dry_run',
    'outfile',
]


def convert_scale_string_to_config(scale):
    scale_pb = proto.plotty_pb2.Plot.Scale()
    if scale is None or scale == 'none':
        scale_pb = proto.plotty_pb2._PLOT_SCALE.none
    elif scale == 'linear':
        scale_pb = proto.plotty_pb2._PLOT_SCALE.linear
    elif scale == 'log':
        scale_pb = proto.plotty_pb2._PLOT_SCALE.log
    elif scale == 'symlog':
        scale_pb = proto.plotty_pb2._PLOT_SCALE.symlog
    elif scale == 'logit':
        scale_pb = proto.plotty_pb2._PLOT_SCALE.logit
    return scale_pb


def convert_namespace_to_config(options, gen_options=None):
    if gen_options is None:
        gen_options = argparse.Namespace()

    # 1. generic parameters
    for par in GENERIC_PARAMETERS:
        gen_options.__setattr__(par, options.__getattribute__(par))

    # 2. Plot
    plot_pb = proto.plotty_pb2.Plot()
    # copy all the 1:1 options:Plot parameters
    for par in PLOT_PARAMETERS:
        plot_pb.__setattr__(par, options.__getattribute__(par))
    # special parameters
    # figsize
    if options.figsize:
        plot_pb.figsize.width = options.figsize[0]
        plot_pb.figsize.height = options.figsize[1]
    # xscale
    if options.xscale is not None:
        plot_pb.xscale = convert_scale_string_to_config(options.xscale)
    # yscale
    if options.yscale is not None:
        plot_pb.yscale = convert_scale_string_to_config(options.yscale)
    # xlim
    if options.xlim is not None:
        plot_pb.xlim.min = options.xlim[0]
        plot_pb.xlim.max = options.xlim[1]
    # ylim
    if options.ylim is not None:
        plot_pb.ylim.min = options.ylim[0]
        plot_pb.ylim.max = options.ylim[1]
    # ylim2
    if options.ylim2 is not None:
        plot_pb.ylim2.min = options.ylim2[0]
        plot_pb.ylim2.max = options.ylim2[1]
    # twinx
    plot_pb.twinx = (options.twinx > 0 and len(options.infile) > 1)

    # 3. default Line
    # copy all the 1:N options:Line parameters
    for par in DEFAULT_LINE_PARAMETERS:
        # get the default Line element
        plot_pb.default_line.__setattr__(par, options.__getattribute__(par))
    # process complex parameters
    postfilter_list = convert_namespace_to_postfilters(options)
    for postfilter in postfilter_list:
        plot_pb.default_line.postfilter.append(postfilter)

    # 4. Line
    line_id_list = []
    for index in range(len(options.infile)):
        # add a new Line element
        line_pb = plot_pb.line.add()
        # infile does exist
        line_pb.infile = options.infile[index]
        # copy all the 1:1 options:Line parameters
        # label
        label = None
        if index < len(options.label):
            label = options.label[index]
            line_pb.label = label
        else:
            # default label name
            line_pb.label = f'{line_pb.infile} - {index}'
        # id
        line_pb.id = line_pb.label
        line_id_list.append(line_pb.id)
        # prefilter
        if index < len(options.prefilter):
            line_pb.prefilter = options.prefilter[index]
        # fmt
        if index < len(options.fmt):
            line_pb.fmt = options.fmt[index]
        else:
            line_pb.fmt = DEFAULT_FMT_LIST[index]
        # xcol
        if options.xcol:
            i = min(index, len(options.xcol) - 1)
            line_pb.xcol = options.xcol[i]
        else:
            # default value
            line_pb.xcol = '0'
        # ycol
        if options.ycol:
            i = min(index, len(options.ycol) - 1)
            line_pb.ycol = options.ycol[i]
        else:
            # default value
            line_pb.ycol = '1'
        # xcol2
        if options.xcol2:
            i = min(index, len(options.xcol2) - 1)
            line_pb.xcol2 = options.xcol2[i]
        # ycol2
        if options.ycol2:
            i = min(index, len(options.ycol2) - 1)
            line_pb.ycol2 = options.ycol2[i]
        # twinx
        line_pb.twinx = (options.twinx > 0 and index >= options.twinx)
        # postfilter.xshift
        if index < len(options.xshift):
            xshift = float(options.xshift[index])
            postfilter_pb = proto.plotty_pb2.Postfilter()
            postfilter_pb.type = proto.plotty_pb2.Postfilter.Type.xshift
            postfilter_pb.parameter = xshift
            line_pb.postfilter.append(postfilter_pb)
        # postfilter.yshift
        if index < len(options.yshift):
            yshift = float(options.yshift[index])
            postfilter_pb = proto.plotty_pb2.Postfilter()
            postfilter_pb.type = proto.plotty_pb2.Postfilter.Type.yshift
            postfilter_pb.parameter = yshift
            line_pb.postfilter.append(postfilter_pb)

    plot_line_list = []
    plot_id = 'cli'
    for line_id in line_id_list:
        plot_line_list.append([plot_pb, plot_id + '/' + line_id])
    return gen_options, plot_line_list


def convert_namespace_to_postfilters(options):
    postfilter_list = []
    # process histogram
    if options.histogram:
        postfilter_pb = proto.plotty_pb2.Postfilter()
        postfilter_pb.type = proto.plotty_pb2.Postfilter.Type.hist
        postfilter_pb.histogram.CopyFrom(
            convert_namespace_to_histogram(options))
        postfilter_list.append(postfilter_pb)
    # process simpler values
    # TODO(chema): process xshift, yshift
    if options.xfactor:
        postfilter_pb = proto.plotty_pb2.Postfilter()
        postfilter_pb.type = proto.plotty_pb2.Postfilter.Type.xfactor
        postfilter_pb.parameter = options.xfactor
        postfilter_list.append(postfilter_pb)
    if options.yfactor:
        postfilter_pb = proto.plotty_pb2.Postfilter()
        postfilter_pb.type = proto.plotty_pb2.Postfilter.Type.yfactor
        postfilter_pb.parameter = options.yfactor
        postfilter_list.append(postfilter_pb)
    if options.ydelta:
        postfilter_pb = proto.plotty_pb2.Postfilter()
        postfilter_pb.type = proto.plotty_pb2.Postfilter.Type.ydelta
        postfilter_pb.parameter = options.ydelta
        postfilter_list.append(postfilter_pb)
    if options.ycumulative:
        postfilter_pb = proto.plotty_pb2.Postfilter()
        postfilter_pb.type = proto.plotty_pb2.Postfilter.Type.ycumulative
        postfilter_pb.parameter = options.ycumulative
        postfilter_list.append(postfilter_pb)
    if options.use_mean:
        postfilter_pb = proto.plotty_pb2.Postfilter()
        postfilter_pb.type = proto.plotty_pb2.Postfilter.Type.mean
        postfilter_list.append(postfilter_pb)
    if options.use_median:
        postfilter_pb = proto.plotty_pb2.Postfilter()
        postfilter_pb.type = proto.plotty_pb2.Postfilter.Type.median
        postfilter_list.append(postfilter_pb)
    if options.use_stddev:
        postfilter_pb = proto.plotty_pb2.Postfilter()
        postfilter_pb.type = proto.plotty_pb2.Postfilter.Type.stddev
        postfilter_list.append(postfilter_pb)
    if options.use_regression:
        postfilter_pb = proto.plotty_pb2.Postfilter()
        postfilter_pb.type = proto.plotty_pb2.Postfilter.Type.regression
        postfilter_list.append(postfilter_pb)
    if options.use_moving_average is not None:
        postfilter_pb = proto.plotty_pb2.Postfilter()
        postfilter_pb.type = proto.plotty_pb2.Postfilter.Type.moving_average
        postfilter_pb.parameter = options.use_moving_average
        postfilter_list.append(postfilter_pb)
    if options.use_ewma is not None:
        postfilter_pb = proto.plotty_pb2.Postfilter()
        postfilter_pb.type = proto.plotty_pb2.Postfilter.Type.ewma
        postfilter_pb.parameter = options.use_ewma
        postfilter_list.append(postfilter_pb)
    return postfilter_list


def convert_namespace_to_histogram(options):
    if not options.histogram:
        return None
    histogram_pb = proto.plotty_pb2.Histogram()
    histogram_pb.enable = options.histogram
    histogram_pb.bins = options.histogram_bins
    histogram_pb.nozeroes = options.histogram_nozeroes
    if options.histogram_sigma is not None:
        histogram_pb.sigma = options.histogram_sigma
    histogram_pb.type = (proto.plotty_pb2._HISTOGRAM_TYPE.values_by_name[
       options.histogram_type].number)
    return histogram_pb


# Plot/Line accessors
def get_line_pb(plot_pb, line_id):
    for line_pb in plot_pb.line:
        if line_pb.id == line_id:
            return line_pb
    return None


# list of parameters where the result is the coalescing (AND) of the values
# of the line and of the default line
PARAMETER_AND_LIST = [
    'prefilter',
    'postfilter',
]


# list of parameters where the result is the value of the line, or the value
# of the default line if the former does not exist
PARAMETER_OPTION_LIST = [
    'id',
    'xshift',
    'yshift',
    'fmt',
    'color',
    'label',
    'infile',
    'header',
    'xcol',
    'ycol',
    'xcol2',
    'ycol2',
    'sep',
    'sep2',
    'xlim',
    'ylim',
]


def get_parameter(plot_pb, line_pb, par):
    # 1. Plot parameters
    if par == 'xfmt':
        xfmt_id = plot_pb.xfmt
        return proto.plotty_pb2._PLOT_COLUMNFMT.values_by_number[xfmt_id].name
    if par == 'yfmt':
        yfmt_id = plot_pb.yfmt
        return proto.plotty_pb2._PLOT_COLUMNFMT.values_by_number[yfmt_id].name

    # 2. Line parameters
    if par in PARAMETER_OPTION_LIST:
        # first try the element in the line
        if line_pb.__getattribute__(par):
            return line_pb.__getattribute__(par)
        # otherwise use the element in the default line
        return plot_pb.default_line.__getattribute__(par)

    if par == 'prefilter':
        # get the prefilter list
        prefilter_list = []
        if line_pb.__getattribute__(par):
            prefilter_list.append(line_pb.__getattribute__(par))
        if plot_pb.default_line.__getattribute__(par):
            prefilter_list.append(plot_pb.default_line.__getattribute__(par))
        # prefilter just does AND
        prefilter = ' AND '.join(prefilter_list)
        return prefilter

    if par == 'postfilter':
        # get the postfilter list
        postfilter_list = []
        if line_pb.__getattribute__(par):
            postfilter_list += line_pb.__getattribute__(par)
        if plot_pb.default_line.__getattribute__(par):
            postfilter_list += plot_pb.default_line.__getattribute__(par)
        # postfilter just does concatenation
        postfilter_list_pb = proto.plotty_pb2.PostfilterList()
        for postfilter in postfilter_list:
            postfilter_list_pb.postfilter.append(postfilter)
        return postfilter_list_pb

    # we should not be here
    raise AssertionError(f'unknown parameter: {par}')


def get_postfilter_type(postfilter_pb):
    return proto.plotty_pb2._POSTFILTER_TYPE.values_by_number[
        postfilter_pb.type].name


def get_histogram_type(histogram_pb):
    return proto.plotty_pb2._HISTOGRAM_TYPE.values_by_number[
        histogram_pb.type].name


def get_column_fmt_type(fmt):
    return proto.plotty_pb2._PLOT_COLUMNFMT.values_by_number[fmt].name


def scale_is_none(scale):
    return scale == proto.plotty_pb2.Plot.Scale.none


def get_scale_type(scale):
    return proto.plotty_pb2._PLOT_SCALE.values_by_number[scale].name
