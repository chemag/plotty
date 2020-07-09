#!/usr/bin/env python3

"""plot.py: simple data plotter.

# http://stackoverflow.com/a/11249340

# runme
# $ echo -e "1 2\n2 2\n3 5\n4 7\n5 1\n" | ./plotty-plot.py -i - /tmp/foo.png
"""

import argparse
import matplotlib.pyplot as plt
import sys


DEFAULT_MARKER = '.'
DEFAULT_COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
DEFAULT_FMT = ['%s%s' % (color, DEFAULT_MARKER) for color in DEFAULT_COLORS]

YSCALE_VALUES = ('linear', 'log', 'symlog', 'logit')

default_values = {
    'debug': 0,
    'title': '--title',
    'xcol': 0,
    'ycol': 1,
    'ycol2': None,
    'ydelta': False,
    'filter': None,
    'sep': None,
    'sep2': None,
    'xlabel': '--xlabel',
    'ylabel': '--ylabel',
    'xshift': [],
    'yshift': [],
    'label': [],
    'fmt': [],
    'xlim': ['-', '-'],
    'ylim': ['-', '-'],
    'yscale': 'linear',
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

    if options.xcol == -1:
        # use line number
        xlist = range(len([row for row in data if row]))
    else:
        xlist = [float(row.split(sep)[options.xcol]) for row in data if row]

    if options.ycol2 is None:
        ylist = [float(row.split(sep)[options.ycol]) for row in data if row]
    else:
        sep2 = options.sep2 if options.sep2 is not None else ' '
        ylist = [float(row.split(sep)[options.ycol].split(sep2)[options.ycol2])
                 for row in data if row]
    # support for plotting `y[k] - y[k-1]` instead of `y[k]`
    if options.ydelta:
        ylist = [y1 - y0 for y0, y1 in zip([ylist[0]] + ylist[:-1], ylist)]

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


def create_graph_draw(ax1, xlist, ylist, fmt, label, options):
    ax1.plot(xlist, ylist, fmt, label=label)
    if options.debug > 1:
        print('ax1.plot(%r, %r, \'%s\', label=%r)' % (
            list(xlist), ylist, fmt, label))


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
                        help='use (y[k] - y[k-1]) for y col',)
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
                        help='use YLABEL label(s)',)
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
    for infile in options.infile:
        xy_data.append(read_data(infile, options))

    ax1 = create_graph_begin(options)
    for index, (xlist, ylist) in enumerate(xy_data):
        fmt = (options.fmt[index] if index < len(options.fmt) else
               DEFAULT_FMT[index])
        label = options.label[index] if index < len(options.label) else ''
        # process shift requests
        xshift = (float(options.xshift[index]) if index < len(options.xshift)
                  else None)
        if xshift is not None:
            print('shifting x by %f' % xshift)
            xlist = [(x + xshift) for x in xlist]
        yshift = (float(options.yshift[index]) if index < len(options.yshift)
                  else None)
        if yshift is not None:
            print('shifting y by %f' % yshift)
            ylist = [(y + yshift) for y in ylist]
        create_graph_draw(ax1, xlist, ylist, fmt, label, options)
    create_graph_end(ax1, options)
    # create_graph(xlist, ylist, options)


if __name__ == '__main__':
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
