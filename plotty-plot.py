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

default_values = {
    'debug': 0,
    'title': 'Plot Title',
    'label': [],
    'xcol': 0,
    'ycol': 1,
    'filter': None,
    'fmt': [],
    'sep': None,
    'xlabel': 'x axis',
    'ylabel': 'y axis',
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
        x = range(len([row for row in data if row]))
    else:
        x = [float(row.split(sep)[options.xcol]) for row in data if row]
    y = [float(row.split(sep)[options.ycol]) for row in data if row]
    return x, y


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


def create_graph_draw(ax1, x, y, fmt, label, options):
    ax1.plot(x, y, fmt, label=label)
    if options.debug > 1:
        print('ax1.plot(%r, %r, \'%s\', label=%r)' % (
            list(x), y, fmt, label))


def create_graph_end(ax1, options):
    _ = ax1.legend(loc='upper right')
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
                        help='use YLABEL label(s)',)
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
    for index, (x, y) in enumerate(xy_data):
        fmt = (options.fmt[index] if index < len(options.fmt) else
               DEFAULT_FMT[index])
        label = options.label[index] if index < len(options.label) else ''
        create_graph_draw(ax1, x, y, fmt, label, options)
    create_graph_end(ax1, options)
    # create_graph(x, y, options)


if __name__ == '__main__':
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
