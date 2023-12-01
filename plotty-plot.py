#!/usr/bin/env python3

"""plot.py: simple data plotter.

# http://stackoverflow.com/a/11249340

# runme
# $ echo -e "1,2\n2,2\n3,5\n4,7\n5,1\n" | ./plotty-plot.py -i - /tmp/plot.png
"""

import csv
import datetime
import importlib
import math
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import os
import sys

config_lib = importlib.import_module("plotty-config")
prefilter_lib = importlib.import_module("prefilter")
postfilter_lib = importlib.import_module("postfilter")
utils = importlib.import_module("utils")


__version__ = "0.3"


def read_file(infile):
    # open infile
    if infile == "-":
        infile = "/dev/fd/0"
    with open(infile, "r") as fin:
        # read data
        raw_data = fin.read()
    return raw_data


def parse_csv(raw_data, sep, header):
    # split the input in lines
    lines = raw_data.split("\n")
    # look for named columns in line 0
    column_names = []
    if lines[0].strip().startswith("#") or header:
        column_names = lines[0].strip().split(sep)
        if column_names[0].startswith("#"):
            column_names[0] = column_names[0][1:]
        # remove extra spaces
        column_names = [colname.strip() for colname in column_names]
        lines = lines[1:]
    # remove comment lines
    lines = [line for line in lines if not line.strip().startswith("#")]
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
    val = list(csv.reader([line], delimiter=sep, quotechar='"'))[0][col]
    if col2 is not None and col2:
        # use sep1, then sep2
        if not val:
            # empty column value
            return None
        # parse value
        val = val.split(sep2)[int(col2)]
    return val


def parse_line(line, i, sep, xcol, ycol, sep2, xcol2, ycol2):
    # get x component
    x = i if xcol == -1 else get_column(line, sep, xcol, sep2, xcol2)
    # get y component
    y = i if ycol == -1 else get_column(line, sep, ycol, sep2, ycol2)
    return x, y


def get_data(plot_pb, line_pb, gen_options):
    # read the input file
    # read the input file
    infile = config_lib.get_parameter(plot_pb, line_pb, "infile")
    if not infile:
        if line_pb.HasField("data"):
            return read_data(line_pb.data)
        raise Exception("Error: no infile or data info")
    raw_data = read_file(infile)
    return get_data_raw_data(raw_data, plot_pb, line_pb, gen_options)


def read_data(data):
    # ensure the minimum values exist
    xlist = []
    ylist = []
    for point in data.point:
        xlist.append(point.x if point.HasField("x") else 0)
        ylist.append(point.y if point.HasField("y") else 0)
    return xlist, ylist


def get_data_raw_data(raw_data, plot_pb, line_pb, gen_options):
    # 1. convert the raw data into lines
    sep = config_lib.get_parameter(plot_pb, line_pb, "sep")
    header = config_lib.get_parameter(plot_pb, line_pb, "header")
    column_names, lines = parse_csv(raw_data, sep, header)

    # 2. fix column names in the prefilter
    prefilter_str = config_lib.get_parameter(plot_pb, line_pb, "prefilter")
    prefilter = prefilter_lib.Prefilter(prefilter_str)
    prefilter.fix_columns(column_names)

    # 3. get column ids
    xcol = config_lib.get_parameter(plot_pb, line_pb, "xcol")
    ycol = config_lib.get_parameter(plot_pb, line_pb, "ycol")
    xcol, ycol = get_column_ids(xcol, ycol, column_names)

    # 4. parse all the lines into (xlist, ylist)
    sep2 = config_lib.get_parameter(plot_pb, line_pb, "sep2")
    sep2 = sep2 if sep2 != "" else None
    xcol2 = config_lib.get_parameter(plot_pb, line_pb, "xcol2")
    ycol2 = config_lib.get_parameter(plot_pb, line_pb, "ycol2")
    xfmt = config_lib.get_parameter(plot_pb, line_pb, "xfmt")
    yfmt = config_lib.get_parameter(plot_pb, line_pb, "yfmt")
    xlist = []
    ylist = []
    for i, line in enumerate(lines):
        if not line:
            # empty line
            continue
        # prefilter lines
        if not prefilter.match_line(line, sep):
            continue
        x, y = parse_line(line, i, sep, xcol, ycol, sep2, xcol2, ycol2)
        if x is not None and y is not None:
            # append values
            xlist.append(fmt_convert(x, xfmt))
            ylist.append(fmt_convert(y, yfmt))

    # 5. postfilter values
    postfilter_list = config_lib.get_parameter(plot_pb, line_pb, "postfilter")
    for postfilter_pb in postfilter_list.postfilter:
        postfilter_type = config_lib.get_postfilter_type(postfilter_pb)
        postfilter = postfilter_lib.Postfilter(
            postfilter_type,
            postfilter_pb.parameter,
            postfilter_pb.parameter_str,
            postfilter_pb.histogram,
            gen_options.debug,
        )
        xlist, ylist = postfilter.run(xlist, ylist)
    return xlist, ylist


# convert axis format
def fmt_convert(item, fmt):
    if fmt == "int":
        return int(float(item))
    elif fmt == "float":
        if item == "":
            return ""
        elif item is None or item == "None":
            return np.nan
        return float(item)
    elif fmt == "unix":
        # convert unix timestamp to matplotlib datenum
        return md.date2num(datetime.datetime.fromtimestamp(float(item)))
    raise Exception("Error: invalid fmt (%s)" % fmt)


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


def create_graph_begin(plot_pb):
    # create figure
    fig = plt.figure(figsize=(plot_pb.figsize.width, plot_pb.figsize.height))
    # plt.gca().set_xlim([xstart, xstart + 1100000000])
    # plt.gca().set_ylim([0, 100])
    ax1 = fig.add_subplot(111)
    ax1.set_title(plot_pb.title)
    ax1.set_xlabel(plot_pb.xlabel)
    xfmt = config_lib.get_column_fmt_type(plot_pb.xfmt)
    if xfmt == "unix":
        xfmt = md.DateFormatter(plot_pb.fmtdate)
        ax1.xaxis.set_major_formatter(xfmt)
    return ax1


def matplotlib_fmt_parse(fmt):
    # https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.plot.html
    # A format string consists of a part for color, marker and linestyle:
    # fmt = '[marker][linestyle][color]'
    # Each of them is optional. If not provided, the value from the style
    # cycle is used. Exception: If linestyle is given, but no marker, the data
    # will be a line without markers.
    # Other combinations such as [color][marker][linestyle] are also supported,
    # but note that their parsing may be ambiguous.
    marker, linestyle, color = None, None, None

    # high-priority syntax: "<color>,<marker>,<linestyle>"
    if fmt.count(",") > 1:
        if fmt.count(",") == 2:
            marker, linestyle, color = fmt.split(",")
        elif fmt.count(",") == 3 and ",,," in fmt:
            # "," is a valid character for the marker
            marker = ","
            linestyle, color = fmt.split(",,,")
        else:
            # invalid fmt
            raise AssertionError(f"error: invalid fmt type: '{fmt}'")
        return marker, linestyle, color

    # let's start with color in '[color][marker][linestyle]'
    # single letter (e.g. 'b'),
    if len(fmt) >= 1 and fmt[0] in config_lib.VALID_MATPLOTLIB_COLORS:
        color = fmt[0]
        fmt = fmt[1:]
    # predefined colors (e.g. 'C0')
    if len(fmt) >= 2 and fmt[0] == "C" and fmt[1].isdigit():
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
            fmt = fmt[len(ls) :]
            break
    else:
        linestyle = None

    # color
    if color is None:
        color = fmt
    return marker, linestyle, color


def create_graph_draw(ax, xlist, ylist, line_pb, plot_pb, gen_options):
    marker, linestyle, color = matplotlib_fmt_parse(line_pb.fmt)

    try:
        ax.plot(
            xlist,
            ylist,
            color=color,
            linestyle=linestyle,
            marker=marker,
            label=line_pb.label,
        )
    except ValueError as vex:
        print(f"error: {vex}")
        print("  ax_plot(")
        print(f"    xlist ({len(xlist)} elements),")
        print(f"    ylist ({len(ylist)} elements),")
        print(f"    color={color},")
        print(f"    linestyle={linestyle},")
        print(f"    marker={marker},")
        print(f"    label={line_pb.label},")
        print(")")

    if gen_options.debug > 1:
        print(f"ax.plot(xlist: {xlist} ylist: {ylist} label: {line_pb.label}")

    if plot_pb.xfmt == "int":
        # make sure the ticks are all integers
        num_values = len(ax.get_xticks())
        step = int(
            math.ceil(
                (
                    int(math.ceil(ax.get_xticks()[-1]))
                    - int(math.floor(ax.get_xticks()[0]))
                )
                / num_values
            )
        )
        ax.set_xticks(range(int(ax.get_xticks()[0]), int(ax.get_xticks()[-1]), step))


def create_graph_end(ax, ylabel, ylim, plot_pb):
    ax.set_ylabel(ylabel)

    # set xlim/ylim
    if plot_pb.HasField("xlim") and plot_pb.xlim.min != "-":
        ax.set_xlim(left=float(plot_pb.xlim.min))
    if plot_pb.HasField("xlim") and plot_pb.xlim.max != "-":
        ax.set_xlim(right=float(plot_pb.xlim.max))

    if plot_pb.HasField("ylim") and ylim[0] != "-":
        ax.set_ylim(bottom=float(ylim[0]))
    if plot_pb.HasField("ylim") and ylim[1] != "-":
        ax.set_ylim(top=float(ylim[1]))

    # set xscale/yscale
    if not config_lib.scale_is_none(plot_pb.xscale):
        ax.set_xscale(config_lib.get_scale_type(plot_pb.xscale))
    if not config_lib.scale_is_none(plot_pb.yscale):
        ax.set_yscale(config_lib.get_scale_type(plot_pb.yscale))


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
    sep = sep if sep != "" else None

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


def main(argv):
    # parse options
    gen_options, plot_line_list = config_lib.get_options(argv)
    if gen_options.version:
        print("version: %s" % __version__)
        sys.exit(0)

    # 1. get all the per-line info into xy_data
    # 1.1. get infile(s)/outfile
    # batch_label_list = []
    # if options.batch_infile is not None:
    #    infile_list = batch_process_file(
    #        options.batch_infile, options.batch_sep, options.batch_col,
    #        options.batch_prefilter, options.header)
    #    batch_label_list = batch_process_data(
    #        read_file(options.batch_infile), options.batch_sep,
    #        options.batch_label_col, options.batch_prefilter)
    # else:

    if gen_options.outfile == "-":
        gen_options.outfile = "/dev/fd/1"

    if gen_options.debug > 0:
        print(f"gen_options: {gen_options}")
        if gen_options.debug > 1:
            print(f"plot_line_list: {plot_line_list}")

    # 1.2. get all the per-line info into xy_data
    # Includes `ycol`, `xlist` (x-axis values), `ylist` (y-axis values),
    # `label`, `fmt`, `color`, and `prefilter`
    xy_data = []
    for plot_pb, plot_line_id in plot_line_list:
        _, line_id = plot_line_id.split("/", maxsplit=1)
        line_pb = config_lib.get_line_pb(plot_pb, line_id)
        # fix infile name
        if line_pb.infile == "-":
            line_pb.infile = "/dev/fd/0"
        # get all the info from the current line
        try:
            xlist, ylist = get_data(plot_pb, line_pb, gen_options)
        except FileNotFoundError:
            infile = config_lib.get_parameter(plot_pb, line_pb, "infile")
            print(f"error: file not found: {infile}")
            raise
        except AssertionError as ae:
            print(f"error: get_data(plot_pb, {line_pb=}, gen_options)\n  {ae}")
            raise
        except:
            print(f"error: get_data(plot_pb, {line_pb=}, gen_options)")
            raise
        xy_data.append([xlist, ylist, line_pb])

    if gen_options.dry_run:
        return xy_data

    # 2. get all the per-axes info
    # create the main axis
    ax = []
    axinfo = []
    ax.append(create_graph_begin(plot_pb))
    ylabel = plot_pb.ylabel
    ylim = (plot_pb.ylim.min, plot_pb.ylim.max)

    axinfo.append([ylabel, ylim])
    # create the twin axis
    if plot_pb.twinx:
        ax.append(ax[0].twinx())
        ylabel2 = plot_pb.ylabel2
        ylim2 = (plot_pb.ylim2.min, plot_pb.ylim2.max)
        axinfo.append([ylabel2, ylim2])

    # 3. create the graph
    # add each of the lines in xy_data
    axid = 0
    for (xlist, ylist, line_pb) in xy_data:
        axid = 1 if line_pb.twinx else 0
        create_graph_draw(ax[axid], xlist, ylist, line_pb, plot_pb, gen_options)

    # set final graph details
    for axid, (ylabel, ylim) in enumerate(axinfo):
        # set the values
        create_graph_end(ax[axid], ylabel, ylim, plot_pb)

    # set common legend
    if plot_pb.HasField("legend_loc") and plot_pb.legend_loc != "none":
        # https://stackoverflow.com/a/14344146
        lin_list = []
        leg_list = []
        for axid in range(len(ax)):
            lin, leg = ax[axid].get_legend_handles_labels()
            lin_list += lin
            leg_list += leg
        _ = ax[0].legend(lin_list, leg_list, loc=plot_pb.legend_loc)

    # save graph
    if gen_options.debug > 0:
        print("output is %s" % gen_options.outfile)
    plt.savefig("%s" % gen_options.outfile)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
