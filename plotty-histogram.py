#!/usr/bin/env python3

"""plotty-histogram.py module description.

Produces a histogram from a column in a given CSV input file.
"""


import argparse
import numpy as np
import pandas as pd
import sys


__version__ = "0.1"

HISTOGRAM_TYPES = ("raw", "pdf", "cdf")


default_values = {
    "debug": 0,
    "dry_run": False,
    "col": None,
    "nbins": 100,
    "range-min": None,
    "range-max": None,
    "header": True,
    "zeroes": True,
    "zeroes-strip": False,
    "zeroes-coalesce": False,
    "sigma": None,
    "type": "raw",
    "infile": None,
    "outfile": None,
}


def calculate_histogram(options):
    # read infile
    indf = pd.read_csv(options.infile, header=0 if options.header else None)
    # get the column
    if isinstance(options.col, int):
        column = indf.iloc[:, int(options.col)]
    else:
        column = indf.loc[:, options.col]
    # remove NaNs
    column = column.dropna()
    # process the data
    # get the histogram
    density = True if (options.type in ("pdf", "cdf")) else False
    range_min = options.range_min if options.range_min is not None else column.min()
    range_max = options.range_max if options.range_max is not None else column.max()
    hist, bin_edges = np.histogram(
        column, bins=options.nbins, range=(range_min, range_max), density=density
    )
    # center the histogram (bin_edges points to the left side)
    bin_edges = (bin_edges[1:] + bin_edges[:-1]) / 2
    # add up the values for a CDF
    if options.type == "cdf":
        raise AssertionError("error: CDF mode unimplemented")
    # create the output dataframe
    outdf = pd.DataFrame({"value": bin_edges, "total": hist})
    if not options.zeroes:
        # remove zeroes
        outdf = outdf[outdf["total"] != 0]
    elif options.zeroes_strip:
        first_element = outdf[outdf.total != 0].index[0]
        last_element = outdf[outdf.total != 0].index[-1]
        outdf = outdf.iloc[first_element:last_element]
    elif options.zeroes_coalesce:
        # remove all rows such that (a) its value for the column total is zero, and
        # (b) the value for the column total is zero for both the top and bottom row
        outdf = outdf[
            (outdf.total != 0)
            | (
                (outdf.total == 0)
                & ((outdf.total.shift() != 0) | (outdf.total.shift(-1) != 0))
            )
        ]
    # add ratios
    total_occurrences = int(outdf.total.sum())
    outdf["ratio"] = outdf.apply(lambda row: row["total"] / total_occurrences, axis=1)
    # write outfile
    outdf.to_csv(options.outfile, index=False)


def get_options(argv):
    """Generic option parser.

    Args:
        argv: list containing arguments

    Returns:
        Namespace - An argparse.ArgumentParser-generated option object
    """
    # init parser
    # usage = 'usage: %prog [options] arg1 arg2'
    # parser = argparse.OptionParser(usage=usage)
    # parser.print_help() to get argparse.usage (large help)
    # parser.print_usage() to get argparse.usage (just usage line)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        dest="version",
        default=False,
        help="Print version",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="count",
        dest="debug",
        default=default_values["debug"],
        help="Increase verbosity (use multiple times for more)",
    )
    parser.add_argument(
        "--quiet",
        action="store_const",
        dest="debug",
        const=-1,
        help="Zero verbosity",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        default=default_values["dry_run"],
        help="Dry run",
    )
    parser.add_argument(
        "--col",
        action="store",
        dest="col",
        default=default_values["col"],
        metavar="COL",
        help="use COL column",
    )
    parser.add_argument(
        "--nbins",
        action="store",
        type=int,
        dest="nbins",
        default=default_values["nbins"],
        metavar="NBINS",
        help="use NBINS bins",
    )
    parser.add_argument(
        "--range-min",
        action="store",
        type=float,
        dest="range_min",
        default=default_values["range-min"],
        metavar="RANGE-MIN",
        help="use RANGE-MIN as range min",
    )
    parser.add_argument(
        "--range-max",
        action="store",
        type=float,
        dest="range_max",
        default=default_values["range-max"],
        metavar="RANGE-MAX",
        help="use RANGE-MAX as range max",
    )
    parser.add_argument(
        "--header",
        dest="header",
        action="store_true",
        default=default_values["header"],
        help="Input file has CSV header%s"
        % (" [default]" if default_values["header"] else ""),
    )
    parser.add_argument(
        "--no-header",
        dest="header",
        action="store_false",
        help="Input file does not have CSV header%s"
        % (" [default]" if not default_values["header"] else ""),
    )
    parser.add_argument(
        "--zeroes",
        dest="zeroes",
        action="store_true",
        default=default_values["zeroes"],
        help="Do not ignore zero elements%s"
        % (" [default]" if default_values["zeroes"] else ""),
    )
    parser.add_argument(
        "--no-zeroes",
        dest="zeroes",
        action="store_false",
        help="Ignore zero elements%s"
        % (" [default]" if not default_values["zeroes"] else ""),
    )
    parser.add_argument(
        "--zeroes-strip",
        dest="zeroes_strip",
        action="store_true",
        default=default_values["zeroes-strip"],
        help="Strip zero elements at the left and right sides%s"
        % (" [default]" if default_values["zeroes-strip"] else ""),
    )
    parser.add_argument(
        "--no-zeroes-strip",
        dest="zeroes_strip",
        action="store_false",
        help="Do not strip zero elements at the left and right sides%s"
        % (" [default]" if not default_values["zeroes-strip"] else ""),
    )
    parser.add_argument(
        "--zeroes-coalesce",
        dest="zeroes_coalesce",
        action="store_true",
        default=default_values["zeroes-coalesce"],
        help="Coalesce adjacent zero elements%s"
        % (" [default]" if default_values["zeroes-coalesce"] else ""),
    )
    parser.add_argument(
        "--no-zeroes-coalesce",
        dest="zeroes_coalesce",
        action="store_false",
        help="Do not coalesce adjacent zero elements%s"
        % (" [default]" if not default_values["zeroes-coalesce"] else ""),
    )
    parser.add_argument(
        "--sigma",
        action="store",
        type=int,
        dest="sigma",
        default=default_values["sigma"],
        metavar="SIGMA",
        help="use SIGMA sigma",
    )
    parser.add_argument(
        "--type",
        action="store",
        type=str,
        dest="type",
        default=default_values["type"],
        choices=HISTOGRAM_TYPES,
        metavar="[%s]"
        % (
            " | ".join(
                HISTOGRAM_TYPES,
            )
        ),
        help="enum arg",
    )
    parser.add_argument(
        "-i",
        "--infile",
        dest="infile",
        type=str,
        default=default_values["infile"],
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        type=str,
        default=default_values["outfile"],
        metavar="output-file",
        help="output file",
    )
    # do the parsing
    options = parser.parse_args(argv[1:])
    if options.version:
        return options
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    if options.version:
        print("version: %s" % __version__)
        sys.exit(0)
    # get infile/outfile
    if options.infile is None or options.infile == "-":
        options.infile = "/dev/fd/0"
    if options.outfile is None or options.outfile == "-":
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(options)
    # do something
    calculate_histogram(options)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
