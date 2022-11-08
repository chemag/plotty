#!/usr/bin/env python3

"""postfilter.py: Postfilter implementation.
"""


import importlib
import math
import numpy as np
import os
import sys
import scipy.optimize
from google.protobuf import text_format

config_lib = importlib.import_module('plotty-config')
utils = importlib.import_module('utils')

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
SCRIPT_ROOT_DIR = os.path.join(SCRIPT_DIR, '.')
sys.path.append(SCRIPT_ROOT_DIR)
import proto.plotty_pb2  # noqa: E402


# simple fitting function
def fit_function(x, a, b):
    return a * x + b


def get_moving_average(ylist, length):
    new_ylist = []
    for i in range(len(ylist)):
        max_index = min(i, len(ylist) - 1)
        min_index = max(0, i - length + 1)
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


def get_histogram(xlist, histogram_pb, debug):
    nbins = histogram_pb.bins
    nozeroes = histogram_pb.nozeroes
    sigma = histogram_pb.sigma
    htype = config_lib.get_histogram_type(histogram_pb)

    if sigma is not None and sigma:
        in_xlist, out_xlist, in_range = remove_outliers(xlist, sigma)
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


# postfilter processing
class Postfilter:
    def __init__(self, _type, parameter, histogram, debug):
        self.type = _type
        self.parameter = parameter
        self.histogram = histogram
        self.debug = debug

    @classmethod
    def constructor(cls, string):
        _type, parameter = string.split()
        return cls(_type, parameter, None, 0)

    def run(self, xlist, ylist):
        # 1. support for shift modes
        if self.type == 'xshift':
            xshift = self.parameter
            xlist = [(x + xshift) for x in xlist]
        elif self.type == 'yshift':
            yshift = self.parameter
            ylist = [(y + yshift) for y in ylist]
        # 2. support for factor modes
        elif self.type == 'xfactor':
            xfactor = self.parameter
            xlist = [(x * float(xfactor)) for x in xlist]
        elif self.type == 'yfactor':
            yfactor = self.parameter
            ylist = [(y * float(yfactor)) for y in ylist]
        # 3. support for ydelta (plotting `y[k] - y[k-1]` instead of `y[k]`)
        elif self.type == 'ydelta':
            ylist = [y1 - y0 for y0, y1 in zip([ylist[0]] + ylist[:-1], ylist)]
        # 4. support for ycumulative (plotting `\Sum y[k]` instead of `y[k]`)
        elif self.type == 'ycumulative':
            new_ylist = []
            for y in ylist:
                prev_y = new_ylist[-1] if new_ylist else 0
                new_ylist.append(y + prev_y)
            ylist = new_ylist
        # 5. support for replacements
        elif self.type == 'mean':
            mean = np.mean(ylist)
            ylist = [mean for _ in ylist]
        elif self.type == 'median':
            median = np.median(ylist)
            ylist = [median for _ in ylist]
        elif self.type == 'stddev':
            stddev = np.std(ylist)
            ylist = [stddev for _ in ylist]
        elif self.type == 'regression':
            # curve fit (linear regression)
            (a, b), _ = scipy.optimize.curve_fit(fit_function, xlist, ylist)
            ylist = [fit_function(x, a, b) for x in xlist]
        elif self.type == 'moving_average':
            # Moving Average fit (linear regression)
            length = int(self.parameter)
            ylist = get_moving_average(ylist, length)
        elif self.type == 'ewma':
            # Exponentially-Weighted Moving Average
            alpha = self.parameter
            ylist = get_ewma(ylist, alpha)
        # 6. support for histograms
        elif self.type == 'hist':
            xlist, ylist = get_histogram(xlist, self.histogram, self.debug)
        # 7. support for shift modes
        elif self.type == 'xsort':
            # do deco sorting based on the first element (x)
            deco_list = list(zip(xlist, ylist))
            deco_list.sort()
            xlist, ylist = zip(*deco_list)

        return xlist, ylist
