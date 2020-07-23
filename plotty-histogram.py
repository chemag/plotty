#!/usr/bin/env python3

"""histogram.py: simple histogram plotter.

# http://matplotlib.org/examples/statistics/histogram_demo_cumulative.html

# runme
# $ echo -e "1\n2\n1\n2\n3\n1\n4\n" | ./plotty-histogram.py -i - /tmp/bar.png
"""

import sys
import importlib

plotty_plot = importlib.import_module('plotty-plot')

if __name__ == '__main__':
    # add "--histogram" option
    sys.argv.append('--histogram')
    # at least the CLI program name: (CLI) execution
    plotty_plot.main(sys.argv)
