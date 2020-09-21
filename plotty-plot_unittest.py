#!/usr/bin/env python3

"""histogram.py: simple histogram plotter.

# http://matplotlib.org/examples/statistics/histogram_demo_cumulative.html

# runme
# $ echo -e "1\n2\n1\n2\n3\n1\n4\n" | ./plotty-histogram.py -i - /tmp/bar.png
"""

import importlib
import unittest

plotty_plot = importlib.import_module('plotty-plot')


data = """ # comment
0,1,1,1,1,1,1,1,0,0,0:0:0:0
0,1,1,1,1,1,1,1,1,1,11:111:1111:11111
0,1,1,1,1,1,1,1,2,2,22:222:2222:22222
0,1,1,1,1,1,1,2,3,3,33:333:3333:33333
0,1,1,1,1,1,2,3,4,4,44:444:4444:44444
0,1,1,1,1,2,3,4,5,5,55:555:5555:55555
0,1,1,1,2,3,4,5,6,6,66:666:6666:66666
0,1,1,2,3,4,5,6,7,7,77:777:7777:77777
0,1,2,3,4,5,6,7,8,8,88:888:8888:88888
0,1,2,3,4,5,6,7,8,100,99:999:9999:99999
"""

parseDataTestCases = [
    {
        'name': 'basic',
        'parameters': {
            'sep': ',',
            'xcol': 1,
            'ycol': 2,
        },
        'xlist': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'ylist': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
    },
    {
        'name': '[x|y]col == -1 implies line number',
        'parameters': {
            'sep': ',',
            'xcol': -1,
            'ycol': 1,
        },
        'xlist': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        'ylist': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
    {
        'name': 'xcol2',
        'parameters': {
            'sep': ',',
            'sep2': ':',
            'xcol': 10,
            'xcol2': 1,
            'ycol': 2,
        },
        'xlist': [0.0, 111.0, 222.0, 333.0, 444.0, 555.0, 666.0, 777.0,
                  888.0, 999.0],
        'ylist': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
    },
    {
        'name': 'ycol2, swapping separators',
        'parameters': {
            'sep': ':',
            'sep2': ',',
            'xcol': 0,
            'xcol2': 1,
            'ycol': 0,
            'ycol2': 8,
        },
        'xlist': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'ylist': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0],
    },
    {
        'name': 'ydelta',
        'parameters': {
            'sep': ',',
            'xcol': 1,
            'ycol': 2,
            'ydelta': True,
        },
        'xlist': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'ylist': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    },
    {
        'name': 'ycumulative',
        'parameters': {
            'sep': ',',
            'xcol': 1,
            'ycol': 2,
            'ycumulative': True,
        },
        'xlist': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'ylist': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0],
    },
    {
        'name': 'prefilter',
        'parameters': {
            'sep': ',',
            'xcol': 1,
            'ycol': 2,
            'filter': ((5, 'ne', '1'),),
        },
        'xlist': [1.0, 1.0, 1.0, 1.0, 1.0],
        'ylist': [1.0, 1.0, 1.0, 2.0, 2.0],
    },
    {
        'name': 'prefilter arithmetic',
        'parameters': {
            'sep': ',',
            'xcol': 1,
            'ycol': 2,
            'filter': ((5, 'gt', '1'),),
        },
        'xlist': [1.0, 1.0, 1.0, 1.0, 1.0],
        'ylist': [1.0, 1.0, 1.0, 2.0, 2.0],
    },
    {
        'name': 'histogram',
        'parameters': {
            'sep': ',',
            'xcol': 2,
            'histogram': True,
            'histogram_bins': 2,
        },
        'xlist': [1.25, 1.75],
        'ylist': [8.0, 2.0],
    },
    {
        'name': 'histogram bins 1',
        'parameters': {
            'sep': ',',
            'xcol': 8,
            'histogram': True,
            'histogram_bins': 2,
        },
        'xlist': [2.0, 6.0],
        'ylist': [4.0, 6.0],
    },
    {
        'name': 'histogram bins 2',
        'parameters': {
            'sep': ',',
            'xcol': 8,
            'histogram': True,
            'histogram_bins': 4,
        },
        'xlist': [1.0, 3.0, 5.0, 7.0],
        'ylist': [2.0, 2.0, 2.0, 4.0],
    },
    {
        'name': 'histogram ratio 1',
        'parameters': {
            'sep': ',',
            'xcol': 8,
            'histogram': True,
            'histogram_bins': 2,
            'histogram_ratio': True,
        },
        'xlist': [2.0, 6.0],
        'ylist': [0.4, 0.6],
    },
    {
        'name': 'histogram ratio 2',
        'parameters': {
            'sep': ',',
            'xcol': 8,
            'histogram': True,
            'histogram_bins': 4,
            'histogram_ratio': True,
        },
        'xlist': [1.0, 3.0, 5.0, 7.0],
        'ylist': [0.2, 0.2, 0.2, 0.4],
    },
    {
        'name': 'histogram sigma 1',
        'parameters': {
            'sep': ',',
            'xcol': 9,
            'histogram': True,
            'histogram_bins': 2,
            'histogram_sigma': 2.0,
        },
        'xlist': [2.0, 6.0],
        'ylist': [4.0, 5.0],
    },
    {
        'name': 'histogram sigma 2',
        'parameters': {
            'sep': ',',
            'xcol': 9,
            'histogram': True,
            'histogram_bins': 4,
            'histogram_sigma': 2.0,
        },
        'xlist': [1.0, 3.0, 5.0, 7.0],
        'ylist': [2.0, 2.0, 2.0, 3.0],
    },
    {
        'name': 'histogram sigma 2b',
        'parameters': {
            'sep': ',',
            'xcol': 9,
            'histogram': True,
            'histogram_bins': 4,
            'histogram_sigma': 20.0,
        },
        'xlist': [12.5, 37.5, 62.5, 87.5],
        'ylist': [9.0, 0.0, 0.0, 1.0],
    },
]


class MyTest(unittest.TestCase):

    def testParseDataBasic(self):
        """Simplest parse_data test."""
        for test_case in parseDataTestCases:
            xlist, ylist = plotty_plot.parse_data_internal(
                data, **test_case['parameters'])
            msg = 'unittest failed: %s' % test_case['name']
            self.assertEqual(test_case['xlist'], xlist, msg=msg)
            self.assertEqual(test_case['ylist'], ylist, msg=msg)


if __name__ == '__main__':
    unittest.main()
    # plotty_plot.main(sys.argv)
