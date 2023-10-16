#!/usr/bin/env python3

"""plotty-plot_unittest.py: plotty unittest.

# runme
# $ ./plotty-plot_unittest.py
"""

import importlib
import math
import os
import shlex
import string
import tempfile
import unittest

config_lib = importlib.import_module("plotty-config")
plotty_plot = importlib.import_module("plotty-plot")


dataGetDataTestCases = """ # comment
0,1,1,1,1,1,1,1,0,0,0,9,foobar,0:0:0:0
0,1,1,1,1,1,1,1,1,1,1,8,foobar,11:111:1111:11111
0,1,1,1,1,1,1,1,2,2,2,7,foobarbaz,22:222:2222:22222
0,1,1,1,1,1,1,2,3,3,3,6,foobarbaz,33:333:3333:33333
0,1,1,1,1,1,2,3,4,4,4,5,foo,44:444:4444:44444
0,1,1,1,1,2,3,4,5,5,5,4,foo,55:555:5555:55555
0,1,1,1,2,3,4,5,6,6,6,3,foo,66:666:6666:66666
0,1,1,2,3,4,5,6,7,7,7,2,bar,77:777:7777:77777
0,1,2,3,4,5,6,7,8,8,8,1,bar,88:888:8888:88888
0,1,2,3,4,5,6,7,8,9,100,0,bar,99:999:9999:99999
"""

parseDataTestCases = [
    {
        "name": "basic",
        "parameters": {
            "sep": ",",
            "xcol": 1,
            "ycol": 2,
        },
        "xlist": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "ylist": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
    },
    {
        "name": "[x|y]col == -1 implies line number",
        "parameters": {
            "sep": ",",
            "xcol": -1,
            "ycol": 1,
        },
        "xlist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ylist": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
    {
        "name": "xcol2",
        "parameters": {
            "sep": ",",
            "sep2": ":",
            "xcol": 13,
            "xcol2": 1,
            "ycol": 2,
        },
        "xlist": [0.0, 111.0, 222.0, 333.0, 444.0, 555.0, 666.0, 777.0, 888.0, 999.0],
        "ylist": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
    },
    {
        "name": "ycol2, swapping separators",
        "parameters": {
            "sep": ":",
            "sep2": ",",
            "xcol": 0,
            "xcol2": 1,
            "ycol": 0,
            "ycol2": 8,
        },
        "xlist": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "ylist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0],
    },
    # postfilter cases
    {
        "name": "postfilter empty",
        "parameters": {
            "sep": ",",
            "xcol": -1,
            "ycol": 9,
        },
        "xlist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ylist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    },
    {
        "name": "postfilter ydelta",
        "parameters": {
            "sep": ",",
            "xcol": -1,
            "ycol": 9,
            "ydelta": True,
        },
        "xlist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ylist": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
    {
        "name": "postfilter ycumulative",
        "parameters": {
            "sep": ",",
            "xcol": -1,
            "ycol": 9,
            "ycumulative": True,
        },
        "xlist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ylist": [0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0],
    },
    # {
    #    'name': 'postfilter xshift',
    #    'parameters': {
    #        'sep': ',',
    #        'xcol': -1,
    #        'ycol': 9,
    #        'xshift': 10,
    #    },
    #    'xlist': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
    #    'ylist': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    # },
    # {
    #    'name': 'postfilter yshift',
    #    'parameters': {
    #        'sep': ',',
    #        'xcol': -1,
    #        'ycol': 9,
    #        'yshift': 10,
    #    },
    #    'xlist': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    #    'ylist': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
    # },
    {
        "name": "postfilter mean",
        "parameters": {
            "sep": ",",
            "xcol": -1,
            "ycol": 9,
            "use-mean": True,
        },
        "xlist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ylist": [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5],
    },
    {
        "name": "postfilter median",
        "parameters": {
            "sep": ",",
            "xcol": -1,
            "ycol": 9,
            "use-median": True,
        },
        "xlist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ylist": [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5],
    },
    {
        "name": "postfilter regression",
        "parameters": {
            "sep": ",",
            "xcol": -1,
            "ycol": 9,
            "use-regression": True,
        },
        "xlist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ylist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    },
    {
        "name": "postfilter moving average 1",
        "parameters": {
            "sep": ",",
            "xcol": -1,
            "ycol": 9,
            "use-moving-average": 1,
        },
        "xlist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ylist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    },
    {
        "name": "postfilter moving average 2",
        "parameters": {
            "sep": ",",
            "xcol": -1,
            "ycol": 9,
            "use-moving-average": 2,
        },
        "xlist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ylist": [0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
    },
    {
        "name": "postfilter moving average 3",
        "parameters": {
            "sep": ",",
            "xcol": -1,
            "ycol": 9,
            "use-moving-average": 3,
        },
        "xlist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ylist": [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    },
    {
        "name": "postfilter ewma 1.0",
        "parameters": {
            "sep": ",",
            "xcol": -1,
            "ycol": 9,
            "use-ewma": 1.0,
        },
        "xlist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ylist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    },
    {
        "name": "postfilter ewma 0.5",
        "parameters": {
            "sep": ",",
            "xcol": -1,
            "ycol": 9,
            "use-ewma": 0.5,
        },
        "xlist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ylist": [
            0.0,
            0.5,
            1.25,
            2.125,
            3.0625,
            4.03125,
            5.015625,
            6.0078125,
            7.00390625,
            8.001953125,
        ],
    },
    {
        "name": "postfilter ewma 0.1",
        "parameters": {
            "sep": ",",
            "xcol": -1,
            "ycol": 9,
            "use-ewma": 0.1,
        },
        "xlist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ylist": [
            0.0,
            0.1,
            0.29,
            0.561,
            0.9049,
            1.31441,
            1.782969,
            2.3046721,
            2.87420489,
            3.4867844,
        ],
    },
    {
        "name": "postfilter xsort",
        "parameters": {
            "sep": ",",
            "xcol": 11,
            "ycol": 9,
            "xsort": True,
        },
        "xlist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ylist": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
    },
    {
        "name": "postfilter ysort",
        "parameters": {
            "sep": ",",
            "xcol": 11,
            "ycol": 9,
            "ysort": True,
        },
        "xlist": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        "ylist": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    },
    # prefilter cases
    {
        "name": "prefilter basic",
        "parameters": {
            "sep": ",",
            "xcol": 1,
            "ycol": 2,
            "prefilter": "5 ne 1",
        },
        "xlist": [1.0, 1.0, 1.0, 1.0, 1.0],
        "ylist": [1.0, 1.0, 1.0, 2.0, 2.0],
    },
    {
        "name": "prefilter arithmetic",
        "parameters": {
            "sep": ",",
            "xcol": 1,
            "ycol": 2,
            "prefilter": "5 gt 1",
        },
        "xlist": [1.0, 1.0, 1.0, 1.0, 1.0],
        "ylist": [1.0, 1.0, 1.0, 2.0, 2.0],
    },
    {
        "name": "prefilter multiple (and)",
        "parameters": {
            "sep": ",",
            "xcol": 1,
            "ycol": 2,
            "prefilter": "5 gt 1 and 2 ne 2",
        },
        "xlist": [1.0, 1.0, 1.0],
        "ylist": [1.0, 1.0, 1.0],
    },
    {
        "name": "prefilter multiple (or)",
        "parameters": {
            "sep": ",",
            "xcol": 1,
            "ycol": 2,
            "prefilter": "5 gt 1 or 2 ne 2",
        },
        "xlist": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "ylist": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
    },
    {
        "name": "prefilter multiple (and/or)",
        "parameters": {
            "sep": ",",
            "xcol": 1,
            "ycol": 2,
            "prefilter": "5 gt 1 and 2 ne 2 or 2 ne 2",
        },
        "xlist": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "ylist": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
    {
        "name": "prefilter contains",
        "parameters": {
            "sep": ",",
            "xcol": 9,
            "ycol": 9,
            "prefilter": "12 contains bar",
        },
        "xlist": [0.0, 1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
        "ylist": [0.0, 1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
    },
    # prefilter cases
    {
        "name": "histogram",
        "parameters": {
            "sep": ",",
            "xcol": 2,
            "histogram": True,
            "histogram-bins": 2,
        },
        "xlist": [1.25, 1.75],
        "ylist": [8.0, 2.0],
    },
    {
        "name": "histogram bins 1",
        "parameters": {
            "sep": ",",
            "xcol": 8,
            "histogram": True,
            "histogram-bins": 2,
        },
        "xlist": [2.0, 6.0],
        "ylist": [4.0, 6.0],
    },
    {
        "name": "histogram bins 2",
        "parameters": {
            "sep": ",",
            "xcol": 8,
            "histogram": True,
            "histogram-bins": 4,
        },
        "xlist": [1.0, 3.0, 5.0, 7.0],
        "ylist": [2.0, 2.0, 2.0, 4.0],
    },
    {
        "name": "histogram pdf 1",
        "parameters": {
            "sep": ",",
            "xcol": 8,
            "histogram": True,
            "histogram-bins": 2,
            "histogram-type": "pdf",
        },
        "xlist": [2.0, 6.0],
        "ylist": [0.4, 0.6],
    },
    {
        "name": "histogram pdf 2",
        "parameters": {
            "sep": ",",
            "xcol": 8,
            "histogram": True,
            "histogram-bins": 4,
            "histogram-type": "pdf",
        },
        "xlist": [1.0, 3.0, 5.0, 7.0],
        "ylist": [0.2, 0.2, 0.2, 0.4],
    },
    {
        "name": "histogram cdf 1",
        "parameters": {
            "sep": ",",
            "xcol": 8,
            "histogram": True,
            "histogram-bins": 2,
            "histogram-type": "cdf",
        },
        "xlist": [2.0, 6.0],
        "ylist": [0.4, 1.0],
    },
    {
        "name": "histogram cdf 2",
        "parameters": {
            "sep": ",",
            "xcol": 8,
            "histogram": True,
            "histogram-bins": 4,
            "histogram-type": "cdf",
        },
        "xlist": [1.0, 3.0, 5.0, 7.0],
        "ylist": [0.2, 0.4, 0.6, 1.0],
    },
    {
        "name": "histogram sigma 1",
        "parameters": {
            "sep": ",",
            "xcol": 10,
            "histogram": True,
            "histogram-bins": 2,
            "histogram-sigma": 2.0,
        },
        "xlist": [2.0, 6.0],
        "ylist": [4.0, 5.0],
    },
    {
        "name": "histogram sigma 2",
        "parameters": {
            "sep": ",",
            "xcol": 10,
            "histogram": True,
            "histogram-bins": 4,
            "histogram-sigma": 2.0,
        },
        "xlist": [1.0, 3.0, 5.0, 7.0],
        "ylist": [2.0, 2.0, 2.0, 3.0],
    },
    {
        "name": "histogram sigma 2b",
        "parameters": {
            "sep": ",",
            "xcol": 10,
            "histogram": True,
            "histogram-bins": 4,
            "histogram-sigma": 20.0,
        },
        "xlist": [12.5, 37.5, 62.5, 87.5],
        "ylist": [9.0, 0.0, 0.0, 1.0],
    },
]

dataMainTestCases = """#col0,col1,col2
0,0,100,
1,10,110,
2,20,120,
3,30,130,
4,40,140,
"""

parseMainTestCases = [
    {
        "name": "basic",
        "argv": (
            '--title "mytitle" --xcol 0 --xlabel "xl1" '
            '--ycol 1 --ylabel "yl1" -i $file --fmt "g." --label "l0" '
            "out.png"
        ),
        "xy_data": [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],  # xlist
                [0.0, 10.0, 20.0, 30.0, 40.0],  # ylist
                "l0",  # label
                "g.",  # fmt
            ],
        ],
    },
    {
        "name": "basic (no x col)",
        "argv": (
            '--title "mytitle" --xcol -1 --xlabel "xl1" '
            '--ycol 1 --ylabel "yl1" -i $file --fmt "g." --label "l0" '
            "out.png"
        ),
        "xy_data": [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],  # xlist
                [0.0, 10.0, 20.0, 30.0, 40.0],  # ylist
                "l0",  # label
                "g.",  # fmt
            ],
        ],
    },
    {
        "name": "basic (named xcol)",
        "argv": (
            '--title "mytitle" --xcol col0 --xlabel "xl1" '
            '--ycol 1 --ylabel "yl1" -i $file --fmt "g." --label "l0" '
            "out.png"
        ),
        "xy_data": [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],  # xlist
                [0.0, 10.0, 20.0, 30.0, 40.0],  # ylist
                "l0",  # label
                "g.",  # fmt
            ],
        ],
    },
    {
        "name": "basic (named ycol)",
        "argv": (
            '--title "mytitle" --xcol 0 --xlabel "xl1" '
            '--ycol col1 --ylabel "yl1" -i $file --fmt "g." --label "l0" '
            "out.png"
        ),
        "xy_data": [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],  # xlist
                [0.0, 10.0, 20.0, 30.0, 40.0],  # ylist
                "l0",  # label
                "g.",  # fmt
            ],
        ],
    },
    {
        "name": "basic (named xcol and ycol)",
        "argv": (
            '--title "mytitle" --xcol col0 --xlabel "xl1" '
            '--ycol col1 --ylabel "yl1" -i $file --fmt "g." --label "l0" '
            "out.png"
        ),
        "xy_data": [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],  # xlist
                [0.0, 10.0, 20.0, 30.0, 40.0],  # ylist
                "l0",  # label
                "g.",  # fmt
            ],
        ],
    },
    {
        "name": "twinx",
        "argv": (
            '--title "mytitle" --xcol 0 --xlabel "xcol" '
            # line 0
            '--ycol 1 --ylabel "yl1" -i $file --fmt "g." --label "l0" '
            "--twinx "
            '--ycol 2 --ylabel "yl2" -i $file --fmt "r." --label "l1" '
            "out.png"
        ),
        "xy_data": [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],  # xlist
                [0.0, 10.0, 20.0, 30.0, 40.0],  # ylist
                "l0",  # label
                "g.",  # fmt
            ],
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],  # xlist
                [100.0, 110.0, 120.0, 130.0, 140.0],  # ylist
                "l1",  # label
                "r.",  # fmt
            ],
        ],
    },
]


batch_data = """#comment
0,foo,file0
1,bar,file1
2,bar,file2
3,bar,file3
4,bar,file4
"""


BatchProcessDataTestCases = [
    {
        "name": "basic",
        "parameters": {
            "sep": ",",
            "col": 2,
            "f": "1 eq bar and 0 lt 3",
        },
        "infile_list": ["file1", "file2"],
    },
]


def statisticsIsClose(d1, d2):
    # make sure keys are the same
    if set(d1.keys()) != set(d2.keys()):
        return False
    # make sure elements are close enough same
    rel_tol = 1e-3
    key_list = d1.keys()
    l1 = [d1[k] for k in key_list]
    l2 = [d2[k] for k in key_list]
    return compareFloatList(l1, l2, rel_tol)


# compares float lists, returning True if they are the same, and False if
# they are not
def compareFloatList(l1, l2, rel_tol=1e-09, abs_tol=0.001):
    # zip() only runs through the smallest list, so we need to check the
    # list sizes first
    if len(l1) != len(l2):
        return False
    same_list = [
        math.isclose(f1, f2, rel_tol=rel_tol, abs_tol=abs_tol) for f1, f2 in zip(l1, l2)
    ]
    # reduce the list
    return all(same_list)


class MyTest(unittest.TestCase):
    def testGetDataBasic(self):
        """Simplest get_data test."""
        for test_case in parseDataTestCases:
            print("...running %s" % test_case["name"])
            argv = []
            for k, v in test_case["parameters"].items():
                if v is False:
                    continue
                if type(v) in (list, tuple):
                    # flatten the list/tuple
                    for index, vv in enumerate(v):
                        if type(vv) not in (list, tuple):
                            if index == 0:
                                argv.append("--%s" % k)
                            # list/tuple of non-lists/non-tuples
                            argv.append(str(vv))
                        else:  # list/tuple of lists/tuples
                            argv.append("--%s" % k)
                            argv += [str(it) for it in vv]
                elif type(v) in (bool,):
                    if v is True:
                        # only append the key name
                        argv.append("--%s" % k)
                else:
                    argv.append("--%s" % k)
                    argv.append("%s" % v)
            # add progname and required args
            argv = (
                [
                    "progname",
                ]
                + argv
                + [
                    "-i",
                    "/dev/null",
                    "outfile",
                ]
            )
            gen_options, plot_line_list = config_lib.get_options(argv)
            plot_pb, _ = plot_line_list[0]
            line_pb = plot_pb.line[0]
            xlist, ylist = plotty_plot.get_data_raw_data(
                dataGetDataTestCases, plot_pb, line_pb, gen_options
            )
            msg = 'unittest failed: "%s"' % test_case["name"]
            self.assertTrue(
                compareFloatList(test_case["xlist"], xlist),
                msg=f'{msg} {test_case["xlist"]} != {xlist}',
            )
            self.assertTrue(
                compareFloatList(test_case["ylist"], ylist),
                msg=f'{msg} {test_case["ylist"]} != {ylist}',
            )

    def testMain(self):
        """Simplest main test (dry-run mode)."""
        for test_case in parseMainTestCases:
            print("...running %s" % test_case["name"])
            argv_str = "./plotty-plot.py " + test_case["argv"] + " --dry-run"
            # replace the "$file" templates with tempfiles
            # https://stackoverflow.com/questions/8300644/

            class TempfileDict(dict):
                tempfile_list = []

                def __init__(self, file_contents):
                    self.file_contents = file_contents

                def __missing__(self, key):
                    if key == "file":
                        fid, name = tempfile.mkstemp(dir="/tmp")
                        with os.fdopen(fid, "w") as f:
                            f.write(self.file_contents)
                        return name
                    raise KeyError(key)

            tempfiles = TempfileDict(dataMainTestCases)
            argv_template = string.Template(argv_str)
            argv_str = argv_template.substitute(tempfiles)
            argv = shlex.split(argv_str)
            xy_data = plotty_plot.main(argv)
            msg = 'unittest failed: "%s"' % test_case["name"]
            # compare the xy_data values
            self.assertTrue(len(test_case["xy_data"]) == len(xy_data), msg)
            for i in range(len(xy_data)):
                expected = test_case["xy_data"][i]
                value = xy_data[i]
                # compare xlist
                self.assertTrue(
                    compareFloatList(expected[0], value[0]),
                    msg=f"{msg} {expected[0]} != {value[0]}",
                )
                # compare ylist
                self.assertTrue(
                    compareFloatList(expected[1], value[1]),
                    msg=f"{msg} {expected[1]} != {value[1]}",
                )
                # compare label, fmt
                line_pb = value[2]
                self.assertTrue(
                    expected[2] == line_pb.label,
                    msg=f"{msg} {expected[2]} != {line_pb.label}",
                )
                self.assertTrue(
                    expected[3] == line_pb.fmt,
                    msg=f"{msg} {expected[3]} != {line_pb.fmt}",
                )

    def _testBatchProcessData(self):
        """Simplest batch_process_data test."""
        for test_case in BatchProcessDataTestCases:
            print("...running %s" % test_case["name"])
            infile_list = plotty_plot.batch_process_data(
                batch_data, **test_case["parameters"]
            )
            msg = 'unittest failed: "%s"' % test_case["name"]
            self.assertEqual(
                test_case["infile_list"],
                infile_list,
                msg=f'{msg} {test_case["infile_list"]} != {infile_list}',
            )


if __name__ == "__main__":
    unittest.main()
