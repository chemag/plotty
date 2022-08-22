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

config_lib = importlib.import_module('plotty-config')
plotty_plot = importlib.import_module('plotty-plot')


dataParseDataTestCases = """ # comment
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
        'statistics': {'median': 1.0, 'mean': 1.2, 'stddev': 0.4}
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
        'statistics': {'median': 1.0, 'mean': 1.0, 'stddev': 0.0}
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
        'statistics': {'median': 1.0, 'mean': 1.2, 'stddev': 0.4}
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
        'statistics': {'median': 4.5, 'mean': 4.4, 'stddev': 2.72764}
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
        'statistics': {'median': 0.0, 'mean': 0.1, 'stddev': 0.3}
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
        'statistics': {'median': 5.5, 'mean': 5.8, 'stddev': 3.34066}
    },
    {
        'name': 'prefilter',
        'parameters': {
            'sep': ',',
            'xcol': 1,
            'ycol': 2,
            'prefilter': '5 ne 1',
        },
        'xlist': [1.0, 1.0, 1.0, 1.0, 1.0],
        'ylist': [1.0, 1.0, 1.0, 2.0, 2.0],
        'statistics': {'median': 1.0, 'mean': 1.4, 'stddev': 0.4899}
    },
    {
        'name': 'prefilter arithmetic',
        'parameters': {
            'sep': ',',
            'xcol': 1,
            'ycol': 2,
            'prefilter': '5 gt 1',
        },
        'xlist': [1.0, 1.0, 1.0, 1.0, 1.0],
        'ylist': [1.0, 1.0, 1.0, 2.0, 2.0],
        'statistics': {'median': 1.0, 'mean': 1.4, 'stddev': 0.4899}

    },
    {
        'name': 'prefilter multiple (and)',
        'parameters': {
            'sep': ',',
            'xcol': 1,
            'ycol': 2,
            'prefilter': '5 gt 1 and 2 ne 2',
        },
        'xlist': [1.0, 1.0, 1.0],
        'ylist': [1.0, 1.0, 1.0],
        'statistics': {'median': 1.0, 'mean': 1.0, 'stddev': 0.0}
    },
    {
        'name': 'prefilter multiple (or)',
        'parameters': {
            'sep': ',',
            'xcol': 1,
            'ycol': 2,
            'prefilter': '5 gt 1 or 2 ne 2',
        },
        'xlist': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'ylist': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
        'statistics': {'median': 1.0, 'mean': 1.2, 'stddev': 0.4}
    },
    {
        'name': 'prefilter multiple (and/or)',
        'parameters': {
            'sep': ',',
            'xcol': 1,
            'ycol': 2,
            'prefilter': '5 gt 1 and 2 ne 2 or 2 ne 2',
        },
        'xlist': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'ylist': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'statistics': {'median': 1.0, 'mean': 1.0, 'stddev': 0.0}
    },
    {
        'name': 'histogram',
        'parameters': {
            'sep': ',',
            'xcol': 2,
            'histogram': True,
            'histogram-bins': 2,
        },
        'xlist': [1.25, 1.75],
        'ylist': [8.0, 2.0],
        'statistics': {'median': 1.0, 'mean': 1.2, 'stddev': 0.4}
    },
    {
        'name': 'histogram bins 1',
        'parameters': {
            'sep': ',',
            'xcol': 8,
            'histogram': True,
            'histogram-bins': 2,
        },
        'xlist': [2.0, 6.0],
        'ylist': [4.0, 6.0],
        'statistics': {'median': 4.5, 'mean': 4.4, 'stddev': 2.72764}

    },
    {
        'name': 'histogram bins 2',
        'parameters': {
            'sep': ',',
            'xcol': 8,
            'histogram': True,
            'histogram-bins': 4,
        },
        'xlist': [1.0, 3.0, 5.0, 7.0],
        'ylist': [2.0, 2.0, 2.0, 4.0],
        'statistics': {'median': 4.5, 'mean': 4.4, 'stddev': 2.72764}
    },
    {
        'name': 'histogram pdf 1',
        'parameters': {
            'sep': ',',
            'xcol': 8,
            'histogram': True,
            'histogram-bins': 2,
            'histogram-type': 'pdf',
        },
        'xlist': [2.0, 6.0],
        'ylist': [0.4, 0.6],
        'statistics': {'median': 4.5, 'mean': 4.4, 'stddev': 2.72764}
    },
    {
        'name': 'histogram pdf 2',
        'parameters': {
            'sep': ',',
            'xcol': 8,
            'histogram': True,
            'histogram-bins': 4,
            'histogram-type': 'pdf',
        },
        'xlist': [1.0, 3.0, 5.0, 7.0],
        'ylist': [0.2, 0.2, 0.2, 0.4],
        'statistics': {'median': 4.5, 'mean': 4.4, 'stddev': 2.72764}
    },
    {
        'name': 'histogram cdf 1',
        'parameters': {
            'sep': ',',
            'xcol': 8,
            'histogram': True,
            'histogram-bins': 2,
            'histogram-type': 'cdf',
        },
        'xlist': [2.0, 6.0],
        'ylist': [0.4, 1.0],
        'statistics': {'median': 4.5, 'mean': 4.4, 'stddev': 2.72764}
    },
    {
        'name': 'histogram cdf 2',
        'parameters': {
            'sep': ',',
            'xcol': 8,
            'histogram': True,
            'histogram-bins': 4,
            'histogram-type': 'cdf',
        },
        'xlist': [1.0, 3.0, 5.0, 7.0],
        'ylist': [0.2, 0.4, 0.6, 1.0],
        'statistics': {'median': 4.5, 'mean': 4.4, 'stddev': 2.72764}
    },
    {
        'name': 'histogram sigma 1',
        'parameters': {
            'sep': ',',
            'xcol': 9,
            'histogram': True,
            'histogram-bins': 2,
            'histogram-sigma': 2.0,
        },
        'xlist': [2.0, 6.0],
        'ylist': [4.0, 5.0],
        'statistics': {'median': 4.5, 'mean': 13.6, 'stddev': 28.904}
    },
    {
        'name': 'histogram sigma 2',
        'parameters': {
            'sep': ',',
            'xcol': 9,
            'histogram': True,
            'histogram-bins': 4,
            'histogram-sigma': 2.0,
        },
        'xlist': [1.0, 3.0, 5.0, 7.0],
        'ylist': [2.0, 2.0, 2.0, 3.0],
        'statistics': {'median': 4.5, 'mean': 13.6, 'stddev': 28.904}
    },
    {
        'name': 'histogram sigma 2b',
        'parameters': {
            'sep': ',',
            'xcol': 9,
            'histogram': True,
            'histogram-bins': 4,
            'histogram-sigma': 20.0,
        },
        'xlist': [12.5, 37.5, 62.5, 87.5],
        'ylist': [9.0, 0.0, 0.0, 1.0],
        'statistics': {'median': 4.5, 'mean': 13.6, 'stddev': 28.904}
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
        'name': 'basic',
        'argv': ('--title "mytitle" --xcol 0 --xlabel "xl1" '
                 '--ycol 1 --ylabel "yl1" -i $file --fmt "g." --label "l0" '
                 'out.png'),
        'xy_data': [
            [[0.0, 1.0, 2.0, 3.0, 4.0],  # xlist
             [0.0, 10.0, 20.0, 30.0, 40.0],  # ylist
             {},  # statistics
             'l0',  # label
             'g.',  # fmt
             None,  # color
             ],
        ],
    },
    {
        'name': 'basic (no x col)',
        'argv': ('--title "mytitle" --xcol -1 --xlabel "xl1" '
                 '--ycol 1 --ylabel "yl1" -i $file --fmt "g." --label "l0" '
                 'out.png'),
        'xy_data': [
            [[0.0, 1.0, 2.0, 3.0, 4.0],  # xlist
             [0.0, 10.0, 20.0, 30.0, 40.0],  # ylist
             {},  # statistics
             'l0',  # label
             'g.',  # fmt
             None,  # color
             ],
        ],
    },
    {
        'name': 'basic (named xcol)',
        'argv': ('--title "mytitle" --xcol col0 --xlabel "xl1" '
                 '--ycol 1 --ylabel "yl1" -i $file --fmt "g." --label "l0" '
                 'out.png'),
        'xy_data': [
            [[0.0, 1.0, 2.0, 3.0, 4.0],  # xlist
             [0.0, 10.0, 20.0, 30.0, 40.0],  # ylist
             {},  # statistics
             'l0',  # label
             'g.',  # fmt
             None,  # color
             ],
        ],
    },
    {
        'name': 'basic (named ycol)',
        'argv': ('--title "mytitle" --xcol 0 --xlabel "xl1" '
                 '--ycol col1 --ylabel "yl1" -i $file --fmt "g." --label "l0" '
                 'out.png'),
        'xy_data': [
            [[0.0, 1.0, 2.0, 3.0, 4.0],  # xlist
             [0.0, 10.0, 20.0, 30.0, 40.0],  # ylist
             {},  # statistics
             'l0',  # label
             'g.',  # fmt
             None,  # color
             ],
        ],
    },
    {
        'name': 'basic (named xcol and ycol)',
        'argv': ('--title "mytitle" --xcol col0 --xlabel "xl1" '
                 '--ycol col1 --ylabel "yl1" -i $file --fmt "g." --label "l0" '
                 'out.png'),
        'xy_data': [
            [[0.0, 1.0, 2.0, 3.0, 4.0],  # xlist
             [0.0, 10.0, 20.0, 30.0, 40.0],  # ylist
             {},  # statistics
             'l0',  # label
             'g.',  # fmt
             None,  # color
             ],
        ],
    },
    {
        'name': 'twinx',
        'argv': ('--title "mytitle" --xcol 0 --xlabel "xcol" '
                 # line 0
                 '--ycol 1 --ylabel "yl1" -i $file --fmt "g." --label "l0" '
                 '--twinx '
                 '--ycol 2 --ylabel "yl2" -i $file --fmt "r." --label "l1" '
                 'out.png'),
        'xy_data': [
            [[0.0, 1.0, 2.0, 3.0, 4.0],  # xlist
             [0.0, 10.0, 20.0, 30.0, 40.0],  # ylist
             {},  # statistics
             'l0',  # label
             'g.',  # fmt
             None,  # color
             ],
            [[0.0, 1.0, 2.0, 3.0, 4.0],  # xlist
             [100.0, 110.0, 120.0, 130.0, 140.0],  # ylist
             {},  # statistics
             'l1',  # label
             'r.',  # fmt
             None,  # color
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
        'name': 'basic',
        'parameters': {
            'sep': ',',
            'col': 2,
            'f': '1 eq bar and 0 lt 3',
        },
        'infile_list': ['file1', 'file2']
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
def compareFloatList(l1, l2, rel_tol=1e-09):
    # zip() only runs through the smallest list, so we need to check the
    # list sizes first
    if len(l1) != len(l2):
        return False
    same_list = [math.isclose(f1, f2, rel_tol=rel_tol) for f1, f2 in
                 zip(l1, l2)]
    # reduce the list
    return all(same_list)


class MyTest(unittest.TestCase):

    def testParseDataBasic(self):
        """Simplest parse_data test."""
        xshift = 0
        yshift = 0
        for test_case in parseDataTestCases:
            print('...running %s' % test_case['name'])
            argv = []
            for k, v in test_case['parameters'].items():
                if v is False:
                    continue
                if type(v) in (list, tuple):
                    # flatten the list/tuple
                    for index, vv in enumerate(v):
                        if type(vv) not in (list, tuple):
                            if index == 0:
                                argv.append('--%s' % k)
                            # list/tuple of non-lists/non-tuples
                            argv.append(str(vv))
                        else:  # list/tuple of lists/tuples
                            argv.append('--%s' % k)
                            argv += [str(it) for it in vv]
                elif type(v) in (bool, ):
                    if v is True:
                        # only append the key name
                        argv.append('--%s' % k)
                else:
                    argv.append('--%s' % k)
                    argv.append('%s' % v)
            # add progname and required args
            argv = ['progname', ] + argv + ['-i', '/dev/null', 'outfile', ]
            options = config_lib.get_options(argv)
            ycol = options.ycol[0] if options.ycol else None
            prefilter = options.prefilter[0] if options.prefilter else None
            xlist, ylist, statistics = plotty_plot.parse_data(
                dataParseDataTestCases, ycol, xshift, yshift, prefilter,
                options)
            msg = 'unittest failed: %s' % test_case['name']
            self.assertTrue(compareFloatList(test_case['xlist'], xlist),
                            msg=msg)
            self.assertTrue(compareFloatList(test_case['ylist'], ylist),
                            msg=msg)
            self.assertTrue(statisticsIsClose(test_case['statistics'],
                                              statistics), msg=msg)

    def testMain(self):
        """Simplest main test (dry-run mode)."""
        for test_case in parseMainTestCases:
            print('...running %s' % test_case['name'])
            argv_str = './plotty-plot.py ' + test_case['argv'] + ' --dry-run'
            # replace the "$file" templates with tempfiles
            # https://stackoverflow.com/questions/8300644/

            class TempfileDict(dict):
                tempfile_list = []

                def __init__(self, file_contents):
                    self.file_contents = file_contents

                def __missing__(self, key):
                    if key == 'file':
                        fid, name = tempfile.mkstemp(dir='/tmp')
                        with os.fdopen(fid, 'w') as f:
                            f.write(self.file_contents)
                        return name
                    raise KeyError(key)

            tempfiles = TempfileDict(dataMainTestCases)
            argv_template = string.Template(argv_str)
            argv_str = argv_template.substitute(tempfiles)
            argv = shlex.split(argv_str)
            xy_data = plotty_plot.main(argv)
            msg = 'unittest failed: %s' % test_case['name']
            # compare the xy_data values
            self.assertTrue(len(test_case['xy_data']) == len(xy_data), msg)
            for i in range(len(xy_data)):
                expected = test_case['xy_data'][i]
                value = xy_data[i]
                # compare xlist
                self.assertTrue(compareFloatList(expected[0], value[0]),
                                msg=msg)
                # compare ylist
                self.assertTrue(compareFloatList(expected[1], value[1]),
                                msg=msg)
                # skip statistics
                # compare label, fmt, color

                self.assertTrue(expected[3:] == value[3:], msg=msg)

    def testBatchProcessData(self):
        """Simplest batch_process_data test."""
        for test_case in BatchProcessDataTestCases:
            print('...running %s' % test_case['name'])
            infile_list = plotty_plot.batch_process_data(
                batch_data, **test_case['parameters'])
            msg = 'unittest failed: %s' % test_case['name']
            self.assertEqual(test_case['infile_list'], infile_list, msg=msg)


if __name__ == '__main__':
    unittest.main()
