#!/usr/bin/env python3

"""utils.py: common utils.
"""


def is_int(s):
    if isinstance(s, int):
        return True
    return s[1:].isdigit() if s[0] in ("-", "+") else s.isdigit()
