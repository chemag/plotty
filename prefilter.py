#!/usr/bin/env python3

"""prefilter.py: Prefilter implementation.
"""


import importlib

utils = importlib.import_module("utils")


# prefilter processing
class Prefilter:
    # prefilter ops
    VALID_FILTER_OPS = "eq", "ne", "gt", "ge", "lt", "le", "contains"
    VALID_BOOL_OPS = "and", "or"

    def __init__(self, string):
        self.item_list = []
        self.bool_list = []
        self.string = string
        if self.string is None:
            # empty prefilter
            return
        # PREFILTER-SPEC := ITEM [BOP ITEM]*
        # ITEM := FCOL FOP FVAL
        # where
        #   ITEM: a 3-element function returning a boolean
        #   FCOL: column ID (number or column name)
        #   FOP: valid prefilter operation
        #   FVAL: value
        #   BOP: valid bool operation
        # notes
        #   * BOP evaluation is left-to-right. This means "A and B or C"
        #     is evaluated as "(A and B) or C"
        #   * Adding parenthesis/NOT operation to BOP will require a more
        #     complex approach (tree structure). Not worth it yet.
        f = self.string.split()
        if len(f) == 0:
            # empty prefilter
            self.string = None
            return
        # break the prefilter in group of 3x items
        assert len(f) % 4 == 3, f'incorrect num of elements in "{self.string}"'
        item = f[0 : 0 + 3]
        self.assert_item(*item)
        self.item_list.append(item)
        i = 3
        while i < len(f):
            bop = f[i]
            self.assert_bool(bop)
            self.bool_list.append(bop)
            item = f[i + 1 : i + 4]
            self.assert_item(*item)
            self.item_list.append(item)
            i += 4

    def assert_bool(self, bop):
        assert bop in self.VALID_BOOL_OPS, (
            f'invalid bool operation ("{bop}") in prefilter. '
            f"Options: {self.VALID_BOOL_OPS}"
        )

    def assert_item(self, fcol, fop, fval):
        assert fop in self.VALID_FILTER_OPS, (
            f'invalid prefilter op ("{fop}") in "{fcol} {fop} {fval}". '
            f"Options: {self.VALID_FILTER_OPS}"
        )

    def fix_columns(self, column_names):
        if self.string is None:
            return
        new_item_list = []
        for fcol, fop, fval in self.item_list:
            if utils.is_int(fcol):
                fcol = int(fcol)
            else:
                # look for named columns
                assert fcol in column_names, (
                    f'error: invalid fcol name: "{fcol}" '
                    f"(column_names: {column_names})"
                )
                fcol = column_names.index(fcol)
            new_item_list.append([fcol, fop, fval])
        self.item_list = new_item_list

    def match_item(self, item, val_list):
        fcol, fop, fval = item
        lval = val_list[fcol]
        # implement eq and ne
        if fop in ("eq", "ne"):
            if (fop == "eq" and lval != fval) or (fop == "ne" and lval == fval):
                return False
        # implement gt, ge, lt, le
        elif fop in ("gt", "ge", "lt", "le"):
            # make sure line val and prefilter val are numbers
            try:
                lval = float(lval)
            except ValueError:
                # comparisons with no data are always False
                if lval == "":
                    return False
                # support for invalid comparisons in "False and <invalid>"
                # and "True or <invalid>" cases
                return "invalid"
            fval = float(fval)
            if (
                (fop == "ge" and lval < fval)
                or (fop == "gt" and lval <= fval)
                or (fop == "le" and lval > fval)
                or (fop == "lt" and lval >= fval)
            ):
                return False
        # implement contains
        elif fop in ("contains"):
            # make sure line val and prefilter val are strings
            lval = str(lval)
            fval = str(fval)
            return fval in lval
        return True

    def run_bool_op(self, bool_op, val1, val2):
        # support "invalid" cases
        if bool_op == bool.__or__ and (val1 == True or val2 == True):
            # True or <x> = True
            return True
        elif bool_op == bool.__and__ and (val1 == False or val2 == False):
            # False and <x> = False
            return False
        return bool_op(val1, val2)

    def match_line(self, line, sep):
        if self.string is None:
            return True
        val_list = line.split(sep)
        # run all the items
        item_vals = [self.match_item(item, val_list) for item in self.item_list]
        # coalesce them using the boolean ops
        i = 0
        ret = item_vals[i]
        for bop in self.bool_list:
            bool_op = bool.__and__ if bop == "and" else bool.__or__
            i += 1
            ret = self.run_bool_op(bool_op, ret, item_vals[i])
        return ret
