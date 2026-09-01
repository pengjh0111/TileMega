#!/usr/bin/env python3
"""Evaluate the V-F static layouts with CUTLASS pycute."""

import json

from pycute import (
    Layout,
    coalesce,
    composition,
    flatten,
    left_inverse,
    logical_divide,
    right_inverse,
    shape_div,
    zipped_divide,
)


def describe(layout):
    return {"shape": str(layout.shape), "stride": str(layout.stride), "text": str(layout)}


def main():
    outer = Layout((4, 8), (1, 4))
    inner = Layout((2, 4), (1, 2))
    source = Layout((4, 3), (1, 4))
    row_major = Layout((4, 3), (3, 1))
    divided = Layout((12, 8), (1, 12))
    dialect_divided = Layout((6, 8), (8, 1))
    tiler = (3, 4)
    nested = Layout((3, (4, 5)), (8, (1, 4)))
    flat_nested = Layout(flatten(nested.shape), flatten(nested.stride))
    report = {
        "composition": describe(composition(outer, inner)),
        "right_inverse": describe(right_inverse(source)),
        "left_inverse": describe(left_inverse(row_major)),
        "logical_divide": describe(logical_divide(divided, tiler)),
        "zipped_divide": describe(zipped_divide(divided, tiler)),
        "flatten": describe(flat_nested),
        "coalesce": describe(coalesce(flat_nested)),
        "shape_div": str(shape_div((12, 8), (3, 4))),
        "ceil_div_equivalent": str((13 + 4 - 1) // 4),
        "dialect_shared_cases": {
            "logical_divide": describe(logical_divide(dialect_divided, tiler)),
            "zipped_divide": describe(zipped_divide(dialect_divided, tiler)),
        },
        "dynamic_support": {
            "supported": False,
            "reason": "pycute.typing.Integer accepts subclasses of int only; symbolic extents are not part of its public value domain",
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
