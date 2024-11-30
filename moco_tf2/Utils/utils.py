import os
import random
from collections import defaultdict

import numpy as np


def chebyshev_distance(A: np.ndarray, B: np.ndarray):
    if A is None or B is None:
        return 0.0
    if A.shape != B.shape:
        return 100
    else:
        return float(np.max(np.abs(A - B)))


def get_input_shape_str(model_name):
    if model_name in ["lenet"]:
        return "[1, 28, 28, 1]"
    elif model_name in ["pointnet", "lstm"]:
        return "[2, 2048, 3]"
    else:
        return "[1, 224, 224, 3]"


def get_max_number(str):
    number = 1
    try:
        if "(" in str or "[" in str:
            str = str[1:-1]
        l = str.split(", ")
        s = [int(_) for _ in l]
        number = max(s)
    except:
        pass
    return number


def cut(nodes, limit):
    buckets = defaultdict(list)
    for node in nodes:
        buckets[node.mutate_info].append(node)

    for bucket in buckets.values():
        bucket.sort(key=lambda x: x.weight)

    truncated = []
    for bucket in buckets.values():
        cutoff = len(bucket) // 2 + 1
        truncated.extend(bucket[:cutoff])

    if len(truncated) > limit:
        result = []
        while len(result) < limit:
            for bucket in buckets.values():
                if bucket and len(result) < limit:
                    result.append(bucket.pop(0))
        return result
    else:
        return truncated
