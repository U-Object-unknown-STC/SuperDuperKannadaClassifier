import torch
import numpy as np


def euclidean(x1, x2):
    return ((x1 - x2) ** 2).sum().sqrt()


def k_moment(output, k):
    # output: list of outputs
    # 0 for target, 1, 2, ..., for source
    num_domain = len(output)
    for i in range(num_domain):
        output[i] = (output[i] ** k).mean(0)

    # k moment
    k_moment_dist = 0
    for i in range(num_domain):
        for j in range(i + 1, num_domain):
            k_moment_dist += euclidean(output[i], output[j])
    return k_moment_dist
