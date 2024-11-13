import numpy as np


def paris_mean(ct):
    terms = []
    row, col = ct.shape
    for r in range(row - 1):
        for c in range(col - 1):
            if r == row - 1 and c == col - 1:
                pass
            product = np.sum(ct[r, c])
            terms.append((product / (np.sum(ct[r, :]) + np.sum(ct[:, c]) - product)))
    return sum(terms) / 2


def paris_weighted_mean(ct, weights):
    terms = []
    row, col = ct.shape
    for r in range(row - 1):
        for c in range(col - 1):
            if r == row - 1 and c == col - 1:
                pass
            product = np.sum(ct[r, c])
            terms.append((product / (np.sum(ct[r, :]) + np.sum(ct[:, c]) - product)))
    return sum(np.array(terms) * np.array(weights))


def paris_max(ct):
    terms = []
    row, col = ct.shape
    for r in range(row - 1):
        for c in range(col - 1):
            if r == row - 1 and c == col - 1:
                pass
            product = np.sum(ct[r, c])
            terms.append((product / (np.sum(ct[r, :]) + np.sum(ct[:, c]) - product)))
    return max(terms)


def paris_min(ct):
    terms = []
    row, col = ct.shape
    for r in range(row - 1):
        for c in range(col - 1):
            if r == row - 1 and c == col - 1:
                pass
            product = np.sum(ct[r, c])
            terms.append((product / (np.sum(ct[r, :]) + np.sum(ct[:, c]) - product)))
    return min(terms)


def paris_term1(ct):
    terms = []
    row, col = ct.shape
    for r in range(row - 1):
        for c in range(col - 1):
            if r == row - 1 and c == col - 1:
                pass
            product = np.sum(ct[r, c])
            terms.append((product / (np.sum(ct[r, :]) + np.sum(ct[:, c]) - product)))
    return terms[0]


def paris_term2(ct):
    terms = []
    row, col = ct.shape
    for r in range(row - 1):
        for c in range(col - 1):
            if r == row - 1 and c == col - 1:
                pass
            product = np.sum(ct[r, c])
            terms.append((product / (np.sum(ct[r, :]) + np.sum(ct[:, c]) - product)))
    return terms[1]


def paris_merge(ct):
    product = ct[0:-1, 0:-1].sum()
    sum_ = ct.sum() - ct[-1, -1]
    return product / sum_
