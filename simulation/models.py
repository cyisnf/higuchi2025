import numpy as np


def mutual_information(arr_ct):
    row, col = arr_ct.shape
    prob_ct = arr_ct / np.sum(arr_ct)

    h_x = 0  # H(X)
    marg_by_col = np.sum(prob_ct, axis=1)
    for row_i in range(row):
        h_x += -1 * marg_by_col[row_i] * np.log2(marg_by_col[row_i])

    h_y = 0  # H(Y)
    marg_by_row = np.sum(prob_ct, axis=0)
    for col_i in range(col):
        h_y += -1 * marg_by_row[col_i] * np.log2(marg_by_row[col_i])

    h_xy = 0  # H(X,Y)
    for row_i in range(row):
        for col_i in range(col):
            h_xy += -1 * prob_ct[row_i, col_i] * np.log2(prob_ct[row_i, col_i])

    mi = 0  # I(X;Y)
    for row_i in range(row):
        for col_i in range(col):
            mi += prob_ct[row_i, col_i] * np.log2(
                prob_ct[row_i, col_i] / (marg_by_col[row_i] * marg_by_row[col_i])
            )

    return mi


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
    ct = ct.reshape(3, 2)
    # row, col = ct.shape
    a, b, m, n, c, _ = (
        ct[0, 0],
        ct[0, 1],
        ct[1, 0],
        ct[1, 1],
        ct[2, 0],
        ct[2, 1],
    )
    return (a + m) / (a + m + n + b + c)
