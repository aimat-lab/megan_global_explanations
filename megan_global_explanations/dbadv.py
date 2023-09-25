import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.stats as st

undefined = False


def binary_search_perplexity(X, perplexity):
    n = len(X)
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    beta = np.ones((n, 1))
    logU = np.log(perplexity)
    tol = 1e-5

    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]

        P = np.exp(-Di.copy() * beta[i])
        sumP = sum(P)
        H = np.log(sumP) + beta[i] * np.sum(Di * P) / sumP

        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            P = np.exp(-Di.copy() * beta[i])
            sumP = sum(P)
            H = np.log(sumP) + beta[i] * np.sum(Di * P) / sumP
            Hdiff = H - logU
            tries += 1

    sigma = np.sqrt(1 / beta)
    return sigma


def DBADV(X, perplexity, MinPts, probability, metric='euclidean'):
    c = -1
    n = len(X)
    label = [undefined] * n
    D = squareform(pdist(X, metric))
    sigma = binary_search_perplexity(X, perplexity)

    F = np.zeros((n, 1))
    for i in range(n):
        F[i] = st.norm.ppf(probability, loc=0, scale=sigma[i])

    F_i = np.repeat(F, n).reshape(n, n)
    F_j = np.transpose(F_i)
    neighborhoods_mutual = (D <= F_i) & (D <= F_j)

    for i in range(n):
        if label[i] is not undefined:
            continue
        neighbors1 = np.where(neighborhoods_mutual[i])[0]
        if len(neighbors1) < MinPts:
            label[i] = -1
            continue
        c = c + 1
        label[i] = c
        Seed = list(neighbors1)
        Seed_set = set(neighbors1)
        Seed.remove(i)
        for j in Seed:
            if label[j] == -1:
                label[j] = c
            if label[j] is not undefined:
                continue
            label[j] = c
            neighbors2 = np.where(neighborhoods_mutual[j])[0]
            neighbors2_set = set(neighbors2)
            if len(neighbors2) < MinPts:
                continue
            diff_set = neighbors2_set - Seed_set
            Seed_set = Seed_set.union(neighbors2_set)
            diff = list(diff_set)
            diff.sort()
            Seed.extend(diff)
            
    return label