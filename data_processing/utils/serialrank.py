"""
Class to run the Serial Rank algorithm
Copied from https://github.com/Chau999/SpectralRankingWithCovariates/tree/main [1], original algorithm from [2]

[1] Chau, SL. (2020). Spectral Ranking With Covariates. GitHub. Retrieved 1
[2] Fogel, F. (2014). SerialRank: Spectral Ranking using Seriation. 
"""

from numpy import sign, count_nonzero, ones, shape, reshape, eye, \
    dot, argsort, allclose, concatenate, diag
from numpy.linalg import eig
from numpy.linalg import eigh



def compute_upsets(r, C, verbose=True, which_method=""):
    n = shape(r)[0]
    totmatches = count_nonzero(C) / 2
    if (len(shape(r)) == 1):
        r = reshape(r, (n, 1))
    e = ones((n, 1))
    Chat = r.dot(e.T) - e.dot(r.T)
    upsetsplus = count_nonzero(sign(Chat[C != 0]) != sign(C[C != 0]))
    upsetsminus = count_nonzero(sign(-Chat[C != 0]) != sign(C[C != 0]))
    winsign = 2 * (upsetsplus < upsetsminus) - 1
    if (verbose):
        print(which_method + " upsets(+): %.4f" %
              (upsetsplus / float(2 * totmatches)))
        print(which_method + " upsets(-): %.4f" %
              (upsetsminus / float(2 * totmatches)))
    return upsetsplus / float(2 * totmatches), upsetsminus / float(2 * totmatches), winsign


def centering_matrix(n):
    # centering matrix, projection to the subspace orthogonal
    # to all-ones vector
    return eye(n) - ones((n, n)) / n

def Compute_Sim(C):
    """
    Compute the Similarity matrix
    """
    n = C.shape[0]
    ones_mat = n * dot(ones(n).reshape(-1, 1), ones(n).reshape(1, -1))
    S = 0.5 * (ones_mat + dot(C, C.T))
    return S

def get_the_subspace_basis(n, verbose=True):
    # returns the orthonormal basis of the subspace orthogonal
    # to all-ones vector
    H = centering_matrix(n)
    s, Zp = eigh(H)
    ind = argsort(-s)  # order eigenvalues descending
    s = s[ind]
    Zp = Zp[:, ind]  # second axis !!
    if (verbose):
        print("...forming the Z-basis")
        print("check eigenvalues: ", allclose(
            s, concatenate((ones(n - 1), [0]), 0)))

    Z = Zp[:, :(n - 1)]
    if (verbose):
        print("check ZZ'=H: ", allclose(dot(Z, Z.T), H))
        print("check Z'Z=I: ", allclose(dot(Z.T, Z), eye(n - 1)))
    return Z

def GraphLaplacian(G):
    """
    Input a simlarity graph G and return graph GraphLaplacian
    """
    D = diag(G.sum(axis=1))
    L = D - G

    return L

class SerialRank:
    """
    A class that runs the Serial Rank algorithm

    The C matrix is a binarised comparison matrix.
    """
    def __init__(self, C):
        self.C = C

    def fit(self):
        # First compute similarity matrix
        S = Compute_Sim(self.C)
        n = S.shape[0]
        Z = get_the_subspace_basis(n, verbose=False)

        Ls = GraphLaplacian(S)
        ztLsz = dot(dot(Z.T, Ls), Z)
        w, v = eig(ztLsz)

        ind = argsort(w)
        v = v[:, ind]
        r = reshape(Z.dot(v[:, 0]), (n, 1))

        _, _, rsign = compute_upsets(r, S, verbose=False)

        self.r = rsign * r