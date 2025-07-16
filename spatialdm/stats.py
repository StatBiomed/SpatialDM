import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors


def rbfweight(X_loc, l=None, cutoff=0.1, n_neighbors=None, n_neighbor_layers=6,
              single_cell=False, eff_dist=None):
    """
    Compute weight matrix based on radial basis function. cutoff & n_neighbors 
    are two alternative options to restrict signaling range.

    :param l: radial basis function parameter, need to be customized for optimal 
        weight gradient and to restrain the range of signaling before downstream 
        processing.
    :param cutoff: (for secreted signaling) minimum weight to be kept from the 
        rbf weight matrix. Weight below cutoff will be made zero
    :param n_neighbors: (for secreted signaling) number of neighbors per spot 
        from the rbf weight matrix.
    :param n_neighbor_layers: (for adjacent signaling) number of neighbor layers 
        per spot from the rbf weight matrix. Non-neighbors will be made 0
    :param single_cell: if single_cell, diagonal will be made 0.
    :param eff_dist: the alternative way to set l parameter by restricting the 
        effective distance.
    :return: secreted signaling weight matrix: W_spatial, and adjacent 
        signaling weight matrix: KNN_connectivities

    Examples
    --------

    .. plot::

        >>> import numpy as np
        >>> from spatialdm.stats import rbfweight
        >>> X_loc = np.vstack([np.repeat(range(10), 10), np.tile(range(10), 10)]).T
        >>> spatial_W, KNN_connect = rbfweight(X_loc, l=1.2, n_neighbors=16)

        >>> import matplotlib.pyplot as plt
        >>> plt.scatter(X_loc[:, 0], X_loc[:, 1], c=spatial_W.toarray()[35], s=100)
        >>> plt.show()
    """
    def _Euclidean_to_RBF(X, l, single_cell=single_cell):
        """Convert Euclidean distance to RBF distance"""
        from scipy.sparse import issparse
        if issparse(X):
            rbf_d = X
            rbf_d[X.nonzero()] = np.exp(-X[X.nonzero()].A**2 / (2 * l ** 2))
        else:
            rbf_d = np.exp(- X**2 / (2 * l ** 2))
        
        # At single-cell resolution, no within-spot communications
        if single_cell:
            # np.fill_diagonal(rbf_d, 0)
            rbf_d.setdiag(np.zeros(rbf_d.shape[0]))
        else:
            rbf_d.setdiag(np.exp(-X.diagonal()**2 / (2 * l ** 2)))

        return rbf_d
    
    if isinstance(X_loc, pd.DataFrame):
        X_loc = X_loc.values

    if n_neighbors is None:
        n_neighbors = n_neighbor_layers * 31

    if l is None:
        if eff_dist is None:
            raise ValueError('At least one of l and eff_dist params should be' +
                ' specified')
        else:
            l = np.sqrt(-eff_dist/(2*np.log(cutoff)))
    ## large neighborhood for W (5 layers)
    nnbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm='ball_tree', 
        metric='euclidean'
    ).fit(X_loc)
    nbr_d = nnbrs.kneighbors_graph(X_loc, mode='distance')
    rbf_d = _Euclidean_to_RBF(nbr_d, l, single_cell)

    ## small neighborhood for RBF
    nnbrs0 = NearestNeighbors(
        n_neighbors=n_neighbor_layers, 
        algorithm='ball_tree', 
        metric='euclidean'
    ).fit(X_loc)
    nbr_d0 = nnbrs0.kneighbors_graph(X_loc, mode='distance')
    rbf_d0 = _Euclidean_to_RBF(nbr_d0, l, single_cell)

    # NOTE: add more info about cutoff, n_neighbors and n_neighbor_layers
    #if cutoff:
        # not efficient
        # rbf_d[rbf_d < cutoff] = 0
        
        # more efficient: 
        # https://seanlaw.github.io/2019/02/27/set-values-in-sparse-matrix/
    nonzero_mask = np.array(rbf_d[rbf_d.nonzero()] < cutoff)[0]
    rows = rbf_d.nonzero()[0][nonzero_mask]
    cols = rbf_d.nonzero()[1][nonzero_mask]
    rbf_d[rows, cols] = 0

    # elif n_neighbors:
    #     nbrs = NearestNeighbors(n_neighbors=n_neighbors, 
    #                             algorithm='ball_tree').fit(rbf_d)
    #     knn = nbrs.kneighbors_graph(rbf_d).toarray()
    #     rbf_d = rbf_d * knn

    spatial_W = rbf_d * X_loc.shape[0] / rbf_d.sum()
    KNN_connectivities = rbf_d0 * X_loc.shape[0] / rbf_d0.sum()
    return spatial_W, KNN_connectivities


# pure statistics for bivariate Moran's R
def Moran_R_std(spatial_W, by_trace=False):
    """Calculate standard deviation of Moran's R under the null distribution.
    """
    N = spatial_W.shape[0]
    
    if by_trace:
        W = spatial_W.copy()
        H = np.identity(N) - np.ones((N, N)) / N
        HWH = H.dot(W.dot(H))
        var = np.trace(HWH.dot(HWH)) * N**2 / (np.sum(W) * (N-1))**2
    else:
        if issparse(spatial_W):
            nm = N ** 2 * spatial_W.multiply(spatial_W.T).sum() \
                - 2 * N * (spatial_W.sum(0) @ spatial_W.sum(1)).sum() \
                + spatial_W.sum() ** 2
        else:
            nm = N ** 2 * (spatial_W * spatial_W.T).sum() \
                - 2 * N * (spatial_W.sum(1) * spatial_W.sum(0)).sum() \
                + spatial_W.sum() ** 2
        dm = N ** 2 * (N - 1) ** 2
        var = nm / dm
    
    return np.sqrt(var)


def Moran_R(X, Y, spatial_W, standardise=True, nproc=1):
    """Computing Moran's R for pairs of variables
    
    :param X: Variable 1, (n_sample, n_variables) or (n_sample, )
    :param Y: Variable 2, (n_sample, n_variables) or (n_sample, )
    :param spatial_W: spatial weight matrix, sparse or dense, (n_sample, n_sample)
    :param nproc: default to 1. Numpy may use more without much speedup.
    
    :return: (Moran's R, z score and p values)

    Examples
    --------

    .. plot::

        >>> import numpy as np
        >>> from spatialdm.stats import rbfweight, Moran_R
        >>> np.random.seed(0)
        >>> X1 = np.random.rand(100, 5)
        >>> X2 = np.random.rand(100, 5)
        >>> X2[:-1, 0], X2[:, 1], X2[1:, 2] = X1[1:, 0], X1[:, 1], X1[:-1, 2]
        >>> X2 = X2 + 0.01 * np.random.rand(100, 5)
        >>> X_loc = np.vstack([np.repeat(range(10), 10), np.tile(range(10), 10)]).T
        >>> spatial_W, KNN_connect = rbfweight(X_loc, l=1.2, n_neighbors=16)
        >>> R, z, p = Moran_R(X1, X2, spatial_W)
        >>> print(R, z, p)
        
        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure(figsize=(6, 3))
        >>> plt.subplot(1, 2, 1)
        >>> plt.scatter(X_loc[:, 0], X_loc[:, 1], c=X1[:, 0], s=100)
        >>> plt.subplot(1, 2, 2)
        >>> plt.scatter(X_loc[:, 0], X_loc[:, 1], c=X2[:, 0], s=100)
        >>> plt.tight_layout()
        >>> plt.show()
    """
    if len(X.shape) < 2:
        X = X.reshape(-1, 1)
    if len(Y.shape) < 2:
        Y = Y.reshape(-1, 1)
        
    if standardise:
        X = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)
        Y = (Y - np.mean(Y, axis=0, keepdims=True)) / np.std(Y, axis=0, keepdims=True)
        
    # Consider to dense array for speedup (numpy's codes is optimised)
    if X.shape[0] <= 5000 and issparse(spatial_W):
        # Note, numpy may use unnessary too many threads
        # You may use threadpool.threadpool_limits() outside
        from threadpoolctl import threadpool_limits
        
        with threadpool_limits(limits=nproc, user_api='blas'):
            R_val = (spatial_W.toarray() @ X * Y).sum(axis=0) / np.sum(spatial_W)
    else:
        # we assume it's sparse spatial_W when sample size > 5000
        R_val = (spatial_W @ X * Y).sum(axis=0) / np.sum(spatial_W)
        
    _R_std = Moran_R_std(spatial_W)
    R_z_score = R_val / _R_std
    R_p_val = stats.norm.sf(R_z_score)
    
    return R_val, R_z_score, R_p_val
