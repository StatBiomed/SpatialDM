"""
Utils of permutation calculation
"""
import pandas as pd
import numpy as np
import random
from scipy import stats
import time
from tqdm import tqdm
from scipy.sparse import csc_matrix, csr_matrix, issparse, hstack

# To be deprecated in the future
from .stats import Moran_R, Moran_R_std

# global variance
def globle_st_compute(adata):
    N = adata.shape[0]

    if issparse(adata.obsp['weight']):
        nm = N ** 2 * adata.obsp['weight'].multiply(adata.obsp['weight'].T).sum() \
            - 2 * N * (adata.obsp['weight'].sum(0) @ adata.obsp['weight'].sum(1)).sum() \
            + adata.obsp['weight'].sum() ** 2
        nm0 = N ** 2 * adata.obsp['nearest_neighbors'].multiply(adata.obsp['nearest_neighbors'].T).sum() \
            - 2 * N * (adata.obsp['nearest_neighbors'].sum(0) @ adata.obsp['nearest_neighbors'].sum(1)).sum() \
            + adata.obsp['nearest_neighbors'].sum() ** 2
    else:
        nm = N ** 2 * (adata.obsp['weight'] * adata.obsp['weight'].T).sum() \
            - 2 * N * (adata.obsp['weight'].sum(1) * adata.obsp['weight'].sum(0)).sum() \
            + adata.obsp['weight'].sum() ** 2
        nm0 = N ** 2 * (adata.obsp['nearest_neighbors'] * adata.obsp['nearest_neighbors'].T).sum() \
            - 2 * N * (adata.obsp['nearest_neighbors'].sum(1) * adata.obsp['nearest_neighbors'].sum(0)).sum() \
            + adata.obsp['nearest_neighbors'].sum() ** 2
        
    dm = N ** 2 * (N - 1) ** 2
    var = np.hstack(
        (np.repeat(nm0, adata.uns['geneInter'].annotation.isin(['ECM-Receptor', 'Cell-Cell Contact']).sum()),
         np.repeat(nm, adata.uns['geneInter'].annotation.isin(['Secreted Signaling']).sum()))) / dm
    st = var ** (1 / 2)
    return st

def global_I_compute(adata, L_mat, R_mat, n_short_lri, permute=False):
    """Calculate global I (i.e., R) values
    Make sure L_mat and R_mat are numpy.array not matrix
    """
    if permute:
        _idx = np.random.permutation(L_mat.shape[0])
        L_mat0, R_mat0 = L_mat[_idx, :n_short_lri], R_mat[_idx, :n_short_lri]
        L_mat1, R_mat1 = L_mat[_idx, n_short_lri:], R_mat[_idx, n_short_lri:]
    else:
        L_mat0, R_mat0 = L_mat[:, :n_short_lri], R_mat[:, :n_short_lri]
        L_mat1, R_mat1 = L_mat[:, n_short_lri:], R_mat[:, n_short_lri:]

    # Consider to dense array for speedup (numpy's codes is optimised)
    if adata.shape[0] >= 5000 or ~issparse(adata.obsp['weight']):
        RV = np.hstack((
            (adata.obsp['nearest_neighbors'] @ L_mat0 * R_mat0).sum(axis=0),
            (adata.obsp['weight'] @ L_mat1 * R_mat1).sum(axis=0)
        ))
    else:
        # Note, numpy may use unnessary too many threads
        # You may use threadpool.threadpool_limits() outside
        RV = np.hstack((
            (adata.obsp['nearest_neighbors'].toarray() @ L_mat0 * R_mat0).sum(axis=0),
            (adata.obsp['weight'].toarray() @ L_mat1 * R_mat1).sum(axis=0)
        ))

    return RV


def generate_perm_tbl(adata, n_perm, num_spots):
    """shuffle neighbors for n_perm times by shuffling spot lables"""
    perm = np.zeros((n_perm, num_spots))
    perm = perm.astype(type(adata.obs_names.values[0]))
    mylist = list(adata.obs_names)
    for i in range(n_perm):
        random.shuffle(mylist)
        perm[i] = mylist
    return perm


def compute_var_local(adata, sigma1_sq,sigma2_sq,wij_sq,n ,wii=1):
    if adata.uns['single_cell']:
        wii = 0
    var_I = (2 * (n-1)**2/n**2* wij_sq * sigma1_sq * sigma2_sq + 
             2 * (n-1)**2/n**2 * sigma1_sq * sigma2_sq * wii)

    std_I=var_I**(1/2)
    return std_I

def _standardise(X, Local=False, axis=None):
    """Standardise an array
    """
    N = len(X)
    if type(X[0]) == csr_matrix:
        X = np.array([i.toarray()[:,0] for i in X])
        # X = X.T
    if Local==False:
        X = X.T
        N = 1
    X = X - X.mean(axis=axis, keepdims=True)
    X = X / np.sqrt(np.sum(X**2, axis=axis, keepdims=True)/N)
    return X


def pair_selection_matrix(adata, n_perm, sel_ind, method):
    if adata.uns['mean'] == 'geometric':
        from scipy.stats.mstats import gmean
    # local variables (only live in this function scope)
    ligand = adata.uns['ligand'].loc[sel_ind]
    receptor = adata.uns['receptor'].loc[sel_ind]
    type_interaction = adata.uns['geneInter'].loc[sel_ind, 'annotation']
    n_short_lri = (type_interaction != 'Secreted Signaling').sum()

    # averaged ligand values
    L1 = [pd.Series(x[0]).dropna().values for x in ligand.values]
    L_mat = [adata[:, L1[l]].X.astype(np.float64)[:, 0] for l in range(len(L1))]
    for i, k in enumerate(ligand.index):
        if len(ligand.loc[k].dropna()) > 1:
            if adata.uns['mean'] == 'geometric':
                L_mat[i] = gmean(adata[:, ligand.loc[k].dropna()].X, axis=1)
            else:
                L_mat[i] = adata[:, ligand.loc[k].dropna()].X.mean(1)

    # averaged receptor values
    R1 = [pd.Series(x[0]).dropna().values for x in receptor.values]
    R_mat = [adata[:, R1[r]].X.astype(np.float64)[:, 0] for r in range(len(R1))]
    for i, k in enumerate(receptor.index):
        if len(receptor.loc[k].dropna()) > 1:
            if adata.uns['mean'] == 'geometric':
                R_mat[i] = gmean(adata[:, receptor.loc[k].dropna()].X, axis=1)
            else:
                R_mat[i] = adata[:, receptor.loc[k].dropna()].X.mean(1)

    ## Check non-expressed pairs
    idx_use = np.array([(L_mat[i].sum() > 0) * (R_mat[i].sum() > 0) for i in range(len(L_mat))])
    if (np.mean(idx_use) < 1):
        print('Warning: some LR pairs have no expression.')
    adata.uns['ligand'] = ligand.loc[idx_use]
    adata.uns['receptor'] = receptor.loc[idx_use]
    
    if issparse(adata.X):
        L_mat = csc_matrix(hstack(L_mat)).T
        R_mat = csc_matrix(hstack(R_mat)).T
        
        # TODO: not support sparse matrix as intermediate results
        L_mat = L_mat.toarray()
        R_mat = R_mat.toarray()
    else:
        R_mat = np.array(R_mat)
        L_mat = np.array(L_mat)

    R_mat_use = _standardise(R_mat[idx_use], axis=0)
    L_mat_use = _standardise(L_mat[idx_use], axis=0)
    adata.uns['global_I'] = global_I_compute(adata, L_mat_use, R_mat_use, n_short_lri)

    ## Calculate p values
    if method in ['both', 'z-score']:
        adata.uns['global_stat']['z']['z'] = (
            adata.uns['global_I'] / adata.uns['global_stat']['z']['st'][idx_use])
        adata.uns['global_stat']['z']['z'] = adata.uns['global_stat']['z']['z'].astype(np.float64)
        adata.uns['global_stat']['z']['z_p'] = stats.norm.sf(
            adata.uns['global_stat']['z']['z'])
    if method in ['both', 'permutation']:
        adata.uns['global_stat']['perm']['global_perm'] = np.zeros((L_mat_use.shape[1], n_perm))
        for i in tqdm(range(n_perm)):
            ## NOTE: most heavy computation, consider speedup in future (e.g., in parallel or tensor)
            adata.uns['global_stat']['perm']['global_perm'][:, i] = global_I_compute(
                adata, L_mat_use, R_mat_use, n_short_lri, permute=True
            )

def norm_max(X):
    if type(X)==csr_matrix:
        X=X.toarray()[0]
    X = X/X.max()
    X=np.where(np.isnan(X), 0, X)
    return X


def spot_selection_matrix(adata, ligand, receptor, ind, n_perm, method, scale_X=True):
    # local variables (only live in this function scope)
    # normalize raw counts
    raw_norm = adata.raw.to_adata()
    raw_norm.X = csr_matrix([norm_max(X) for X in raw_norm.X.T]).T
    import scanpy as sc
    if scale_X:
        sc.pp.scale(raw_norm, zero_center=False)
    if adata.uns['mean'] == 'geometric':
        from scipy.stats.mstats import gmean
    L1 = [pd.Series(x[0]).dropna().values for x in ligand.values]
    L_mat0 = [raw_norm[:, L1[l]].X.toarray().astype(np.float64)[:, 0] for l in range(len(L1))]
    for i, k in enumerate(ligand.index):
        if len(ligand.loc[k].dropna()) > 1:
            if adata.uns['mean'] == 'geometric':
                L_mat0[i] = gmean(raw_norm[:, ligand.loc[k].dropna()].X.toarray(), axis=1)
            else:
                L_mat0[i] = raw_norm[:, ligand.loc[k].dropna()].X.toarray().mean(1)

    # averaged receptor values
    R1 = [pd.Series(x[0]).dropna().values for x in receptor.values]
    R_mat0 = [raw_norm[:, R1[r]].X.toarray().astype(np.float64)[:, 0] for r in range(len(R1))]
    for i, k in enumerate(receptor.index):
        if len(receptor.loc[k].dropna()) > 1:
            if adata.uns['mean'] == 'geometric':
                R_mat0[i] = gmean(raw_norm[:, receptor.loc[k].dropna()].X.toarray(), axis=1)
            else:
                R_mat0[i] = raw_norm[:, receptor.loc[k].dropna()].X.toarray().mean(1)
    n_short_lri = (adata.uns['geneInter'].loc[ligand.index, 'annotation'] \
                   != 'Secreted Signaling').sum()
    ranges = [np.arange(n_short_lri), np.arange(n_short_lri, len(L1))]
    weight_matrices = [adata.obsp['nearest_neighbors'], adata.obsp['weight']]
    N = adata.shape[0]
    L_mat0 = np.array(L_mat0)
    R_mat0 = np.array(R_mat0)
    pos = np.zeros((N, len(ligand)))

    for r, weight_matrix in zip(ranges, weight_matrices):
        if len(r) == 0:
            continue
        R_mat = R_mat0[r].T
        L_mat = L_mat0[r].T
        R_mat_use = _standardise(R_mat, Local=True, axis=0)
        L_mat_use = _standardise(L_mat, Local=True, axis=0)
        pos[:, r] = (L_mat_use > 0) + (R_mat_use > 0)

        if issparse(weight_matrix):
            wij_sq = weight_matrix.power(2).sum(1).A.reshape(-1)
        else:
            wij_sq = (weight_matrix ** 2).sum(1)
        rbf_d = csc_matrix(weight_matrix)
        adata.uns['local_stat']['local_I'][:, r] = (rbf_d @ R_mat_use) * L_mat_use
        adata.uns['local_stat']['local_I_R'][:, r] = (rbf_d @ L_mat_use) * R_mat_use
            ## Calculate p values
        if method in ['both', 'z-score']:
            norm_res1 = [stats.norm.fit(L_mat_use[:, i]) for i in range(L_mat_use.shape[1])]
            norm_res2 = [stats.norm.fit(R_mat_use[:, i]) for i in range(R_mat_use.shape[1])]
            norm_res1 = np.array(norm_res1)
            norm_res2 = np.array(norm_res2)
            mu1_ls, std_L_ls = norm_res1[:, 0], norm_res1[:, 1]
            mu2_ls, std_R_ls = norm_res2[:, 0], norm_res2[:, 1]
            sigma1_sq_ls = [(std1 * N / (N - 1)) for std1 in std_L_ls]
            sigma2_sq_ls = [(std2 * N / (N - 1)) for std2 in std_R_ls]
            std_ls = [compute_var_local(adata, sigma1_sq, sigma2_sq, wij_sq, N) \
                      for (sigma1_sq, sigma2_sq) in zip(sigma1_sq_ls, sigma2_sq_ls)]
            adata.uns['local_z'][r] = (adata.uns['local_stat']['local_I'][:, r] + \
                                       adata.uns['local_stat']['local_I_R'][:, r]).T / std_ls
            adata.uns['local_z_p'][r] = stats.norm.sf(adata.uns['local_z'][r])

        if method in ['both', 'permutation']:
            for i in tqdm(range(n_perm)):
                _idx = np.random.permutation(L_mat.shape[0])
                adata.uns['local_stat']['local_permI'][r, i, :] = ((rbf_d @ R_mat_use[_idx, :]) * L_mat_use).T
                adata.uns['local_stat']['local_permI_R'][r, i, :] = ((rbf_d @ L_mat_use[_idx, :]) * R_mat_use).T

    try:
        adata.uns['local_z_p'] = np.where(pos.T == False, 1, adata.uns['local_z_p'])
        adata.uns['local_z_p'] = pd.DataFrame(adata.uns['local_z_p'], index=ind, columns=adata.obs_names)
    except Exception:
        pass

    try:
        adata.uns['local_perm_p'] = (np.expand_dims(adata.uns['local_stat']['local_I'].T + \
                                                    adata.uns['local_stat']['local_I_R'].T, 1) <= \
                                     (adata.uns['local_stat']['local_permI'] + adata.uns['local_stat'][
                                         'local_permI_R'])).sum(1) / n_perm
        adata.uns['local_perm_p'] = np.where(pos.T == False, 1, adata.uns['local_perm_p'])
        adata.uns['local_perm_p'] = pd.DataFrame(adata.uns['local_perm_p'], index=ind, columns=adata.obs_names)
    except Exception:
        pass

# def compute_pathway(sample=None,
#                     all_interactions=None,
#         interaction_ls=None, name=None, dic=None):
#     """
#     Compute enriched pathways for a list of pairs or a dic of SpatialDE results.
#     :param sample: spatialdm obj
#     :param ls: a list of LR interaction names for the enrichment analysis
#     :param path_name: str. For later recall sample.path_summary[path_name]
#     :param dic: a dic of SpatialDE results (See tutorial)
#     """
#     if interaction_ls is not None:
#         dic = {name: interaction_ls}
#     if sample is not None:
#         all_interactions = sample.uns['geneInter']
#     df = pd.DataFrame(all_interactions.groupby('pathway_name').interaction_name)
#     df = df.set_index(0)
#     total_feature_num = len(all_interactions)
#     result = []
#     for n,ls in dic.items():
#         qset = set([x.upper() for x in ls]).intersection(all_interactions.index)
#         query_set_size = len(qset)
#         for modulename, members in df.iterrows():
#             module_size = len(members.values[0])
#             overlap_features = qset.intersection(members.values[0])
#             overlap_size = len(overlap_features)

#             negneg = total_feature_num + overlap_size - module_size - query_set_size
#             # Fisher's exact test
#             p_FET = stats.fisher_exact([[overlap_size, query_set_size - overlap_size],
#                                         [module_size - overlap_size, negneg]], 'greater')[1]
#             result.append((p_FET, modulename, module_size, overlap_size, overlap_features, n))
#     result = pd.DataFrame(result).set_index(1)
#     result.columns = ['fisher_p', 'pathway_size', 'selected', 'selected_inters', 'name']
#     if sample is not None:
#         sample.uns['pathway_summary'] = result
#     return result