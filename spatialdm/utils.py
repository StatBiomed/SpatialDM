"""
Utils of permutation calculation
"""
import pandas as pd
import numpy as np
import random
from scipy import stats
import time
from tqdm import tqdm
from scipy.sparse import csc_matrix


# global variance
def var_compute(sample):
    N = sample.N
    nm = N ** 2 * (sample.rbf_d * sample.rbf_d.T).sum() \
         - 2 * N * (sample.rbf_d.sum(1) * sample.rbf_d.sum(0)).sum() \
         + sample.rbf_d.sum() ** 2
    dm = N ** 2 * (N - 1) ** 2
    return nm / dm


def perm_global(sample, k, PermTbl, LEN_div, ii,
                x, R, mean2, x_sq, y_sq):

    idx_row = PermTbl.reshape(-1, 1)[:, 0]
    value = sample.exp.loc[idx_row, R] - mean2
    value = value.mean(axis=1).values  # mean along row, e.g. ['Tgfbr1' 'Tgfbr2'], --> numpy
    local_mat = np.matmul(sample.rbf_d, value.reshape(-1, sample.N).T).T * x
    sample.global_perm[k, (LEN_div * ii):(LEN_div * (ii + 1))] = local_mat.sum(axis=1) / \
                                    ((sum(x_sq) * sum(y_sq)) ** (1 / 2))
    return



def generate_perm_tbl(logcounts, n_perm, num_spots):
    """shuffle neighbors for n_perm times by shuffling spot lables"""
    perm = np.zeros((n_perm, num_spots))
    perm = perm.astype(type(logcounts.index.values[0]))
    mylist = list(logcounts.index)
    for i in range(n_perm):
        random.shuffle(mylist)
        perm[i] = mylist
    return perm


def compute_var_local(sigma1_sq,sigma2_sq,wij_sq,n):
    var_I=2 * (n-1)**2/n**2 * sigma1_sq * sigma2_sq * wij_sq + \
        2 * (n-1)**2/n**2 * sigma1_sq * sigma2_sq
    std_I=var_I**(1/2)
    return std_I

def pair_selection(sample, n_perm, sel_ind, method):
    # local variables (only live in this function scope)
    ligand = sample.ligand
    receptor = sample.receptor
    logcounts = sample.logcounts
    for k in tqdm(sel_ind):
        start = time.time()
        L = ligand[k]
        R = receptor[k]
        mean1, mean2 = logcounts.loc[:, L].mean().mean(), logcounts.loc[:, R].mean().mean()
        x = (logcounts.loc[:, L] - mean1).mean(axis=1).values
        y = (logcounts.loc[:, R] - mean2).mean(axis=1).values
        x_sq, y_sq = x ** 2, y ** 2

        sample.global_I[k] = np.matmul(np.matmul(sample.rbf_d, y), x) / \
                      ((sum(x_sq) * sum(y_sq)) ** (1 / 2))
        if method in ['both', 'z-score']:
            sample.z[k] = sample.global_I[k] / sample.st
            sample.z_p[k] = stats.norm.sf(sample.z[k])
        if method in ['both', 'permutation']:
            PermTbl = generate_perm_tbl(logcounts, n_perm, sample.N)
            idx_row = PermTbl.reshape(-1, 1)[:, 0]
            value = logcounts.loc[idx_row, R] - mean2
            value = value.mean(axis=1).values  # mean along row, e.g. ['Tgfbr1' 'Tgfbr2'], --> numpy
            local_mat = np.matmul(sample.rbf_d, value.reshape(n_perm, sample.N).T).T * x
            sample.global_perm[k] = local_mat.sum(axis=1) / \
                              ((sum(x_sq) * sum(y_sq)) ** (1 / 2))
        #print(str(k) + ' pairs global selection finished in: ' + str(time.time()-start))

def _standardise(X, Local=False, axis=None):
    """Standardise an array
    """
    X = X - X.mean(axis=axis, keepdims=True)
    if not Local:
        X = X / np.sqrt(np.sum(X**2, axis=axis, keepdims=True))
    return X

def pair_selection_matrix(sample, n_perm, sel_ind, method):
    # local variables (only live in this function scope)
    ligand = sample.ligand[sel_ind]
    receptor = sample.receptor[sel_ind]
    ind = sample.ind[sel_ind]
    logcounts = sample.logcounts
    ## Check if the sparse matrix helps
    rbf_d = csc_matrix(sample.rbf_d)

    # averaged ligand values
    L1 = [x[0] for x in ligand]
    L_mat = logcounts.loc[:, L1].values.astype(np.float)
    for k in range(len(ligand)):
        if len(ligand[k]) > 1:
            L_mat[:, k] = logcounts.loc[:, ligand[k]].values.mean(1)

    # averaged receptor values
    R1 = [x[0] for x in receptor]
    R_mat = logcounts.loc[:, R1].values.astype(np.float)
    for k in range(len(receptor)):
        if len(receptor[k]) > 1:
            R_mat[:, k] = logcounts.loc[:, receptor[k]].values.mean(1)

    ## Check non-expressed pairs
    idx_use = (L_mat.sum(0) > 0) * (R_mat.sum(0) > 0)
    if (np.mean(idx_use) < 1):
        print('Warning: some LR pairs have no expression.')
    sample.ligand = ligand[idx_use]
    sample.receptor = receptor[idx_use]
    sample.ind = ind[idx_use]
    # sample.ligand_use = ligand
    # sample.receptor_use = receptor
    R_mat_use = _standardise(R_mat[:, idx_use], axis=0)
    L_mat_use = _standardise(L_mat[:, idx_use], axis=0)

    sample.global_I = ((rbf_d @ L_mat_use) * R_mat_use).sum(axis=0)

    ## Calculate p values
    if method in ['both', 'z-score']:
        sample.z = sample.global_I / sample.st
        sample.z_p = stats.norm.sf(sample.z)
    if method in ['both', 'permutation']:
        sample.global_perm = np.zeros((L_mat_use.shape[1], n_perm))
        for i in tqdm(range(n_perm)):
            _idx = np.random.permutation(L_mat.shape[0])
            sample.global_perm[:, i] = np.sum(
                (rbf_d @ L_mat_use[_idx, :]) * R_mat_use, axis=0)


def spot_selection_matrix(sample, n_perm, method):
    # local variables (only live in this function scope)
    ligand = sample.ligand_sel
    receptor = sample.receptor_sel
    ind = sample.ind_sel
    rawcounts = sample.rawcounts
    # averaged ligand values
    L1 = [x[0] for x in ligand]
    L_mat = rawcounts.loc[:, L1].values.astype(np.float)
    for k in range(len(ligand)):
        if len(ligand[k]) > 1:
            L_mat[:, k] = rawcounts.loc[:, ligand[k]].values.mean(1)

    # averaged receptor values
    R1 = [x[0] for x in receptor]
    R_mat = rawcounts.loc[:, R1].values.astype(np.float)
    for k in range(len(receptor)):
        if len(receptor[k]) > 1:
            R_mat[:, k] = rawcounts.loc[:, receptor[k]].values.mean(1)
    pos = (L_mat > 0) + (R_mat > 0)
    R_mat_use = _standardise(R_mat, Local=True, axis=0)
    L_mat_use = _standardise(L_mat, Local=True, axis=0)
    wij_sq = (sample.rbf_d ** 2).sum(1)
    rbf_d = csc_matrix(sample.rbf_d)

    sample.local_I = (rbf_d @ R_mat_use) * L_mat_use
    sample.local_I_R = (rbf_d @ L_mat_use) * R_mat_use

    ## Calculate p values
    if method in ['both', 'z-score']:
        norm_res1 = [stats.norm.fit(L_mat_use[:, i]) for i in range(L_mat_use.shape[1])]
        norm_res2 = [stats.norm.fit(R_mat_use[:, i]) for i in range(R_mat_use.shape[1])]
        norm_res1 = np.array(norm_res1)
        norm_res2 = np.array(norm_res2)
        mu1_ls, std_L_ls = norm_res1[:,0], norm_res1[:,1]
        mu2_ls, std_R_ls = norm_res2[:,0], norm_res2[:,1]
        sigma1_sq_ls = [(std1 * sample.N / (sample.N - 1)) for std1 in std_L_ls]
        sigma2_sq_ls = [(std2 * sample.N / (sample.N - 1)) for std2 in std_R_ls]
        std_ls = [compute_var_local(sigma1_sq, sigma2_sq, wij_sq, sample.N) \
                    for (sigma1_sq, sigma2_sq) in zip(sigma1_sq_ls, sigma2_sq_ls)]
        sample.local_z = (sample.local_I +sample.local_I_R).T / std_ls
        sample.local_z_p = stats.norm.sf(sample.local_z)
        sample.local_z_p = np.where(pos.T == False, 1, sample.local_z_p)
        sample.local_z_p = pd.DataFrame(sample.local_z_p, index=ind)

    if method in ['both', 'permutation']:
        sample.local_permI = np.zeros((L_mat.shape[1], n_perm, sample.N))
        sample.local_permI_R = np.zeros((L_mat.shape[1], n_perm, sample.N))
        for i in tqdm(range(n_perm)):
            _idx = np.random.permutation(L_mat.shape[0])
            sample.local_permI[:, i,:] = ((rbf_d @ R_mat_use[_idx, :]) * L_mat_use).T
            sample.local_permI_R[:, i,:] = ((rbf_d @ L_mat_use[_idx, :]) * R_mat_use).T
        sample.local_perm_p = (np.expand_dims(sample.local_I.T + sample.local_I_R.T, 1) < \
         (sample.local_permI + sample.local_permI_R)).sum(1) / n_perm
        sample.local_perm_p = np.where(pos.T == False, 1, sample.local_perm_p)
        sample.local_perm_p = pd.DataFrame(sample.local_perm_p, index=ind)

def permutation(sample, LEN_div, ii,
                x, y, L, R, mean1, mean2):
    PermTbl_sub = sample.PermTbl[(LEN_div * ii):(LEN_div * (ii + 1)), :]
    idx_row = PermTbl_sub.reshape(-1, 1)[:, 0]
    value = sample.logcounts.loc[idx_row, R] - mean2
    value = value.mean(axis=1).values  # mean along row, e.g. ['Tgfbr1' 'Tgfbr2'], --> numpy
    value_R = sample.logcounts.loc[idx_row, L] - mean1
    value_R = value_R.mean(axis=1).values
    sample.local_permI[:, (LEN_div * ii):(LEN_div * (ii + 1)), :] = \
        np.matmul(sample.rbf_d, value.reshape(-1, sample.N).T).T * x
    sample.local_permI_R[:, (LEN_div * ii):(LEN_div * (ii + 1)), :] = \
        np.matmul(sample.rbf_d, value_R.reshape(-1, sample.N).T).T * y
    return
