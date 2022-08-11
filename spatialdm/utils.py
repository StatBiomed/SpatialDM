"""
Utils of permutation calculation
"""
import os
import pandas as pd
import numpy as np
import random
from scipy import stats
import time
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm


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


def permutation(sample, LEN_div, ii,
                x, y, L, R, mean1, mean2):
    PermTbl_sub = sample.PermTbl[(LEN_div * ii):(LEN_div * (ii + 1)), :]
    idx_row = PermTbl_sub.reshape(-1, 1)[:, 0]
    value = sample.exp.loc[idx_row, R] - mean2
    value = value.mean(axis=1).values  # mean along row, e.g. ['Tgfbr1' 'Tgfbr2'], --> numpy
    value_R = sample.exp.loc[idx_row, L] - mean1
    value_R = value_R.mean(axis=1).values
    sample.local_permI[:, (LEN_div * ii):(LEN_div * (ii + 1)), :] = \
        np.matmul(sample.rbf_d, value.reshape(-1, sample.N).T).T * x
    sample.local_permI_R[:, (LEN_div * ii):(LEN_div * (ii + 1)), :] = \
        np.matmul(sample.rbf_d, value_R.reshape(-1, sample.N).T).T * y
    return


def generate_perm_tbl(exp, n_perm, num_spots):
    """shuffle neighbors for n_perm times by shuffling spot lables"""
    perm = np.zeros((n_perm, num_spots))
    perm = perm.astype(type(exp.index.values[0]))
    mylist = list(exp.index)
    for i in range(n_perm):
        random.shuffle(mylist)
        perm[i] = mylist
    return perm


def compute_var_local(sigma1_sq,sigma2_sq,wij_sq,n):
    var_I=2 * (n-1)**2/n**2 * sigma1_sq * sigma2_sq * wij_sq + \
        2 * (n-1)**2/n**2 * sigma1_sq * sigma2_sq
    std_I=var_I**(1/2)
    return (var_I, std_I)

def pair_selection(sample, n_perm, sel_ind, method):
    # local variables (only live in this function scope)
    ligand = sample.ligand
    receptor = sample.receptor
    exp = sample.exp
    for k in tqdm(sel_ind):
        start = time.time()
        L = ligand[k]
        R = receptor[k]
        mean1, mean2 = exp.loc[:, L].mean().mean(), exp.loc[:, R].mean().mean()
        x = (exp.loc[:, L] - mean1).mean(axis=1).values
        y = (exp.loc[:, R] - mean2).mean(axis=1).values
        x_sq, y_sq = x ** 2, y ** 2

        sample.global_I[k] = np.matmul(np.matmul(sample.rbf_d, y), x) / \
                      ((sum(x_sq) * sum(y_sq)) ** (1 / 2))
        if method in ['both', 'z-score']:
            sample.z[k] = sample.global_I[k] / sample.st
            sample.z_p[k] = stats.norm.sf(sample.z[k])
        if method in ['both', 'permutation']:
            PermTbl = generate_perm_tbl(exp, n_perm, sample.N)
            idx_row = PermTbl.reshape(-1, 1)[:, 0]
            value = exp.loc[idx_row, R] - mean2
            value = value.mean(axis=1).values  # mean along row, e.g. ['Tgfbr1' 'Tgfbr2'], --> numpy
            local_mat = np.matmul(sample.rbf_d, value.reshape(n_perm, sample.N).T).T * x
            sample.global_perm[k] = local_mat.sum(axis=1) / \
                              ((sum(x_sq) * sum(y_sq)) ** (1 / 2))
        #print(str(k) + ' pairs global selection finished in: ' + str(time.time()-start))
        

def pair_selection_matrix(sample, n_perm, sel_ind, method, via_sparse=True):
    # local variables (only live in this function scope)
    ligand = sample.ligand[sel_ind]
    receptor = sample.receptor[sel_ind]
    GEX = sample.exp
    
    ## Check if the sparse matrix helps
    if via_sparse:
        from scipy.sparse import csc_matrix, csr_matrix
        rbf_d = csc_matrix(sample.rbf_d)
    else:
        rbf_d = sample.rbf_d
    
    def _standardise(X, axis=None):
        """Standardise an array
        """
        X = X - X.mean(axis=axis, keepdims=True)
        X = X / np.sqrt(np.sum(X**2, axis=axis, keepdims=True))
        return X

    # averaged ligand values
    L1 = [x[0] for x in ligand]
    L_mat = GEX.loc[:, L1].values.astype(np.float)
    for k in range(len(ligand)):
        if len(ligand[k]) > 1:
            L_mat[:, k] = GEX.loc[:, ligand[k]].values.mean(1)

    # averaged receptor values
    R1 = [x[0] for x in receptor]
    R_mat = GEX.loc[:, R1].values.astype(np.float)
    for k in range(len(receptor)):
        if len(receptor[k]) > 1:
            R_mat[:, k] = GEX.loc[:, receptor[k]].values.mean(1)
    
    ## Check non-epxressed pairs
    idx_use = (L_mat.sum(0) > 0) * (R_mat.sum(0) > 0)
    if (np.mean(idx_use) < 1):
        print('Warning: some LR pairs have no expression.')
    
    sample.ligand_use = ligand[idx_use]
    sample.receptor_use = receptor[idx_use]
    R_mat_use = _standardise(R_mat[:, idx_use], axis=0)
    L_mat_use = _standardise(L_mat[:, idx_use], axis=0)
    
    ## Change to sparse matrix
    # if via_sparse:
    #     from scipy.sparse import csc_matrix, csr_matrix
    #     rbf_d = csr_matrix(rbf_d)
    #     L_mat_use = csc_matrix(L_mat_use)
    #     R_mat_use = csr_matrix(R_mat_use)  
    #     sample.global_I = np.sum((rbf_d @ L_mat_use).multiply(R_mat_use), 
    #                              axis=0).A

    sample.global_I = ((rbf_d @ L_mat_use) * R_mat_use).sum(axis=0)

    ## Calculate p values
    if method in ['both', 'z-score']:
        sample.z = sample.global_I / sample.st
        sample.z_p = stats.norm.sf(sample.z)
    if method in ['both', 'permutation']:
        sample.global_perm = np.zeros((L_mat.shape[1], n_perm))
        for i in tqdm(range(n_perm)):
            _idx = np.random.permutation(L_mat.shape[0])
            sample.global_perm[:, i] = np.sum(
                (rbf_d @ L_mat_use[_idx, :]) * R_mat_use, axis=0)
                