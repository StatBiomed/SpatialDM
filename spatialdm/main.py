import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import fdrcorrection
from scipy import spatial
import json
from threadpoolctl import threadpool_limits
from scipy.sparse import csc_matrix, save_npz, load_npz
from .utils import * #TODO .utils
from itertools import zip_longest


# class SpatialDM(object):
#     """
#     class SpatialDM(object)
#     """
    # def __init__(self, logcounts, rawcounts, spatialcoord):
    #     """
    #     load spatial data
    #             Index names for logcounts, rawcounts and spatialcoord should be exactly the same
    #     :param logcounts: exression matrix dataframe (logcounts): genes in columns,  spots in rows.
    #     :param rawcounts: exression matrix dataframe (rawcounts): genes in columns,  spots in rows.
    #     :param spatialcoord: spatial coordinate dataframe (spatialcoord): two columns named 'x' and 'y', spots in rows.
    #     """
    #     adata.logcounts = logcounts
    #     adata.rawcounts = rawcounts
    #     adata.spatialcoord = spatialcoord
    #     adata.N = spatialcoord.shape[0]
    #     adata.spot_names = spatialcoord.index
    #     adata.gene_names = rawcounts.columns
    #     adata.logcounts = adata.logcounts.reindex(index=adata.spatialcoord.index)

def weight_matrix(adata, l, cutoff=None, n_neighbors=None, n_nearest_neighbors=6, single_cell=False):
    """
    compute weight matrix based on radial basis kernel.
    cutoff & n_neighbors are two alternative options to \
    make the matrix sparse
    :param l: radial basis kernel parameter, need to be customized to restrain the range of signaling
     before downstream processing.
    :param cutoff: minimum weight to be kept from the rbf weight matrix. Weight below cutoff will be made zero
    :param n_neighbors: number of neighbors per spot from the rbf weight matrix.
    Non-neighbors will be made 0
    :param single_cell: if single cell resolution, diagonal will be made 0.
    :return: rbf_d weight matrix in obj attribute
    """
    pdist = spatial.distance.pdist(adata.obsm['spatial'], 'sqeuclidean')
    pdist = spatial.distance.squareform(pdist)
    rbf_d = np.exp(-pdist / (2 * l ** 2))  # RBF Distance
    if rbf_d.shape[0] > 1000:
        rbf_d = rbf_d.astype(np.float16)

    nnbrs = NearestNeighbors(n_nearest_neighbors, algorithm='ball_tree').fit(rbf_d)
    knn0 = nnbrs.kneighbors_graph(rbf_d).toarray()
    rbf_d0 = rbf_d * knn0

    if cutoff:
        rbf_d[rbf_d < cutoff] = 0

    elif n_neighbors:
        nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(rbf_d)
        knn = nbrs.kneighbors_graph(rbf_d).toarray()
        rbf_d = rbf_d * knn

    if single_cell:
        np.fill_diagonal(rbf_d, 0)
        np.fill_diagonal(rbf_d0, 0)
    else:
        pass

    adata.obsp['weight'] = rbf_d * adata.shape[0] / rbf_d.sum()
    adata.obsp['nearest_neighbors'] = rbf_d0 * adata.shape[0] / rbf_d0.sum()
    return

def extract_lr(adata, species, mean='algebra', min_cell=0):
    """
        find overlapping LRs from CellChatDB
    :param species: only 'human' or 'mouse' is supported
    :param min_cell: for each selected pair, the spots logcountsressing ligand or receptor should be larger than the min,
    respectively.
    :return: ind, ligand, receptor for further selection
    """
    if mean=='geometric':
        from scipy.stats.mstats import gmean
    adata.uns['mean'] = mean
    if species == 'mouse':
        geneInter = pd.read_csv('https://figshare.com/ndownloader/files/36638919', index_col=0)
        comp = pd.read_csv('https://figshare.com/ndownloader/files/36638916', header=0, index_col=0)

    elif species == 'human':
        geneInter = pd.read_csv('https://figshare.com/ndownloader/files/36638943', header=0, index_col=0)
        comp = pd.read_csv('https://figshare.com/ndownloader/files/36638940', header=0, index_col=0)
    else:
        raise ValueError("species type: {} is not supported currently. Please have a check.".format(species))
    geneInter = geneInter.sort_values('annotation')
    ligand = geneInter.ligand.values
    receptor = geneInter.receptor.values
    t = []
    for i in range(len(ligand)):
        for n in [ligand, receptor]:
            l = n[i]
            if l in comp.index:
                n[i] = comp.loc[l].dropna().values[pd.Series \
                    (comp.loc[l].dropna().values).isin(adata.var_names)]
            else:
                n[i] = pd.Series(l).values[pd.Series(l).isin(adata.var_names)]
        if (len(ligand[i]) > 0) * (len(receptor[i]) > 0):
            if mean=='geometric':
                meanL = gmean(adata[:, ligand[i]].X, axis=1)
                meanR = gmean(adata[:, receptor[i]].X, axis=1)
            else:
                meanL = adata[:, ligand[i]].X.mean(axis=1)
                meanR = adata[:, receptor[i]].X.mean(axis=1)
            if (sum(meanL > 0) >= min_cell) * \
                    (sum(meanR > 0) >= min_cell):
                t.append(True)
            else:
                t.append(False)
        else:
            t.append(False)
    ind = geneInter[t].index
    adata.uns['ligand'] = pd.DataFrame.from_records(zip_longest(*pd.Series(ligand[t]).values)).transpose()
    adata.uns['ligand'].columns = ['Ligand' + str(i) for i in range(adata.uns['ligand'].shape[1])]
    adata.uns['ligand'].index = ind
    adata.uns['receptor'] = pd.DataFrame.from_records(zip_longest(*pd.Series(receptor[t]).values)).transpose()
    adata.uns['receptor'].columns = ['Receptor' + str(i) for i in range(adata.uns['receptor'].shape[1])]
    adata.uns['receptor'].index = ind
    adata.uns['num_pairs'] = len(ind)
    adata.uns['geneInter'] = geneInter.loc[ind]
    if adata.uns['num_pairs'] == 0:
        raise ValueError("No effective RL. Please have a check on input count matrix/species.")
    return

def spatialdm_global(adata, n_perm=1000, specified_ind=None, method='z-score', nproc=1):
    """
        global selection. 2 alternative methods can be specified.
    :param n_perm: number of times for shuffling receptor expression for a given pair, default to 1000.
    :param specified_ind: array containing queried indices for quick test/only run selected pair(s).
    If not specified, selection will be done for all pairs
    :param method: default to 'z-score' for computation efficiency.
        Alternatively, can specify 'permutation' or 'both'.
        Two approaches should generate consistent results in general.
    :param nproc: default to 1. Please decide based on your system.
    :return: 'global_res' dataframe in obj attribute containing pair info and p-values
    """
    if type(specified_ind) == type(None):
        specified_ind = adata.uns['geneInter'].index.values  # default to all pairs
    total_len = len(specified_ind)
    adata.uns['ligand'] = adata.uns['ligand'].loc[specified_ind]#.values
    adata.uns['receptor'] = adata.uns['receptor'].loc[specified_ind]#.values
    adata.uns['global_I'] = np.zeros(total_len)
    adata.uns['global_stat'] = {}
    if method in ['z-score', 'both']:
        adata.uns['global_stat']['z']={}
        adata.uns['global_stat']['z']['st'] = globle_st_compute(adata)
        adata.uns['global_stat']['z']['z'] = np.zeros(total_len)
        adata.uns['global_stat']['z']['z_p'] = np.zeros(total_len)
    if method in ['both', 'permutation']:
        adata.uns['global_stat']['perm']={}
        adata.uns['global_stat']['perm']['global_perm'] = np.zeros((total_len, n_perm)).astype(np.float16)

    if not (method in ['both', 'z-score', 'permutation']):
        raise ValueError("Only one of ['z-score', 'both', 'permutation'] is supported")

    with threadpool_limits(limits=nproc, user_api='blas'):
        pair_selection_matrix(adata, n_perm, specified_ind, method)

    adata.uns['global_res'] = pd.concat((adata.uns['ligand'], adata.uns['receptor']),axis=1)
    # adata.uns['global_res'].columns = ['Ligand1', 'Ligand2', 'Ligand3', 'Receptor1', 'Receptor2', 'Receptor3', 'Receptor4']
    if method in ['z-score', 'both']:
        adata.uns['global_stat']['z']['z_p'] = np.where(np.isnan(adata.uns['global_stat']['z']['z_p']),
                                                      1, adata.uns['global_stat']['z']['z_p'])
        adata.uns['global_res']['z_pval'] = adata.uns['global_stat']['z']['z_p']
        adata.uns['global_res']['z'] = adata.uns['global_stat']['z']['z']

    if method in ['both', 'permutation']:
        adata.uns['global_stat']['perm']['global_p'] = 1 - (adata.uns['global_I'] \
                             > adata.uns['global_stat']['perm']['global_perm'].T).sum(axis=0) / n_perm
        adata.uns['global_res']['perm_pval'] = adata.uns['global_stat']['perm']['global_p']
    return

def sig_pairs(adata, method='z-score', fdr=True, threshold=0.1):
    """
        select significant pairs
    :param method: only one of 'z-score' or 'permutation' to select significant pairs.
    :param fdr: True or False. If fdr correction will be done for p-values.
    :param threshold: 0-1. p-value or fdr cutoff to retain significant pairs. Default to 0.1.
    :return: 'selected' column in global_res containing whether or not a pair should be retained
    """
    if method == 'z-score':
        _p = adata.uns['global_res']['z_pval'].values
    elif method == 'permutation':
        _p = adata.uns['global_res']['perm_pval'].values
    else:
        raise ValueError("Only one of ['z-score', 'permutation'] is supported")
    if fdr:
        _p = fdrcorrection(_p)[1]
        adata.uns['global_res']['fdr'] = _p
    adata.uns['global_res']['selected'] = (_p < threshold)

def spatialdm_local(adata, n_perm=1000, method='z-score', specified_ind=None,
                    nproc=1):
    """
        local spot selection
    :param n_perm: number of times for shuffling neighbors partner for a given spot, default to 1000.
    :param method: default to 'z-score' for computation efficiency.
        Alternatively, can specify 'permutation' or 'both' (recommended for spot number < 1000, multiprocesing).
    :param specified_ind: array containing queried indices in sample pair(s).
    If not specified, local selection will be done for all sig pairs
    :param nproc: default to 1.
    :return: local p-value matrix in obj attribute.
    """
    adata.uns['local_stat'] = {}

    if (int(n_perm / nproc) != (n_perm / nproc)):
        raise ValueError("n_perm should be divisible by nproc")
    if type(specified_ind) == type(None):
        specified_ind = adata.uns['global_res'][adata.uns['global_res']['selected']].index  # default to global selected pairs
    # total_len = len(specified_ind)
    ligand = adata.uns['ligand'].loc[specified_ind]
    receptor = adata.uns['receptor'].loc[specified_ind]
    ind = ligand.index
    adata.uns['local_stat']['local_I'] = np.zeros((adata.shape[0], len(ind)))
    adata.uns['local_stat']['local_I_R'] = np.zeros((adata.shape[0], len(ind)))
    N = adata.shape[0]
    if method in ['both', 'permutation']:
        adata.uns['local_stat']['local_permI'] = np.zeros((len(ind), n_perm, N))
        adata.uns['local_stat']['local_permI_R'] = np.zeros((len(ind), n_perm, N))
    if method in ['both', 'z-score']:
        adata.uns['local_z'] = np.zeros((len(ind),adata.shape[0]))
        adata.uns['local_z_p'] = np.zeros((len(ind),adata.shape[0]))

    ## different approaches
    with threadpool_limits(limits=nproc, user_api='blas'):
        spot_selection_matrix(adata, ligand, receptor, ind, n_perm, method)

    # if method in ['z-score', 'boh']:
    #         #     adata.local_z_p = pd.DataFrame(adata.ltocal_z_p, index=ind)
    # if method in ['both', 'permutation']:
    #     adata.local_perm_p = pd.DataFrame(adata.local_perm_p, index=ind)

def sig_spots(adata, method='z-score', fdr=True, threshold=0.1):
    """
        pick significantly co-expressing spots
    :param method: one of the methods from spatialdm_local, default to 'z-score'.
    :param fdr: True or False, default to True
    :param threshold: p-value or fdr cutoff to retain significant pairs. Default to 0.1.
    :return: obj attributes 1) selected_spots: a binary matrix of which spots being selected for each pair;
     2) n_spots: number of selected spots for each pair.
    """
    if method == 'z-score':
        _p = adata.uns['local_z_p']
    if method == 'permutation':
        _p = adata.uns['local_perm_p']
        if fdr:
            _p = fdrcorrection(np.hstack(_p))[1].reshape(_p.shape)
            adata.local_fdr = _p
    adata.uns['selected_spots'] = (_p < threshold)
    adata.uns['local_stat']['n_spots'] = adata.uns['selected_spots'].sum(1)
    adata.uns['local_stat']['local_method'] = method
    return

def drop_uns_na(adata, global_stat=False, local_stat=False):
    adata.uns['geneInter'] = adata.uns['geneInter'].fillna('NA')
    adata.uns['ligand'] = adata.uns['ligand'].fillna('NA')
    adata.uns['receptor'] = adata.uns['receptor'].fillna('NA')
    adata.uns['geneInter'].pop('ligand')
    adata.uns['geneInter'].pop('receptor')
    if global_stat and ('global_stat' in adata.uns.keys()):
        adata.uns.pop('global_stat')
    if local_stat and ('local_stat' in adata.uns.keys()):
        adata.uns.pop('local_stat')

def write_spatialdm_h5ad(adata, filename=None):
    if filename is None:
        filename = 'spatialdm_out.h5ad'
    elif not filename.endswith('h5ad'):
        filename = filename+'.h5ad'
    drop_uns_na(adata)
    adata.write(filename)

#     def save_spataildm(adata, result_dir, exclude=[]):
#         """
#         save spataildm output to a specified folder
#         :param result_dir: name of the specified folder
#         """
#         try:
#             os.makedirs(result_dir)
#         except OSError as e:
#             if e.errno != e.errno:
#                 raise
#         dic = {}
#         for attr in adata.__dict__.keys():
#             if not attr in exclude:
#                 res = getattr(self, attr)
#                 if attr in ['logcounts', 'rawcounts']:
#                     save_npz(result_dir + attr + '.npz', csc_matrix(res))
#                 elif type(res) == pd.core.frame.DataFrame:
#                     res.to_csv(os.path.join(result_dir, attr + '.csv'), index=True)
#                 elif type(res) == np.ndarray:
#                     np.save(os.path.join(result_dir, attr), res)
#                 elif type(res) in [pd.core.indexes.base.Index, pd.core.series.Series]:
#                     adata.__dict__[attr] = res.values
#                     np.save(os.path.join(result_dir, attr), res)
#                 else:
#                     dic[attr] = res
# 
#         with open(os.path.join(result_dir, 'other'), "w") as fp:
#             json.dump(dic, fp)
#         return
# 
# def read_spataildm(read_dir):
#     """
#     read previously save spatialdm obj from a folder
#     :param read_dir: the dir of saved spatialdm obj
#     :return: spatialdm obj
#     """
#     spot_names = np.load(os.path.join(read_dir, 'spot_names'), allow_pickle=True)
#     gene_names = np.load(os.path.join(read_dir, 'spot_names'), allow_pickle=True)
# 
#     logcounts = load_npz(os.path.join(read_dir, 'logcounts.csv'))
#     logcounts = pd.DataFrame(logcounts, index=spot_names, columns=gene_names)
# 
#     spatialcoord = pd.read_csv(os.path.join(read_dir, 'spatialcoord.csv'), index_col=0)
# 
#     rawcounts = load_npz(os.path.join(read_dir, 'rawcounts.csv'))
#     rawcounts = pd.DataFrame(rawcounts, index=spot_names, columns=gene_names)
# 
#     read_sample = SpatialDM(logcounts, rawcounts, spatialcoord)  # load spatial data
# 
#     for f in os.listdir(read_dir):
#         if f in ['logcounts.csv', 'rawcounts.csv', 'spatialcoord.csv']:
#             pass
#         elif f.endswith('csv'):
#             read_sample.__dict__[f.split('.')[0]] = pd.read_csv(os.path.join(read_dir, f), index_col=0)
#         elif f.endswith('npy'):
#             read_sample.__dict__[f.split('.')[0]] = np.load(os.path.join(read_dir, f), allow_pickle=True)
#         elif f == 'other':
#             with open(os.path.join(read_dir, 'other')) as json_file:
#                 _data = json.load(json_file)
#             for k in _data.keys():
#                 read_sample.__dict__[k] = _data[k]
#     return read_sample


