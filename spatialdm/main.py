import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import fdrcorrection
from scipy import spatial
# import json
from threadpoolctl import threadpool_limits
from .utils import *
from itertools import zip_longest
import anndata as ann 


def weight_matrix(adata, l=None, cutoff=0.1, n_neighbors=None, 
    n_nearest_neighbors=6, single_cell=False, eff_dist=None):
    """
    compute weight matrix based on radial basis kernel.
    cutoff & n_neighbors are two alternative options to restrict signaling range.
    :param l: radial basis kernel parameter, need to be customized for optimal weight gradient and \
    to restrain the range of signaling before downstream processing.
    :param cutoff: (for secreted signaling) minimum weight to be kept from the rbf weight matrix. \
    Weight below cutoff will be made zero
    :param n_neighbors: (for secreted signaling) number of neighbors per spot from the rbf weight matrix.
    :param n_nearest_neighbors: (for adjacent signaling) number of neighbors per spot from the rbf \
    weight matrix.
    Non-neighbors will be made 0
    :param single_cell: if single cell resolution, diagonal will be made 0.
    :return: secreted signaling weight matrix: adata.obsp['weight'], \
            and adjacent signaling weight matrix: adata.obsp['nearest_neighbors']

    Check more by spatialdm.stats.rbfweight()
    """
    from .stats import rbfweight

    adata.uns['single_cell'] = single_cell
    if isinstance(adata.obsm['spatial'], pd.DataFrame):
        X_loc = adata.obsm['spatial'].values
    else:
        X_loc = adata.obsm['spatial']

    spatial_W, KNN_connect = rbfweight(X_loc, l=l, cutoff=cutoff, 
        n_neighbors=n_neighbors, n_neighbor_layers=n_nearest_neighbors,
        single_cell=single_cell, eff_dist=eff_dist)

    adata.obsp['weight'] = spatial_W
    adata.obsp['nearest_neighbors'] = KNN_connect
    return

def extract_lr(adata, species, mean='algebra', min_cell=0, datahost='builtin'):
    """
    find overlapping LRs from CellChatDB
    :param adata: AnnData object
    :param species: support 'human', 'mouse' and 'zebrafish'
    :param mean: 'algebra' (default) or 'geometric'
    :param min_cell: for each selected pair, the spots expressing ligand or receptor should be larger than the min,
    respectively.
    :param datahost: the host of the ligand-receptor data. 'builtin' for package built-in otherwise from figshare
    :return: ligand, receptor, geneInter (containing comprehensive info from CellChatDB) dataframes \
            in adata.uns
    """
    if mean=='geometric':
        from scipy.stats.mstats import gmean
    adata.uns['mean'] = mean

    if datahost == 'package':
        if species in ['mouse', 'human', 'zerafish']:
            datapath = './datasets/LR_data/%s-' %(species)
        else:
            raise ValueError("species type: {} is not supported currently. Please have a check.".format(species))
        
        import pkg_resources
        stream1 = pkg_resources.resource_stream(__name__, datapath + 'interaction_input_CellChatDB.csv.gz')
        geneInter = pd.read_csv(stream1, index_col=0, compression='gzip')

        stream2 = pkg_resources.resource_stream(__name__, datapath + 'complex_input_CellChatDB.csv')
        comp = pd.read_csv(stream2, header=0, index_col=0)
    else:
        if species == 'mouse':
            geneInter = pd.read_csv('https://figshare.com/ndownloader/files/36638919', index_col=0)
            comp = pd.read_csv('https://figshare.com/ndownloader/files/36638916', header=0, index_col=0)
        elif species == 'human':
            geneInter = pd.read_csv('https://figshare.com/ndownloader/files/36638943', header=0, index_col=0)
            comp = pd.read_csv('https://figshare.com/ndownloader/files/36638940', header=0, index_col=0)
        elif species == 'zebrafish':
            geneInter = pd.read_csv('https://figshare.com/ndownloader/files/38756022', header=0, index_col=0)
            comp = pd.read_csv('https://figshare.com/ndownloader/files/38756019', header=0, index_col=0)
        else:
            raise ValueError("species type: {} is not supported currently. Please have a check.".format(species))
        
    geneInter = geneInter.sort_values('annotation')
    ligand = geneInter.ligand.values
    receptor = geneInter.receptor.values
    geneInter.pop('ligand')
    geneInter.pop('receptor')

    ## NOTE: the following for loop needs speed up
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
    If not specified, selection will be done for all extracted pairs
    :param method: default to 'z-score' for computation efficiency.
        Alternatively, can specify 'permutation' or 'both'.
        Two approaches should generate consistent results in general.
    :param nproc: default to 1. Please decide based on your system.
    :return: 'global_res' dataframe in adata.uns containing pair info and Moran p-values
    """
    if specified_ind is None:
        specified_ind = adata.uns['geneInter'].index.values  # default to all pairs
    else:
        adata.uns['geneInter'] = adata.uns['geneInter'].loc[specified_ind]
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
    adata.uns['global_stat']['method'] = method
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
                    nproc=1, scale_X=True):
    """
        local spot selection
    :param n_perm: number of times for shuffling neighbors partner for a given spot, default to 1000.
    :param method: default to 'z-score' for computation efficiency.
        Alternatively, can specify 'permutation' or 'both' (recommended for spot number < 1000, multiprocesing).
    :param specified_ind: array containing queried indices in sample pair(s).
    If not specified, local selection will be done for all sig pairs
    :param nproc: default to 1.
    :return: 'local_stat' & 'local_z_p' and/or 'local_perm_p' in adata.uns.
    """
    adata.uns['local_stat'] = {}
    if (int(n_perm / nproc) != (n_perm / nproc)):
        raise ValueError("n_perm should be divisible by nproc")
    if type(specified_ind) == type(None):
        specified_ind = adata.uns['global_res'][
            adata.uns['global_res']['selected']].index  # default to global selected pairs
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
        adata.uns['local_z'] = np.zeros((len(ind), adata.shape[0]))
        adata.uns['local_z_p'] = np.zeros((len(ind), adata.shape[0]))

    ## different approaches
    with threadpool_limits(limits=nproc, user_api='blas'):
        spot_selection_matrix(adata, ligand, receptor, ind, n_perm, method, scale_X)


def sig_spots(adata, method='z-score', fdr=True, threshold=0.1):
    """
        pick significantly co-expressing spots
    :param method: one of the methods from spatialdm_local, default to 'z-score'.
    :param fdr: True or False, default to True
    :param threshold: p-value or fdr cutoff to retain significant pairs. Default to 0.1.
    :return:  1) 'selected_spots' in adata.uns: a binary frame of which spots being selected for each pair;
     2) 'n_spots' in adata.uns['local_stat']: number of selected spots for each pair.
    """
    if method == 'z-score':
        _p = adata.uns['local_z_p']
    if method == 'permutation':
        _p = adata.uns['local_perm_p']
    if fdr:
        _fdr = fdrcorrection(np.hstack(_p.values))[1].reshape(_p.shape)
        _p.loc[:,:] = _fdr
        adata.uns['local_stat']['local_fdr'] = _p
    adata.uns['selected_spots'] = (_p < threshold)
    adata.uns['local_stat']['n_spots'] = adata.uns['selected_spots'].sum(1)
    adata.uns['local_stat']['local_method'] = method
    return

def drop_uns_na(adata, global_stat=False, local_stat=False):
    adata.uns['geneInter'] = adata.uns['geneInter'].fillna('NA')
    adata.uns['global_res'] = adata.uns['global_res'].fillna('NA')
    adata.uns['ligand'] = adata.uns['ligand'].fillna('NA')
    adata.uns['receptor'] = adata.uns['receptor'].fillna('NA')
    adata.uns['local_stat']['n_spots'] = pd.DataFrame(adata.uns['local_stat']['n_spots'], columns=['n_spots'])
    if global_stat and ('global_stat' in adata.uns.keys()):
        adata.uns.pop('global_stat')
    if local_stat and ('local_stat' in adata.uns.keys()):
        adata.uns.pop('local_stat')

def restore_uns_na(adata):
    adata.uns['geneInter'] = adata.uns['geneInter'].replace('NA', np.nan)
    adata.uns['global_res'] = adata.uns['global_res'].replace('NA', np.nan)
    adata.uns['ligand'] = adata.uns['ligand'].replace('NA', np.nan)
    adata.uns['receptor'] = adata.uns['receptor'].replace('NA', np.nan)
    adata.uns['local_stat']['n_spots'] =  adata.uns['local_stat']['n_spots'].n_spots

def write_spatialdm_h5ad(adata, filename=None):
    if filename is None:
        filename = 'spatialdm_out.h5ad'
    elif not filename.endswith('h5ad'):
        filename = filename+'.h5ad'
    drop_uns_na(adata)
    adata.write(filename)

def read_spatialdm_h5ad(filename):
    adata = ann.read_h5ad(filename)
    restore_uns_na(adata)
    return adata

def compute_pathway(sample=None,
                    all_interactions=None,
        interaction_ls=None, name=None, dic=None):
    """
    Compute enriched pathways for a list of pairs or a dic of SpatialDE results.
    :param sample: spatialdm obj
    :param ls: a list of LR interaction names for the enrichment analysis
    :param path_name: str. For later recall sample.path_summary[path_name]
    :param dic: a dic of SpatialDE results (See tutorial)
    """
    if interaction_ls is not None:
        dic = {name: interaction_ls}
    if sample is not None:
        all_interactions = sample.uns['geneInter']
    df = pd.DataFrame(all_interactions.groupby('pathway_name').interaction_name)
    df = df.set_index(0)
    total_feature_num = len(all_interactions)
    result = []
    all_path_genes = set(np.concatenate(list(dic.values())))
    all_path_genes = {x.upper() for x in all_path_genes}.intersection(all_interactions.index)
    for n,ls in dic.items():
        qset = set([x.upper() for x in ls]).intersection(all_interactions.index)
        query_set_size = len(qset)
        background = all_path_genes - qset
        for modulename, members in df.iterrows():
            module_size = len(members[1])
            overlap_features = qset.intersection(members[1])
            overlap_size = len(overlap_features)

            background_mapped = background.intersection(members[1])
            background_unmapped = background-background_mapped

            negneg = total_feature_num + overlap_size - module_size - query_set_size
            # Fisher's exact test
            p_FET = stats.fisher_exact([[overlap_size, query_set_size - overlap_size],
                                        [module_size - overlap_size, negneg]], 'greater')[1]
            #result.append((p_FET, modulename, module_size, overlap_size, overlap_features, n))
            result.append((modulename,members[1],qset,overlap_features,background_mapped,
                           background_unmapped,overlap_size,p_FET,n))
    result = pd.DataFrame(result).set_index(0)
    result.index.name = 'pathway'
    result.columns = ['total_genes', 'query_genes', 'overlapped_genes','background_mapped_genes', 
                      'background_unmapped_genes', 'selected','fisher_p','pattern']
    # if sample is not None:
    #     sample.uns['pathway_summary'] = result
    return result
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
