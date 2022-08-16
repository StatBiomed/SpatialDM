import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import fdrcorrection
from scipy import spatial
import json
from threadpoolctl import threadpool_limits
from scipy.sparse import csc_matrix, save_npz, load_npz
from .utils import *


class SpatialDM(object):
    """
    class SpatialDM(object)
    """

    def __init__(self, logcounts, rawcounts, spatialcoord):
        """
        load spatial data
                Index names for logcounts, rawcounts and spatialcoord should be exactly the same
        :param logcounts: exression matrix dataframe (logcounts): genes in columns,  spots in rows.
        :param rawcounts: exression matrix dataframe (rawcounts): genes in columns,  spots in rows.
        :param spatialcoord: spatial coordinate dataframe (spatialcoord): two columns named 'x' and 'y', spots in rows.
        """
        self.logcounts = logcounts
        self.rawcounts = rawcounts
        self.spatialcoord = spatialcoord
        self.N = spatialcoord.shape[0]
        self.spot_names = spatialcoord.index
        self.gene_names = rawcounts.columns
        self.logcounts = self.logcounts.reindex(index=self.spatialcoord.index)

    def weight_matrix(self, l, cutoff=None, n_neighbors=None, single_cell=False):
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
        pdist = spatial.distance.pdist(self.spatialcoord.values, 'sqeuclidean')
        pdist = spatial.distance.squareform(pdist)
        rbf_d = np.exp(-pdist / (2 * l ** 2))  # RBF Distance
        if rbf_d.shape[0] > 1000:
            rbf_d = rbf_d.astype(np.float16)
        if cutoff:
            rbf_d[rbf_d < cutoff] = 0
        elif n_neighbors:
            nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(rbf_d)
            knn = nbrs.kneighbors_graph(rbf_d).toarray()
            rbf_d = rbf_d * knn
        self.rbf_d = rbf_d * self.N / rbf_d.sum()

        if single_cell:
            np.fill_diagonal(self.rbf_d, 0)
        else:
            pass

        return

    def extract_lr(self, species, min_cell=0):
        """
            find overlapping LRs from CellChatDB
        :param species: only 'human' or 'mouse' is supported
        :param min_cell: for each selected pair, the spots logcountsressing ligand or receptor should be larger than the min,
        respectively.
        :return: ind, ligand, receptor for further selection
        """
        if species == 'mouse':
            geneInter = pd.read_csv('https://figshare.com/ndownloader/files/36638919', index_col=0)
            comp = pd.read_csv('https://figshare.com/ndownloader/files/36638916', header=0, index_col=0)

        elif species == 'human':
            geneInter = pd.read_csv('https://figshare.com/ndownloader/files/36638943', header=0, index_col=0)
            comp = pd.read_csv('https://figshare.com/ndownloader/files/36638940', header=0, index_col=0)
        else:
            raise ValueError("species type: {} is not supported currently. Please have a check.".format(species))
        ligand = geneInter.ligand.values
        receptor = geneInter.receptor.values
        t = []
        for i in range(len(ligand)):
            for n in [ligand, receptor]:
                l = n[i]
                if l in comp.index:
                    n[i] = comp.loc[l].dropna().values[pd.Series \
                        (comp.loc[l].dropna().values).isin(self.logcounts.columns)]
                else:
                    n[i] = pd.Series(l).values[pd.Series(l).isin(self.logcounts.columns)]
            if (len(ligand[i]) > 0) * (len(receptor[i]) > 0):
                if (sum(self.logcounts.loc[:, ligand[i]].mean(axis=1) > 0) >= min_cell) * \
                        (sum(self.logcounts.loc[:, ligand[i]].mean(axis=1) > 0) >= min_cell):
                    t.append(True)
                else:
                    t.append(False)
            else:
                t.append(False)
        self.ligand, self.receptor, self.ind = ligand[t], receptor[t], geneInter[t].index
        self.num_pairs = len(self.ind)
        self.geneInter = geneInter.loc[self.ind]
        if self.num_pairs == 0:
            raise ValueError("No effective RL. Please have a check on input count matrix/species.")
        return

    def spatialdm_global(self, n_perm=1000, select_num=None, method='z-score', nproc=1):
        """
            global selection. 2 alternative methods can be specified.
        :param n_perm: number of times for shuffling receptor expression for a given pair, default to 1000.
        :param select_num: array containing queried indices for quick test/only run selected pair(s).
        If not specified, selection will be done for all pairs
        :param method: default to 'z-score' for computation efficiency.
            Alternatively, can specify 'permutation' or 'both'.
            Two approaches should generate consistent results in general.
        :param nproc: default to 1. Please decide based on your system.
        :return: 'global_res' dataframe in obj attribute containing pair info and p-values
        """
        if type(select_num) == type(None):
            select_num = np.arange(self.num_pairs)  # default to all pairs
        total_len = len(select_num)
        self.ligand = self.ligand[select_num]
        self.receptor = self.receptor[select_num]
        self.ind = self.ind[select_num]
        self.global_I = np.zeros(total_len)

        if method in ['z-score', 'both']:
            self.v = var_compute(self)
            self.st = self.v ** (1 / 2)
            self.z, self.z_p = np.zeros(total_len), np.zeros(total_len)
        if method in ['both', 'permutation']:
            self.global_perm = np.zeros((total_len, n_perm)).astype(np.float16)

        if not (method in ['both', 'z-score', 'permutation']):
            raise ValueError("Only one of ['z-score', 'both', 'permutation'] is supported")

        with threadpool_limits(limits=nproc, user_api='blas'):
            pair_selection_matrix(self, n_perm, select_num, method)

        self.global_res = pd.DataFrame({'ligand': self.ligand,
                                        'receptor': self.receptor}, index=self.ind)
        if method in ['z-score', 'both']:
            self.z_p = np.where(np.isnan(self.z_p), 1, self.z_p)
            self.global_res['z_pval'] = self.z_p
        if method in ['both', 'permutation']:
            self.global_p = 1 - (self.global_I > self.global_perm.T).sum(axis=0) / n_perm
            self.global_res['perm_pval'] = self.global_p
        return

    def sig_pairs(self, method='z-score', fdr=True, threshold=0.1):
        """
            select significant pairs
        :param method: only one of 'z-score' or 'permutation' to select significant pairs.
        :param fdr: True or False. If fdr correction will be done for p-values.
        :param threshold: 0-1. p-value or fdr cutoff to retain significant pairs. Default to 0.1.
        :return: 'selected' column in global_res containing whether or not a pair should be retained
        """
        if method == 'z-score':
            _p = self.global_res['z_pval'].values
        elif method == 'permutation':
            _p = self.global_res['perm_pval'].values
        else:
            raise ValueError("Only one of ['z-score', 'permutation'] is supported")
        if fdr:
            _p = fdrcorrection(_p)[1]
            self.global_res['fdr'] = _p
        self.global_res['selected'] = (_p < threshold)

    def spatialdm_local(self, n_perm=1000, method='z-score', select_num=None,
                        nproc=1):
        """
            local spot selection
        :param n_perm: number of times for shuffling neighbors partner for a given spot, default to 1000.
        :param method: default to 'z-score' for computation efficiency.
            Alternatively, can specify 'permutation' or 'both' (recommended for spot number < 1000, multiprocesing).
        :param select_num: array containing queried indices in sample pair(s).
        If not specified, local selection will be done for all sig pairs
        :param nproc: default to 1.
        :return: local p-value matrix in obj attribute.
        """
        if (int(n_perm / nproc) != (n_perm / nproc)):
            raise ValueError("n_perm should be divisible by nproc")
        if type(select_num) == type(None):
            select_num = np.arange(len(self.ind))[self.global_res['selected']]  # default to global selected pairs
        # total_len = len(select_num)
        self.ligand_sel = self.ligand[select_num]
        self.receptor_sel = self.receptor[select_num]
        self.ind_sel = self.ind[select_num]
        # logcounts = self.rawcounts

        ## different approaches
        with threadpool_limits(limits=nproc, user_api='blas'):
            spot_selection_matrix(self, n_perm, method)

        # if method in ['z-score', 'boh']:
        #         #     self.local_z_p = pd.DataFrame(self.ltocal_z_p, index=ind)
        # if method in ['both', 'permutation']:
        #     self.local_perm_p = pd.DataFrame(self.local_perm_p, index=ind)

    def sig_spots(self, method='z-score', fdr=True, threshold=0.1):
        """
            pick significantly co-expressing spots
        :param method: one of the methods from spatialdm_local, default to 'z-score'.
        :param fdr: True or False, default to True
        :param threshold: p-value or fdr cutoff to retain significant pairs. Default to 0.1.
        :return: obj attributes 1) selected_spots: a binary matrix of which spots being selected for each pair;
         2) n_spots: number of selected spots for each pair.
        """
        if method == 'z-score':
            _p = self.local_z_p
        if method == 'permutation':
            _p = self.local_perm_p
            if fdr:
                _p = fdrcorrection(np.hstack(_p))[1].reshape(_p.shape)
                self.local_fdr = _p
        self.selected_spots = (_p < threshold)
        self.n_spots = self.selected_spots.sum(1)
        self.local_method = method
        return

    def save_spataildm(self, result_dir, exclude=[]):
        """
        save spataildm output to a specified folder
        :param result_dir: name of the specified folder
        """
        try:
            os.makedirs(result_dir)
        except OSError as e:
            if e.errno != e.errno:
                raise
        dic = {}
        for attr in self.__dict__.keys():
            if not attr in exclude:
                res = getattr(self, attr)
                if attr in ['logcounts', 'rawcounts']:
                    save_npz(result_dir + attr + '.npz', csc_matrix(res))
                elif type(res) == pd.core.frame.DataFrame:
                    res.to_csv(os.path.join(result_dir, attr + '.csv'), index=True)
                elif type(res) == np.ndarray:
                    np.save(os.path.join(result_dir, attr), res)
                elif type(res) in [pd.core.indexes.base.Index, pd.core.series.Series]:
                    self.__dict__[attr] = res.values
                    np.save(os.path.join(result_dir, attr), res)
                else:
                    dic[attr] = res

        with open(os.path.join(result_dir, 'other'), "w") as fp:
            json.dump(dic, fp)
        return

def read_spataildm(read_dir):
    """
    read previously save spatialdm obj from a folder
    :param read_dir: the dir of saved spatialdm obj
    :return: spatialdm obj
    """
    spot_names = np.load(os.path.join(read_dir, 'spot_names'), allow_pickle=True)
    gene_names = np.load(os.path.join(read_dir, 'spot_names'), allow_pickle=True)

    logcounts = load_npz(os.path.join(read_dir, 'logcounts.csv'))
    logcounts = pd.DataFrame(logcounts, index=spot_names, columns=gene_names)

    spatialcoord = pd.read_csv(os.path.join(read_dir, 'spatialcoord.csv'), index_col=0)

    rawcounts = load_npz(os.path.join(read_dir, 'rawcounts.csv'))
    rawcounts = pd.DataFrame(rawcounts, index=spot_names, columns=gene_names)

    read_sample = SpatialDM(logcounts, rawcounts, spatialcoord)  # load spatial data

    for f in os.listdir(read_dir):
        if f in ['logcounts.csv', 'rawcounts.csv', 'spatialcoord.csv']:
            pass
        elif f.endswith('csv'):
            read_sample.__dict__[f.split('.')[0]] = pd.read_csv(os.path.join(read_dir, f), index_col=0)
        elif f.endswith('npy'):
            read_sample.__dict__[f.split('.')[0]] = np.load(os.path.join(read_dir, f), allow_pickle=True)
        elif f == 'other':
            with open(os.path.join(read_dir, 'other')) as json_file:
                _data = json.load(json_file)
            for k in _data.keys():
                read_sample.__dict__[k] = _data[k]
    return read_sample


def compute_pathway(sample, ls=None, path_name=None, dic=None):
    """
    Compute enriched pathways for a list of pairs or a dic of SpatialDE results.
    :param sample: spatialdm obj
    :param ls: a list of LR pair indices for the enrichment analysis
    :param path_name: str. For later recall sample.path_summary[path_name]
    :param dic: a dic of SpatialDE results (See tutorial)
    """
    if not 'path_summary' in sample.__dict__:
        sample.__dict__['path_summary'] = {}
    if dic != None:
        sample.__dict__['path_summary']['pairs'] = dic
        for i in range(len(dic)):
            i = str(i)
            sample.__dict__['path_summary']['P{}'.format(i)] = {}
            cts = sample.geneInter.loc[dic['Pattern_' + i], 'pathway_name'].value_counts()
            cts = cts[::-1]
            sample.__dict__['path_summary']['P{}'.format(i)]['counts'] = cts
            perc = cts / \
                   sample.geneInter.loc[:, 'pathway_name'].value_counts()
            sample.__dict__['path_summary']['P{}'.format(i)]['perc'] = perc.dropna()
    if type(ls) != type(None):
        sample.__dict__['path_summary'][path_name] = {}
        cts = sample.geneInter.loc[ls, 'pathway_name'].value_counts()
        cts = cts[::-1]
        sample.__dict__['path_summary'][path_name]['counts'] = cts
        perc = cts / \
               sample.geneInter.loc[:, 'pathway_name'].value_counts()
        sample.__dict__['path_summary'][path_name]['perc'] = perc.dropna()
