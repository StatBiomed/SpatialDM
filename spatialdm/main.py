import os
import argparse
import time

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import norm
import matplotlib.pyplot as plt
import json
from utils import *

class SpatialDM(object):
    """ 
    class SpatialDM(object)
    """
    def __init__(self, exp, spatialcoord):
        """
        load spatial data
                Index names for exp and spatialcoord should be exactly the same
        :param exp: expression matrix (exp): genes in columns,  spots in rows.
        :param spatialcoord: spatial coordinate (spatialcoord): two columns named 'x' and 'y', spots in rows.  
        """
        self.exp = exp
        self.spatialcoord = spatialcoord
        self.N = spatialcoord.shape[0]
        self.exp = self.exp.reindex(index=self.spatialcoord.index)
    
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
        dis = (self.spatialcoord.x.values.reshape(-1, 1) - self.spatialcoord.x.values) ** 2 + \
              (self.spatialcoord.y.values.reshape(-1, 1) - self.spatialcoord.y.values) ** 2
        rbf_d = np.exp(-dis / (2 * l ** 2))  # RBF Distance
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
            np.fill_diagonal(rbf_d, 0)
        else:
            pass

        return #rbf_d

    def extract_lr(self, species, dir_db, min_cell=0):
        """
            find overlapping LRs from CellChatDB
        :param species: only 'human' or 'mouse' is supported
        :param dir_db: dir of 0_CellChatDB folder
        :param min_cell: for each selected pair, the spots expressing ligand or receptor should be larger than the min,
        respectively.
        :return: 3 obj attributes: ind, ligand, receptor
        """
        if species == 'mouse':
            geneInter = pd.read_csv(os.path.join(dir_db, '0_CellChatDB',
                                                species, 'interaction_input_CellChatDB.csv'),
                                    index_col=0)
        elif species == 'human':
            geneInter = pd.read_csv(os.path.join(dir_db, '0_CellChatDB',
                                                species, 't.csv'), header=0, index_col=0)
            geneInter.columns = pd.Series(geneInter.columns).str.split('.').str[1]
        else:
            raise ValueError("species type: {} is not supported currently. Please have a check.".format(species))
    
        comp = pd.read_csv(os.path.join(dir_db, '0_CellChatDB', species,
                                        'complex_input_CellChatDB.csv'), header=0, index_col=0)
        ligand = geneInter.ligand.values
        receptor = geneInter.receptor.values
        t = []
        for i in range(len(ligand)):
            for n in [ligand, receptor]:
                l = n[i]
                if l in comp.index:
                    n[i] = comp.loc[l].dropna().values[pd.Series \
                        (comp.loc[l].dropna().values).isin(self.exp.columns)]
                else:
                    n[i] = pd.Series(l).values[pd.Series(l).isin(self.exp.columns)]
            if (len(ligand[i]) > 0) * (len(receptor[i]) > 0):
                if (sum(self.exp.loc[:, ligand[i]].mean(axis=1) > 0) >= min_cell) * \
                        (sum(self.exp.loc[:, ligand[i]].mean(axis=1) > 0) >= min_cell):
                    t.append(True)
                else:
                    t.append(False)
            else:
                t.append(False)
        self.ligand, self.receptor, self.ind = ligand[t], receptor[t], geneInter[t].index
        self.num_pairs = len(self.ind)
        self.geneInter =  geneInter.loc[self.ind]
        if self.num_pairs==0:
            raise ValueError("No effective RL. Please have a check on input exp/species.")
        return

    def spatialdm_global(self, n_perm=1000, select_num=None, method='z-score',nproc=1):
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
            select_num = np.arange(self.num_pairs) # default to all pairs
        total_len = len(select_num)
        self.ligand = self.ligand[select_num]
        self.receptor = self.receptor[select_num]
        self.ind = self.ind[select_num]
        self.global_res = pd.DataFrame({'ligand': self.ligand,
                                 'receptor': self.receptor}, index=self.ind)
        if method in ['z-score', 'both']:
            self.v = var_compute(self)
            self.st = self.v ** (1 / 2)
            self.z, self.z_p = np.zeros(total_len), np.zeros(total_len)
        if method in ['both', 'permutation']:
            self.global_perm = np.zeros((total_len, n_perm)).astype(np.float16)

        if not (method in ['both', 'z-score', 'permutation']):
            raise ValueError("Only one of ['z-score', 'both', 'permutation'] is supported")
        self.global_I = np.zeros(total_len)
        LEN_div = int(len(select_num) / nproc)
        pool = multiprocessing.Pool(processes=nproc)
        for ii in range(nproc):

            sel_ind = np.arange(LEN_div * ii, LEN_div * (ii+1), 1)
            pool.apply_async(pair_selection(self, n_perm, sel_ind, method))
        pool.close()
        pool.join()
        pool = multiprocessing.Pool(processes=nproc)
        if (len(select_num) % nproc) > 0:
            sel_ind = select_num[-(len(select_num) % nproc):]
            print('sel_ind')
            print(sel_ind)
            pool.apply_async(pair_selection(self, n_perm, sel_ind, method))
        pool.close()
        pool.join()

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
        if method=='z-score':
            _p = self.global_res['z_pval'].values
        elif method=='permutation':
            _p = self.global_res['perm_pval'].values
        else:
            raise ValueError("Only one of ['z-score', 'permutation'] is supported")
        if fdr:
            _p = fdrcorrection(_p)[1]
            self.global_res['fdr'] = _p
        self.global_res['selected'] = (_p<threshold)

    def spatialdm_local(self, n_perm=1000, method='z-score', select_num=None, nproc=1):
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
            select_num = np.arange(len(self.ind))[self.global_res['selected']] # default to global selected pairs
        total_len = len(select_num)
        ligand = self.ligand[select_num]  # [sel_ind],
        receptor = self.receptor[select_num]  # [sel_ind]
        ind = self.ind[select_num]  # [sel_ind]
        exp = self.exp
        self.local_I = np.zeros((total_len, self.N))
        self.local_I_R = self.local_I.copy()
        self.pos = self.local_I.copy()

        if method in ['z-score', 'both']:
            self.local_v, self.local_st = np.zeros(total_len), np.zeros(total_len)
            self.local_z, self.local_z_p = np.zeros((total_len, self.N)), np.zeros((total_len, self.N))
            wij_sq = (self.rbf_d ** 2).sum(1)

        if method in ['both', 'permutation']:
            LEN_div = int(n_perm / nproc)
            self.local_permI = np.zeros((total_len, n_perm, self.N),
                                        dtype=np.float32)
            self.local_permI_R = self.local_permI.copy()
            self.local_perm_p =  self.local_I.copy()
            self.PermTbl = generate_perm_tbl(exp, n_perm, self.N)
        for k in range(total_len):
            print('sub')
            print(k)
            start = time.time()
            L = ligand[k]
            R = receptor[k]
            x = exp.loc[:, L].mean(axis=1).values
            y = exp.loc[:, R].mean(axis=1).values
            self.pos[k] = (abs(x) / x + abs(y) / y) / 2
            mean_x = x.mean()
            mean_y = y.mean()
            x = x - x.mean()
            y = y - mean_y
            x_sq, y_sq = x ** 2, y ** 2

            self.local_I[k] = np.matmul(self.rbf_d, y) * x
            self.local_I_R[k] = np.matmul(self.rbf_d, x) * y
            self.pos[k] = np.where(np.isnan(self.pos[k]), 0, self.pos[k])

            if method in ['z-score', 'both']:
                mu1, std1 = norm.fit(x)
                mu2, std2 = norm.fit(y)
                sigma1_sq = std1 * self.N / (self.N - 1)
                sigma2_sq = std2 * self.N / (self.N - 1)
                v, st = compute_var_local(sigma1_sq, sigma2_sq, wij_sq, self.N)
                self.local_z[k] = (self.local_I[k] + self.local_I_R[k]) / st
                self.local_z_p[k] = stats.norm.sf(self.local_z[k].astype(np.float64))
                self.local_z_p[k] = np.where(self.pos[k] == 0, 1, self.local_z_p[k])

            if method in ['both', 'permutation']:
                pool = multiprocessing.Pool(processes=nproc)
                for ii in range(nproc):  # split no_perm permutation into size of LEN_dix
                    pool.apply_async(permutation, (self, LEN_div, ii,
                        x, y, L, R, mean_x, mean_y))
                pool.close()
                pool.join()
                self.local_perm_p[k] = (np.expand_dims(self.local_I[k] + self.local_I_R[k], 0) < \
                                    (self.local_permI[k] + self.local_permI_R[k])).sum(0)/n_perm
                self.local_perm_p[k] = np.where(self.pos[k] == 0, 1, self.local_perm_p[k])

            print(str(k+1) + 'pairs local selection finished in '+ str(time.time() - start))
        if method in ['z-score', 'both']:
            self.local_z_p = pd.DataFrame(self.local_z_p, index=ind)
        if method in ['both', 'permutation']:
            self.local_perm_p = pd.DataFrame(self.local_perm_p, index=ind)


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
                self.local_z_fdr = _p
        self.selected_spots = (_p < threshold)
        self.n_spots = self.selected_spots.sum(1)
        self.local_method = method
        return

    def save_spataildm(my_sample, result_dir):
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
        for attr in my_sample.__dict__.keys():
            res = getattr(my_sample, attr)
            if type(res) == pd.core.frame.DataFrame:
                res.to_csv(os.path.join(result_dir, attr + '.csv'), index=True)
            elif type(res) == np.ndarray:
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
    exp = pd.read_csv(os.path.join(read_dir, 'exp.csv'), index_col=0)
    spatialcoord = pd.read_csv(os.path.join(read_dir, 'spatialcoord.csv'), index_col=0)

    read_sample = SpatialDM(exp, spatialcoord)     # load spatial data

    for f in os.listdir(read_dir):
        if f in ['exp.csv', 'spatialcoord.csv']:
            pass
        elif f.endswith('csv'):
            read_sample.__dict__[f.split('.')[0]] = pd.read_csv(os.path.join(read_dir, f), index_col=0)
        elif f.endswith('npy'):
            read_sample.__dict__[f.split('.')[0]] = np.load(os.path.join(read_dir, f), allow_pickle=True)
        elif f=='other':
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
    sample.__dict__['path_summary']={}
    if dic != None:
        sample.__dict__['path_summary']['pairs'] = dic
        for i in range(len(dic)):
            i=str(i)
            sample.__dict__['path_summary']['P{}'.format(i)]={}
            cts = sample.geneInter.loc[dic['Pattern_'+i],'interaction.pathway_name'].value_counts()
            cts = cts[::-1]
            sample.__dict__['path_summary']['P{}'.format(i)]['counts'] = cts
            perc = cts / \
                sample.geneInter.loc[:,'interaction.pathway_name'].value_counts()
            sample.__dict__['path_summary']['P{}'.format(i)]['perc'] = perc.dropna()
    if type(ls)!=type(None):
        sample.__dict__['path_summary'][path_name] = {}
        cts = sample.geneInter.loc[ls, 'interaction.pathway_name'].value_counts()
        cts = cts[::-1]
        sample.__dict__['path_summary'][path_name]['counts'] = cts
        perc = cts / \
               sample.geneInter.loc[:, 'interaction.pathway_name'].value_counts()
        sample.__dict__['path_summary'][path_name]['perc'] = perc.dropna()

# # def plot_pairs(self, pairs_to_plot, pdf=None, figsize=(35, 5), markersize=18,
#     #                cmap='Green', cmap_l='coolwarm', cmap_r='coolwarm'):
#     #     """
#     #     plot selected spots as well as LR expression.
#     #     :param pairs_to_plot: pair name(s), should be from spatialdm_local pairs .
#     #     :param pdf: pdf file prefix. save plots in a pdf file.
#     #     :param figsize: figsize for each pair. Default to (35, 5).
#     #     :param markersize: markersize for each spot. Default
#     #     :param cmap: cmap for selected local spots.
#     #     :param cmap_l: cmap for selected ligand.
#     #     :param cmap_r: cmap for selected receptor.
#     #     :return:
#     #     """
#     #     if self.local_method == 'z-score':
#     #         selected_ind = self.local_z_p.index
#     #         spots = 1- self.local_z_p
#     #     if self.local_method == 'permutation':
#     #         selected_ind = self.local_perm_p.index
#     #         spots = 1- self.local_perm_p
#     #     if pdf != None:
#     #         with PdfPages(pdf + '.pdf') as pdf:
#     #             for pair in pairs_to_plot:
#     #                 plot_selected_pair(self, pair, spots, selected_ind, figsize, markersize, cmap=cmap,
#     #                                    cmap_l=cmap_l, cmap_r=cmap_r)
#     #                 pdf.savefig()
#     #                 plt.show()
#     #                 plt.close()
#     #
#     #     else:
#     #         for pair in pairs_to_plot:
#     #             plot_selected_pair(self, pair, spots, selected_ind, figsize, markersize, cmap=cmap,
#     #                                cmap_l=cmap_l, cmap_r=cmap_r)
#     #             plt.show()
#     #             plt.close()
#
#     def global_result(result, ind, perm_dir):
#         type(my_sample.spatialcoord) == pd.core.frame.DataFrame
#         global_I, Global_PermI = result
#
#         p = 1 - (global_I > Global_PermI).sum(0) / 1000
#         pairs = ind[p < 0.05]
#         selected = (p < 0.05)
#
#         checkpoint = dict(global_I=global_I, Global_PermI=Global_PermI, p=p,
#                           pairs=pairs, selected=selected)
#
#         for k, v in checkpoint.items():
#             np.save(perm_dir + '/{}.npy'.format(k), np.array(v, dtype=object))
#             print('Successfully save perm_dir/{} ...'.format(k))
#
#
#     def local_result(result, perm_dir, result_dir):
#         global_I, Global_PermI, pos, constant, local_I, local_I_R, geary_C, Local_PermI, Local_PermI_R, Geary_Perm = result
#         Geary_spots = np.sum((np.expand_dims(geary_C, 1) < Geary_Perm), axis=1)
#         Moran_spots = np.sum((np.expand_dims(local_I, 1) > Local_PermI), axis=1)
#         Moran_spots_R = np.sum((np.expand_dims(local_I_R, 1) > Local_PermI_R), axis=1)
#
#         Geary_spots = Geary_spots * pos / 2
#         Geary_spots[Geary_spots < 900] = 0
#         Geary_spots = Geary_spots.astype(float)
#         Geary_spots = np.where(np.isnan(Geary_spots), 0, Geary_spots)
#         no_Geary_spots = (Geary_spots > 0).sum(axis=1)
#
#         Moran_spots = Moran_spots * pos / 2
#         Moran_spots[Moran_spots < 900] = 0
#         Moran_spots = Moran_spots.astype(float)
#         Moran_spots = np.where(np.isnan(Moran_spots), 0, Moran_spots)
#         no_Moran_spots = (Moran_spots > 0).sum(axis=1)
#
#         Moran_spots_R = Moran_spots_R * pos / 2
#         Moran_spots_R[Moran_spots_R < 900] = 0
#         Moran_spots_R = Moran_spots_R.astype(float)
#         Moran_spots_R = np.where(np.isnan(Moran_spots_R), 0, Moran_spots_R)
#         no_Moran_spots_R = (Moran_spots_R > 0).sum(axis=1)
#
#         checkpoint = dict(
#             global_I=global_I,
#             Global_PermI=Global_PermI,
#             pos=pos,
#             constant=constant,
#             local_I=local_I,
#             local_I_R=local_I_R,
#             geary_C=geary_C,
#             Local_PermI=Local_PermI,
#             Local_PermI_R=Local_PermI_R,
#             Geary_Perm=Geary_Perm,
#         )
#         for k, v in checkpoint.items():
#             np.save(perm_dir + '/{}.npy'.format(k), np.array(v, dtype=object))
#             print('Successfully save perm_dir/{} ...'.format(k))
#
#         checkpoint = dict(
#             Geary_spots=Geary_spots,
#             Moran_spots=Moran_spots,
#             Moran_spots_R=Moran_spots_R,
#             no_Geary_spots=no_Geary_spots,
#             no_Moran_spots=no_Moran_spots,
#             no_Moran_spots_R=no_Moran_spots_R
#         )
#         for k, v in checkpoint.items():
#             np.save(result_dir + '/{}.npy'.format(k), np.array(v, dtype=object))
#             print('Successfully save result_dir/{} ...'.format(k))