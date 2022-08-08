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


def show_progress(RV):
    return RV



# local variance
def var_compute(sample):
    N = sample.N
    nm = N ** 2 * (sample.rbf_d.multiply(sample.rbf_d.transpose())).sum() \
         - 2 * N * (sample.rbf_d.sum(1) * sample.rbf_d.sum(0)).sum() \
         + sample.rbf_d.sum() ** 2
    dm = N ** 2 * (N - 1) ** 2
    return (nm / dm)
# load database
def load_db(exp, species, min_cell, data_root):
    if species == 'mouse':
        geneInter = pd.read_csv(os.path.join(data_root, '0_CellChatDB', 'interaction_input_CellChatDB.csv'),
                                index_col=0)
    elif species == 'human':
        geneInter = pd.read_csv(os.path.join(data_root, '0_CellChatDB', 't.csv'), header=0, index_col=0)
        geneInter.columns = pd.Series(geneInter.columns).str.split('.').str[1]
    else:
        raise ValueError("species type: {} is not supported currently. Please have a check.")

    comp = pd.read_csv(os.path.join(data_root, '0_CellChatDB', 'complex_input_CellChatDB.csv'), header=0, index_col=0)
    ligand = geneInter.ligand.values
    receptor = geneInter.receptor.values
    t = []
    for i in range(len(ligand)):
        for n in [ligand, receptor]:
            l = n[i]
            if l in comp.index:
                n[i] = comp.loc[l].dropna().values[pd.Series(comp.loc[l].dropna().values).isin(exp.columns)]
            else:
                n[i] = pd.Series(l).values[pd.Series(l).isin(exp.columns)]
        if (len(ligand[i]) > 0) * (len(receptor[i]) > 0):
            if (sum(exp.loc[:, ligand[i]].mean(axis=1) > 0) >= min_cell) * \
                    (sum(exp.loc[:, ligand[i]].mean(axis=1) > 0) >= min_cell):
                t.append(True)
            else:
                t.append(False)
        else:
            t.append(False)
    ligand, receptor, ind = ligand[t], receptor[t], geneInter[t].index
    return ligand, receptor, ind


def feature_distance_matrix(j_vec, i_vec):
    """geometric distance between two spots on a specified feature"""
    # TODO: GPU implementation
    # return np.ones((i_vec.shape[0], j_vec.shape[0]))  # TODO: fake results

    return (i_vec.reshape(-1, 1) - j_vec.reshape(1, -1)) ** 2



def permutation(sample, LEN_div, ii,
                x, y, L, R, mean1, mean2):
                # args,
                # ):
    """only permute 100 times, select positive pairs p = 0.10
    Arguments:
        rbf_d (?):
        exp (?):
        num_spots (int):
        ligand (?):
        receptor (?):
        closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    """
    PermTbl_sub = sample.PermTbl[(LEN_div * ii):(LEN_div * (ii + 1)), :]
    idx_row = PermTbl_sub.reshape(-1, 1)[:, 0]
    value = sample.exp.loc[idx_row, R] - mean2
    value = value.mean(axis=1).values  # mean along row, e.g. ['Tgfbr1' 'Tgfbr2'], --> numpy
    value_R = sample.exp.loc[idx_row, L] - mean1
    value_R = value_R.mean(axis=1).values
    sample.local_permI[:, (LEN_div * ii):(LEN_div * (ii + 1)), :] = \
        sample.rbf_d.multiply(value.reshape(-1, sample.N).T).T * x
    sample.local_permI_R[:, (LEN_div * ii):(LEN_div * (ii + 1)), :] = \
        sample.rbf_d.multiply(value_R.reshape(-1, sample.N).T).T * y
    return

    # feat_dist_l = feature_distance_matrix((exp.loc[idx_row, L] - mean1).mean(axis=1).values.astype(np.float16),
    #                                       x.astype(np.float16))
    # feat_dist_r = feature_distance_matrix((exp.loc[idx_row, R] - mean2).mean(axis=1).values.astype(np.float16),
    #                                       y.astype(np.float16))
    # feat_dist = feat_dist_l + feat_dist_r
    # feat_dist2 = feat_dist.reshape(num_spots, len, num_spots) * np.expand_dims(rbf_d, 1)
    # final = feat_dist2.reshape(num_spots * len, num_spots).sum(axis=1).reshape((num_spots, len)).transpose()
    # geary_perm[:, ranges] = final





def create_blank_perm(no_pairs, no_spots, n_perm, is_local=True):
    """create blank tables"""
    lp_dim = (no_pairs, n_perm, no_spots)
    global_permI = np.zeros((no_pairs, n_perm)).astype(np.float16)  # global moran I permutation
    if is_local:
        local_permI, geary_perm = np.zeros(lp_dim, dtype=np.float16), np.zeros(lp_dim, dtype=np.float16)
        return [local_permI, global_permI, geary_perm]
    else:
        return [0, global_permI, 0]


def create_blank_constant(no_pairs, no_spots):
    """create blank tables"""
    l_dim = (no_pairs, no_spots)
    local_I, geary_C = np.zeros(l_dim), np.zeros(l_dim)  # Localmoran I & geary C
    global_I = np.zeros(no_pairs)
    return [local_I, global_I, geary_C]


def generate_perm_tbl(exp, n_perm, num_spots):
    """shuffle neighbors for n_perm times by shuffling spot lables"""
    perm = np.zeros((n_perm, num_spots))
    perm = perm.astype(type(exp.index.values[0]))  # drosophila is indexed by int
    mylist = list(exp.index)
    for i in range(n_perm):
        random.shuffle(mylist)
        perm[i] = mylist
    return perm

def plt_util(title):
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.colorbar()

# def plot_selected_pair(sample, pair, spots, selected_ind, figsize, markersize, cmap, cmap_l, cmap_r):
#     i = pd.Series(selected_ind == pair).idxmax()
#     L = sample.ligand[sample.ind == pair][0]
#     R = sample.receptor[sample.ind == pair][0]
#     l1, l2 = len(L), len(R)
#     plt.figure(figsize=figsize)
#     plt.subplot(1, 5, 1)
#     plt.scatter(sample.spatialcoods.x, sample.spatialcoods.y, marker='s', c=spots[i], cmap=cmap,
#                 vmax=1, s=markersize, linewidth=0)
#     plt_util('Moran: ' + str(sample.n_spots[i]) + ' spots')
#     for l in range(l1):
#         plt.subplot(1, 5, 2 + l)
#         plt.scatter(sample.spatialcoods.x, sample.spatialcoods.y, marker='s', c=sample.exp.loc[L[l]].values, cmap=cmap_l,
#                     s=markersize, linewidth=0)
#         plt_util('Ligand: ' + L[l])
#     for l in range(l2):
#         plt.subplot(1, 5, 2 + l1 + l)
#         plt.scatter(sample.spatialcoods.x, sample.spatialcoods.y, marker='s', c=sample.exp.loc[R[l]], cmap=cmap_r,
#                     s=markersize, linewidth=0)
#         plt_util('Receptor: ' + R[l])

# def compute_var(self):
#     N = no_spots
#     nm = N**2 * (rbf_d*rbf_d.T).sum() \
#         - 2*N * (rbf_d.sum(1) * rbf_d.sum(0)).sum() \
#         + rbf_d.sum() **2
#     dm = N**2 * (N-1)**2
#     return(nm/dm)

# data loading & creating dir for output
# def coarse_selection(num_pairs, num_spots, rbf_d, ind, z_dir,
#                      exp, ligand, receptor, args):
#     """only permute 100 times, select positive pairs p = 0.10
#     Arguments:
#         num_pairs (int):
#         num_spots (int):
#         ind ():
#         z_dir (str): path
#         exp (?):
#         ligand (?):
#         receptor (?):
#         closure (callable, optional): A closure that reevaluates the model
#         and returns the loss.
#     """
#     # local variables (only live in this function scope)
#     N = num_spots
#     var = compute_var(N, rbf_d)
#
#     pos = np.zeros((num_pairs, num_spots))
#     constant, var, z, p = [], [], [], []
#     local_I, global_I, geary_C = create_blank_constant(num_pairs, num_spots)
#     Local_PermI, Global_PermI, Geary_Perm = create_blank_perm(num_pairs, num_spots, args.num_permutation)
#     local_I_R, Local_PermI_R = local_I.copy(), Local_PermI.copy()
#
#     local_p = local_I.copy()
#     local_z = local_I.copy()
#
#     wij_sq=(rbf_d**2).sum(1)
#
#     for k in range(num_pairs):  #### BREAK DOWN !!
#         L = ligand[k]
#         R = receptor[k]
#         if args.dmean:
#             mean1, mean2 = exp.loc[:, L].mean().mean(), exp.loc[:, R].mean().mean()
#         else:
#             mean1, mean2 = 0, 0
#         x = (exp.loc[:, L] - mean1).mean(axis=1).values
#         y = (exp.loc[:, R] - mean2).mean(axis=1).values
#         x_sq, y_sq = x ** 2, y ** 2
#         constant.append(N / (W * (sum(x_sq) * sum(y_sq)) ** (1 / 2)))
#         pos[k] = abs(exp.loc[:, L].mean(axis=1).values) / exp.loc[:, L].mean(axis=1).values + \
#                  abs(exp.loc[:, R].mean(axis=1).values) / exp.loc[:, R].mean(axis=1).values
#         pos[k]=np.where(np.isnan(pos[k]),0,pos[k])
#         LEN_div = int(args.num_permutation / args.nproc)
#         global_I[k] = np.matmul(np.matmul(rbf_d, y), x) / \
#                       ((sum(x_sq) * sum(y_sq)) ** (1 / 2))
#         z.append((global_I[k])/ (var ** (1 / 2)))
#         p.append(stats.norm.sf(z[-1]))
#
#         if args.is_local:
#             local_I[k] = np.matmul(rbf_d, y) * x
#             local_I_R[k] = np.matmul(rbf_d, x) * y
# #             L=ligand[k]
# #             R=receptor[k]
# #             Lexp=exp.loc[L[0]].mean(0)
# #             Rexp=exp.loc[R[0]].mean(0)
# #             Lexp=Lexp-Lexp.mean()
# #             Rexp=Rexp-Rexp.mean()
#
#             from scipy.stats import norm
#
#             mu1, std1 = norm.fit(x)#[(Lexp<Lexp.quantile(.95))])
#             mu2, std2 = norm.fit(y)#[Rexp<Rexp.quantile(.95)])
#             sigma1_sq=std1*N/(N-1)
#             sigma2_sq=std2*N/(N-1)
#     #         s1.append(sigma1_sq)
#     #         s2.append(sigma2_sq)
#     #     S1.append(pd.Series(s1).dropna().quantile(0.05))
#     #     S2.append(pd.Series(s2).dropna().quantile(0.05))
#     #         if (sigma1_sq<S1[k]) or np.isnan(sigma1_sq):
#     #             sigma1_sq=0.07#S1[k]
#     #         if (sigma2_sq<S2[k]) or np.isnan(sigma2_sq):
#     #             sigma2_sq=0.07#S2[k]
#
#             v, st =compute_var_I(sigma1_sq,sigma2_sq,wij_sq,n)
#
#             local_z[i]=(local_I[i]+local_I_R[i])/st
#             local_p[i]=stats.norm.sf(local_z[i].astype(np.float64))
#             local_p[i]=np.where(pos[i]==0,1,local_p[i])
#
#             # geary_C[k] = ((feature_distance_matrix(x, x) + feature_distance_matrix(y, y)) * d).sum(axis=1) #TODO:
#             pool = multiprocessing.Pool(processes=args.nproc)
#             result = []
#             for ii in range(args.nproc):
#                 local_permI, global_permI, geary_perm = create_blank_perm(1, num_spots, LEN_div, args.is_local)
#                 local_permI_R = local_permI.copy()
#                 PermTbl = generate_perm_tbl(exp, LEN_div, num_spots)
#                 # global_permI[k] = (np.matmul(d, value.reshape(LEN_div, no_spots).T).T * x).sum(axis=1) * constant[-1]
#                 result.append(
#                     pool.apply_async(permutation,
#                                      (rbf_d, exp, num_spots, PermTbl, range(LEN_div), LEN_div, constant[-1],
#                                       x, y, L, R, mean1, mean2,
#                                       global_permI, local_permI, local_permI_R, geary_perm,
#                                       args), callback=show_progress))
#             pool.close()
#             pool.join()
#             result = [res.get() for res in result]
#             Global_PermI[k], Local_PermI[k], Local_PermI_R[k], Geary_Perm[k] = [
#                 np.hstack([result[ii][nth_df] for ii in range(args.nproc)]) for nth_df in range(4)]
#         else:  # TODO: mp split
#             PermTbl = generate_perm_tbl(exp, args.num_permutation, num_spots)
#             idx_row = PermTbl.reshape(-1, 1)[:, 0]
#             value = exp.loc[idx_row, R] - mean2
#             value = value.mean(axis=1).values  # mean along row, e.g. ['Tgfbr1' 'Tgfbr2'], --> numpy
#             #  value = np.where(value < 0, 0, value)
#             local_mat = np.matmul(rbf_d, value.reshape(args.num_permutation, num_spots).T).T * x
#             Global_PermI[k] = local_mat.sum(axis=1) / \
#                       ((sum(x_sq) * sum(y_sq)) ** (1 / 2))
#         print(str(k) + 'pair coarse selection finished in :')
#
#     pd.DataFrame({'var': var, 'z': z, 'pval': p}, index=ind).to_csv(z_dir + '/z_res.csv')
#     if args.is_local:
#         return global_I, Global_PermI.transpose(), pos, constant, local_I, local_I_R, geary_C,
#             Local_PermI, Local_PermI_R, Geary_Perm,local_z,local_p
#     else:
#         return global_I, Global_PermI.transpose()

def compute_var_local(sigma1_sq,sigma2_sq,wij_sq,n):
    var_I=2 * (n-1)**2/n**2 * sigma1_sq * sigma2_sq * wij_sq + \
        2 * (n-1)**2/n**2 * sigma1_sq * sigma2_sq
    std_I=var_I**(1/2)
    return (var_I, std_I)

def pair_selection(sample, n_perm, sel_ind, method):
    """only permute 100 times, select positive pairs p = 0.10
    Arguments:
        num_pairs (int):
        num_spots (int):
        ind ():
        z_dir (str): path
        closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    """
    # local variables (only live in this function scope)
    ligand = sample.ligand#[sel_ind],
    receptor = sample.receptor#[sel_ind]
    ind = sample.ind#[sel_ind]
    no_pairs =  len(ind)
    exp = sample.exp
    for k in sel_ind:  #### BREAK DOWN !!
        print('sub')
        print(k)
        start = time.time()
        L = ligand[k]
        R = receptor[k]
        # if sample.dmean:
        mean1, mean2 = exp.loc[:, L].mean().mean(), exp.loc[:, R].mean().mean()
        # else:
        #     mean1, mean2 = 0, 0
        x = (exp.loc[:, L] - mean1).mean(axis=1).values
        y = (exp.loc[:, R] - mean2).mean(axis=1).values
        x_sq, y_sq = x ** 2, y ** 2

        # LEN_div = int(args.num_permutation / args.nproc)
        sample.global_I[k] = np.matmul(sample.rbf_d.multiply(y), x) / \
                      ((sum(x_sq) * sum(y_sq)) ** (1 / 2))
        if method in ['both', 'z-score']:
            sample.z[k] = sample.global_I[k] / sample.st
            sample.z_p[k] = stats.norm.sf(sample.z[k])
        if method in ['both', 'permutation']:
            PermTbl = generate_perm_tbl(exp, n_perm, sample.N)
            idx_row = PermTbl.reshape(-1, 1)[:, 0]
            value = exp.loc[idx_row, R] - mean2
            value = value.mean(axis=1).values  # mean along row, e.g. ['Tgfbr1' 'Tgfbr2'], --> numpy
            #  value = np.where(value < 0, 0, value)
            local_mat = sample.rbf_d.multiply(value.reshape(n_perm, sample.N).T).T * x
            sample.global_perm[k] = local_mat.sum(axis=1) / \
                              ((sum(x_sq) * sum(y_sq)) ** (1 / 2))
        print(str(k) + 'pairs global selection finished in: ' + str(time.time()-start))
        # self.global_I = global_I
        # self.global_perm = global_perm

# def spot_selection(sample, n_perm, sel_ind, method):
#     """only permute 100 times, select positive pairs p = 0.10
#     Arguments:
#         num_pairs (int):
#         num_spots (int):
#         ind ():
#         z_dir (str): path
#         closure (callable, optional): A closure that reevaluates the model
#         and returns the loss.
#     """
#     # local variables (only live in this function scope)
#     ligand = sample.ligand[sel_ind],
#     receptor = sample.receptor#[sel_ind]
#     ind = sample.ind#[sel_ind]
#     no_pairs =  len(ind)
#     exp = sample.exp
#     for k in sel_ind:  #### BREAK DOWN !!
#         print('sub')
#         print(k)
#         start = time.time()
#         L = ligand[k]
#         R = receptor[k]
#         # if sample.dmean:
#         mean1, mean2 = exp.loc[:, L].mean().mean(), exp.loc[:, R].mean().mean()
#         # else:
#         #     mean1, mean2 = 0, 0
#         x = (exp.loc[:, L] - mean1).mean(axis=1).values
#         y = (exp.loc[:, R] - mean2).mean(axis=1).values
#         x_sq, y_sq = x ** 2, y ** 2
#
#         sample.local_I[k] = np.matmul(d, y) * x
#         sample.local_I_R[k] = np.matmul(d, x) * y
#         if method in ['both', 'z-score']:
#             sample.z[k] = sample.global_I[k] / sample.st
#             sample.z_p[k] = stats.norm.sf(sample.z[k])
#         if method in ['both', 'permutation']:
#             PermTbl = generate_perm_tbl(exp, n_perm, sample.N)
#             idx_row = PermTbl.reshape(-1, 1)[:, 0]
#             value = exp.loc[idx_row, R] - mean2
#             value = value.mean(axis=1).values  # mean along row, e.g. ['Tgfbr1' 'Tgfbr2'], --> numpy
#             #  value = np.where(value < 0, 0, value)
#             local_mat = np.matmul(sample.rbf_d, value.reshape(n_perm, sample.N).T).T * x
#             sample.local_permI, sample.local_permI_R = create_blank_perm(1, no_spots, LEN_div)
#             local_permI_R = local_permI.copy()
#         print(str(k) + 'pairs global selection finished in: ' + str(time.time()-start))
