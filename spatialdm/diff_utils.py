import pandas as pd
import numpy as np
import anndata as ann
from statsmodels.stats.multitest import fdrcorrection

def concat_db(adatas, species):
    """
    Merge all interaction database from a list of spatialdm obj.
    :param samples: a list of spatialdm obj to be merged.
    :param species: str. 'human' or 'mouse'.
    :return:
    """
    if species == 'mouse':
        geneInter = pd.read_csv('https://figshare.com/ndownloader/files/36638919', index_col=0)
        comp = pd.read_csv('https://figshare.com/ndownloader/files/36638916', header=0, index_col=0)
    elif species == 'human':
        geneInter = pd.read_csv('https://figshare.com/ndownloader/files/36638943', header=0, index_col=0)
        comp = pd.read_csv('https://figshare.com/ndownloader/files/36638940', header=0, index_col=0)
    else:
        raise ValueError("species type: {} is not supported currently. Please have a check.".format(species))
    ligand = pd.concat([sample.uns['ligand'] for sample in adatas], axis=1)
    receptor = pd.concat([sample.uns['receptor'] for sample in adatas], axis=1)
    ligand=ligand[~ligand.index.duplicated()]
    receptor=receptor[~receptor.index.duplicated()]
    geneInter = geneInter.loc[ligand.index]
    return ligand, receptor, geneInter

def concat_obj(samples, names, species, method='z-score', fdr=False):
    # def __init__(self, samples, names, species, method='z-score', fdr=False):
    """
    Merge all global results from a list of spatialdm obj
    :param samples: a list of spatialdm obj to be merged.
    :param names: a list of str for each sample's name.
    :param species: str. 'human' or 'mouse'.
    :param dir_db']: dir containing 0_CellChatdb'] folder.
    :param method: 'z-score' or 'permutation'. Should be the commonly selected method from all samples.
    :param fdr: If use fdr or p-values for differential analysis
    """
    cdata = ann.concat(samples, join='outer')
    cdata.obs['batch'] = np.repeat(names, [sample.shape[0] for sample in samples])
    cdata.uns['ligand'], cdata.uns['receptor'], cdata.uns['geneInter'] = concat_db(samples, species)
    n_samples = len(samples)
    cdata.uns['method'] = method
    cdata.uns['p_df'] = pd.DataFrame(np.zeros((cdata.uns['ligand'].shape[0], n_samples)),
                                     index=cdata.uns['ligand'].index, columns=names)
    cdata.uns['tf_df'] = cdata.uns['p_df'].copy()
    if method == 'z-score':
        cdata.uns['zscore_df'] = cdata.uns['p_df'].copy()
        for sample, d in zip(samples, names):
            sample.uns['global_res']['z'] = pd.Series(sample.uns['global_res']['z'],
                                                       index=sample.uns['ligand'].index)
            if fdr:
                cdata.uns['p_df'][d] = sample.uns['global_res'].fdr
            else:
                cdata.uns['p_df'][d] = sample.uns['global_res'].z_pval
            cdata.uns['p_df'][d] = np.where(np.isnan(cdata.uns['p_df'][d]), 1, cdata.uns['p_df'][d])
            cdata.uns['zscore_df'][d] = sample.uns['global_res']['z']
            cdata.uns['zscore_df'][d] = np.where(np.isnan(cdata.uns['zscore_df'][d]), 0, cdata.uns['zscore_df'][d])
            cdata.uns['tf_df'][d] = sample.uns['global_res'].selected
    elif method == 'permutation':
        print('This function to be updated') #TODO: update!
        for sample, d in zip(samples, names):
            sample.uns['z']['z'] = pd.Series(sample.uns['z']['z'], sample.uns['global_res'].index)
            if fdr:
                cdata.uns['p_df'][d] = sample.uns['global_res'].fdr
            else:
                cdata.uns['p_df'][d] = sample.uns['global_res'].perm_pval
    else:
        raise ValueError("Only one of ['z-score', 'permutation'] is supported")
    cdata.uns['tf_df'] = cdata.uns['tf_df'].fillna(False)
    cdata.uns['tf_df'] = cdata.uns['tf_df'].astype(bool)
    return cdata

def differential_test(cdata, subset, conditions):
    """
    Test whether each pair is differential among 2 or more conditions
    :param subset: list of concat_obj names to perform differential test on.
    :param conditions:
    :return:
    """
    if cdata.uns['method'] == 'z-score':
        import statsmodels.api as sm
        import scipy
        cdata.uns['subset'] = subset
        cdata.uns['conditions'] = conditions
        n_sub = len(subset)
        cdata.uns['n_sub'] = n_sub
        LR_statistic = np.zeros((len(cdata.uns['p_df'])))
        cdata.uns['p_val'] = np.zeros((len(cdata.uns['p_df'])))
        x = conditions
        for i in range(len(cdata.uns['p_df'])):
            # full model
            p = cdata.uns['zscore_df'].index[i]
            y = np.where(cdata.uns['zscore_df'].loc[p, subset].isna(), 0, cdata.uns['zscore_df'].loc[p, subset].values)
            y1 = y
            x1 = np.vstack((np.ones(n_sub), x)).T
            full_model = sm.OLS(y1, x1).fit()
            full_ll = full_model.llf

            # reduced model
            y2 = y
            x2 = np.ones(n_sub)
            reduced_model = sm.OLS(y2, x2).fit()
            reduced_ll = reduced_model.llf

            # calculate likelihood ratio Chi-Squared test statistic
            LR_statistic[i] = -2 * (reduced_ll - full_ll)
            cdata.uns['p_val'][i] = scipy.stats.chi2.sf(LR_statistic[i], 1)

        cdata.uns['diff'] = cdata.uns['zscore_df'].loc[:, np.array(subset)[conditions == 1]].mean(1).values - \
                    cdata.uns['zscore_df'].loc[:, np.array(subset)[conditions == 0]].mean(1).values

        cdata.uns['p_val'] = np.where(np.isnan(cdata.uns['p_val']), 1, cdata.uns['p_val'])

        cdata.uns['diff_fdr'] = fdrcorrection(cdata.uns['p_val'])[1]

def group_differential_pairs(cdata, c1_name, c2_name, diff_quantile1=0.7, diff_quantile2=0.3, fdr_co=0.1):
    _range = np.arange(1, cdata.uns['n_sub'])
    cdata.uns['q1'] = np.quantile(cdata.uns['diff'], diff_quantile1)
    cdata.uns['q2'] = np.quantile(cdata.uns['diff'], diff_quantile2)
    cdata.uns['fdr_co'] = fdr_co
    fdr = cdata.uns['diff_fdr']
    tf_df = cdata.uns['tf_df']
    diff = cdata.uns['diff']

    c1_only = tf_df.loc[(tf_df.loc[:, np.array(cdata.uns['subset'])[cdata.uns['conditions'] == 1]].sum(1) == \
                         (cdata.uns['conditions'] == 1).sum()) & \
                        (tf_df.loc[:, np.array(cdata.uns['subset'])[cdata.uns['conditions'] == 0]].sum(1) == 0)].index

    c2_only = tf_df.loc[(tf_df.loc[:, np.array(cdata.uns['subset'])[cdata.uns['conditions'] == 1]].sum(1) == 0) & \
                        (tf_df.loc[:, np.array(cdata.uns['subset'])[cdata.uns['conditions'] == 0]].sum(1) == \
                         (cdata.uns['conditions'] == 0).sum())].index

    c1_specific = np.hstack((c1_only, diff[(diff > cdata.uns['q1']) & (fdr < fdr_co) & \
                                           (tf_df.sum(1).isin(_range))].index))

    cdata.uns[c1_name + '_specific'] = pd.Series(c1_specific).drop_duplicates().values
    c2_specific = np.hstack((c2_only, diff[(diff < cdata.uns['q2']) & (fdr < fdr_co) & \
                                           ((tf_df.sum(1).isin(_range)))].index))
    cdata.uns[c2_name + '_specific'] = pd.Series(c2_specific).drop_duplicates().values
    cdata.uns[c1_name + '_only'] = c1_only.values
    cdata.uns[c2_name + '_only'] = c2_only.values


