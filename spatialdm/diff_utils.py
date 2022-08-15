import pandas as pd
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
import os

def concat_db(samples, species):
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
    db=pd.DataFrame({'ligand':np.hstack([sample.ligand for sample in samples]),
                 'receptor':np.hstack([sample.receptor for sample in samples])},
                index=np.hstack([sample.ind for sample in samples]))
    db=db[~db.index.duplicated()]
    geneInter = geneInter.loc[db.index]
    return db, geneInter

class concat_obj(object):
    def __init__(self, samples, names, species, method='z-score', fdr=False):
        """
        Merge all global results from a list of spatialdm obj
        :param samples: a list of spatialdm obj to be merged.
        :param names: a list of str for each sample's name.
        :param species: str. 'human' or 'mouse'.
        :param dir_db: dir containing 0_CellChatDB folder.
        :param method: 'z-score' or 'permutation'. Should be the commonly selected method from all samples.
        :param fdr: If use fdr or p-values for differential analysis
        """
        self.db, self.geneInter = concat_db(samples, species)
        self.n_samples = len(samples)
        self.method = method
        if method == 'z-score':
            self.p_df = pd.DataFrame(np.zeros((self.db.shape[0], self.n_samples)),
                                     index=self.db.index, columns=names)
            self.zscore_df = self.p_df.copy()
            self.tf_df = self.p_df.copy()
            for sample, d in zip(samples, names):
                sample.z = pd.Series(sample.z, sample.global_res.index)
                if fdr:
                    self.p_df[d] = sample.global_res.fdr
                else:
                    self.p_df[d] = sample.global_res.z_pval
                self.p_df[d] = np.where(np.isnan(self.p_df[d]), 1, self.p_df[d])
                self.zscore_df[d] = sample.z
                self.zscore_df[d] = np.where(np.isnan(self.zscore_df[d]), 0, self.zscore_df[d])
                self.tf_df[d] = sample.global_res.selected

        elif method == 'permutation':
            print('This function to be updated')
            self.p_df = pd.DataFrame(np.zeros((self.db.shape[0], self.n_samples)),
                                     index=self.db.index, columns=names)
            self.tf_df = self.p_df.copy()
            for sample, d in zip(samples, names):
                sample.z = pd.Series(sample.z, sample.global_res.index)
                if fdr:
                    self.p_df[d] = sample.global_res.fdr
                else:
                    self.p_df[d] = sample.global_res.perm_pval
        else:
            raise ValueError("Only one of ['z-score', 'permutation'] is supported")
        return

    def differential_test(self, subset, conditions):
        """
        Test whether each pair is differential among 2 or more conditions
        :param subset: list of concat_obj names to perform differential test on.
        :param conditions:
        :return:
        """
        if self.method == 'z-score':
            import statsmodels.api as sm
            import scipy
            self.subset = subset
            self.conditions = conditions
            n_sub = len(subset)
            self.n_sub = n_sub
            LR_statistic = np.zeros((len(self.p_df)))
            self.p_val = np.zeros((len(self.p_df)))
            x = conditions
            for i in range(len(self.p_df)):
                # full model
                p = self.zscore_df.index[i]
                y = np.where(self.zscore_df.loc[p, subset].isna(), 0, self.zscore_df.loc[p, subset].values)
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
                self.p_val[i] = scipy.stats.chi2.sf(LR_statistic[i], 1)

            self.diff = self.zscore_df.loc[:, np.array(subset)[conditions == 1]].mean(1) - \
                        self.zscore_df.loc[:, np.array(subset)[conditions == 0]].mean(1)

            self.p_val = np.where(np.isnan(self.p_val), 1, self.p_val)

            self.diff_fdr = fdrcorrection(self.p_val)[1]

    def group_differential_pairs(self, c1_name, c2_name, diff_quantile1=0.7, diff_quantile2=0.3, fdr_co=0.1):
        _range = np.arange(1, self.n_sub)
        self.q1 = np.quantile(self.diff, diff_quantile1)
        self.q2 = np.quantile(self.diff, diff_quantile2)
        self.fdr_co = fdr_co
        fdr = self.diff_fdr
        tf_df = self.tf_df
        diff = self.diff

        c1_only = tf_df.loc[(tf_df.loc[:, np.array(self.subset)[self.conditions == 1]].sum(1) == \
                             (self.conditions == 1).sum()) & \
                            (tf_df.loc[:, np.array(self.subset)[self.conditions == 0]].sum(1) == 0)].index

        c2_only = tf_df.loc[(tf_df.loc[:, np.array(self.subset)[self.conditions == 1]].sum(1) == 0) & \
                            (tf_df.loc[:, np.array(self.subset)[self.conditions == 0]].sum(1) == \
                             (self.conditions == 0).sum())].index

        c1_specific = np.hstack((c1_only, diff[(diff > self.q1) & (fdr < fdr_co) & \
                                               (tf_df.sum(1).isin(_range))].index))

        self.__dict__[c1_name + '_specific'] = pd.Series(c1_specific).drop_duplicates()
        c2_specific = np.hstack((c2_only, diff[(diff < self.q2) & (fdr < fdr_co) & \
                                               ((tf_df.sum(1).isin(_range)))].index))
        self.__dict__[c2_name + '_specific'] = pd.Series(c2_specific).drop_duplicates()
        self.__dict__[c1_name + '_only'] = c1_only
        self.__dict__[c2_name + '_only'] = c2_only


