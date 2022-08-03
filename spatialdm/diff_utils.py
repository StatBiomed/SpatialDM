import pandas as pd
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import os

def concat_db(samples, species, dir_db):
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
    db=pd.DataFrame({'ligand':np.hstack([sample.ligand for sample in samples]),
                 'receptor':np.hstack([sample.receptor for sample in samples])},
                index=np.hstack([sample.ind for sample in samples]))
    db=db[~db.index.duplicated()]
    geneInter = geneInter.loc[db.index]
    return db, geneInter

class concat_obj(object):
    def __init__(self, samples, names, species, dir_db, method='z-score', fdr=False):
        #     def concat_samples_from_zscore(samples, fdr=True):
        self.db, self.geneInter = concat_db(samples, species, dir_db)
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
        if self.method == 'z-score':
            from sklearn.linear_model import LinearRegression
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


def volcano(self, pairs=None, legend=None, xmax = 25, xmin = -20):
    q1 = self.q1
    q2 = self.q2
    fdr_co = self.fdr_co
    color_codes = [(0.4980392156862745, 0.788235294117647, 0.4980392156862745, 1.0),
                 (0.7450980392156863, 0.6823529411764706, 0.8313725490196079, 1.0),
                 (0.9921568627450981, 0.7529411764705882, 0.5254901960784314, 1.0),
                 (1.0, 1.0, 0.6, 1.0),
                 (0.2196078431372549, 0.4235294117647059, 0.6901960784313725, 1.0),
                 (0.9411764705882353, 0.00784313725490196, 0.4980392156862745, 1.0),
                 (0.7490196078431373, 0.3568627450980392, 0.09019607843137253, 1.0),
                 (0.4, 0.4, 0.4, 1.0)]

    _range = np.arange(1, self.n_sub)
    diff_cp = self.diff.copy()
    diff_cp = np.where((diff_cp>xmax), xmax, diff_cp)
    diff_cp = np.where((diff_cp<xmin), xmin, diff_cp)

    plt.scatter(diff_cp[self.tf_df.sum(1).isin(_range)],
                -np.log10(self.diff_fdr)[self.tf_df.sum(1).isin(_range)], s=10, c='grey')
    plt.xlabel('adult z - fetus z')
    plt.ylabel('diff_cperential fdr (log-likelihood, -log10)')
    plt.xlim([xmin-1,xmax+1])

    plt.scatter(diff_cp[(diff_cp>q1) & (self.diff_fdr<fdr_co) & \
                           (self.tf_df.sum(1).isin(_range))],
                -np.log10(self.diff_fdr)[(diff_cp>q1) & (self.diff_fdr<fdr_co) & \
                           (self.tf_df.sum(1).isin(_range))], s=10,c='tab:orange')
    plt.scatter(diff_cp[(diff_cp<q2) & (self.diff_fdr<fdr_co) & \
                           (self.tf_df.sum(1).isin(_range))],
                -np.log10(self.diff_fdr)[(diff_cp<q2) & (self.diff_fdr<fdr_co)& \
                           (self.tf_df.sum(1).isin(_range))], s=10,c='tab:green')
    if pairs!=None:
        for i,pair in enumerate(pairs):
            plt.scatter(diff_cp[self.p_df.index==pair],
                        -np.log10(self.diff_fdr)[self.p_df.index==pair], c=color_codes[i])
    plt.legend(np.hstack(([''], legend, pairs)))

# plt.savefig('fdr/enrich_volcano.pdf')    # updated