import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import linear_model
import scipy.stats as stats
import seaborn as sns

color_codes = [(0.4980392156862745, 0.788235294117647, 0.4980392156862745, 1.0),
                 (0.7450980392156863, 0.6823529411764706, 0.8313725490196079, 1.0),
                 (0.9921568627450981, 0.7529411764705882, 0.5254901960784314, 1.0),
                 (1.0, 1.0, 0.6, 1.0),
                 (0.2196078431372549, 0.4235294117647059, 0.6901960784313725, 1.0),
                 (0.9411764705882353, 0.00784313725490196, 0.4980392156862745, 1.0),
                 (0.7490196078431373, 0.3568627450980392, 0.09019607843137253, 1.0),
                 (0.4, 0.4, 0.4, 1.0)]

def plt_util(title):
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.colorbar()

def plot_selected_pair(sample, pair, spots, selected_ind, figsize, cmap, cmap_l, cmap_r, **kwargs):
    i = pd.Series(selected_ind == pair).idxmax()
    L = sample.ligand[sample.ind == pair][0]
    R = sample.receptor[sample.ind == pair][0]
    l1, l2 = len(L), len(R)
    plt.figure(figsize=figsize)
    plt.subplot(1, 5, 1)
    plt.scatter(sample.spatialcoord.x, sample.spatialcoord.y, c=spots.loc[pair], cmap=cmap,
                vmax=1, **kwargs)
    plt_util('Moran: ' + str(sample.n_spots[i]) + ' spots')
    for l in range(l1):
        plt.subplot(1, 5, 2 + l)
        plt.scatter(sample.spatialcoord.x, sample.spatialcoord.y, c=sample.logcounts.loc[:,L[l]].values,
                    cmap=cmap_l, **kwargs)
        plt_util('Ligand: ' + L[l])
    for l in range(l2):
        plt.subplot(1, 5, 2 + l1 + l)
        plt.scatter(sample.spatialcoord.x, sample.spatialcoord.y, c=sample.logcounts.loc[:,R[l]],
                    cmap=cmap_r, **kwargs)
        plt_util('Receptor: ' + R[l])

def plot_pairs(sample, pairs_to_plot, pdf=None, figsize=(35, 5),
               cmap='Greens', cmap_l='coolwarm', cmap_r='coolwarm', **kwargs):
    """
    plot selected spots as well as LR expression.
    :param pairs_to_plot: list or arrays. pair name(s), should be from spatialdm_local pairs .
    :param pdf: str. pdf file prefix. save plots in a pdf file.
    :param figsize: figsize for each pair. Default to (35, 5).
    :param markersize: markersize for each spot. Default
    :param cmap: cmap for selected local spots.
    :param cmap_l: cmap for selected ligand. If None, no subplot for ligand expression.
    :param cmap_r: cmap for selected receptor. If None, no subplot for receptor expression
    :return:
    """
    if sample.local_method == 'z-score':
        selected_ind = sample.local_z_p.index
        spots = 1 - sample.local_z_p
    if sample.local_method == 'permutation':
        selected_ind = sample.local_perm_p.index
        spots = 1 - sample.local_perm_p
    if pdf != None:
        with PdfPages(pdf + '.pdf') as pdf:
            for pair in pairs_to_plot:
                plot_selected_pair(sample, pair, spots, selected_ind, figsize, cmap=cmap,
                                   cmap_l=cmap_l, cmap_r=cmap_r, **kwargs)
                pdf.savefig()
                plt.show()
                plt.close()

    else:
        for pair in pairs_to_plot:
            plot_selected_pair(sample, pair, spots, selected_ind, figsize, cmap=cmap,
                               cmap_l=cmap_l, cmap_r=cmap_r, **kwargs)
            plt.show()
            plt.close()

def dot_path(sample, name, pdf=None, figsize=(3,5), **kwargs):
    """
    plot pathway enrichment dotplot.
    :param sample: spatialdm obj
    :param name: str. Either 1) same to one of path_name from compute_pathway;
    2) 'P1', 'P2' for SpatialDE results
    :param pdf: str. save pdf with the specified filename
    :param figsize: figsize
    :return: ax: matplotlib Axes
    """
    plt.figure(figsize=figsize)
    cts = sample.__dict__['path_summary'][name]['counts']
    perc = sample.__dict__['path_summary'][name]['perc']
    plt.scatter(cts.values, cts.index, c=perc.loc[cts.index].values, cmap='Reds', **kwargs)
    plt.xlabel('Number of pairs')
    plt.xticks(np.arange(0,max(cts.values)+2))
    plt.tick_params(axis='y', labelsize=10)
    plt.title(name)
    plt.colorbar(location='bottom', label='percentage of pairs out of CellChatDB')
    plt.tight_layout()
    if pdf != None:
        plt.savefig(pdf+'.pdf')


def corr_plot(x, y, max_num=10000, outlier=0.01, line_on=True, method='spearman',
              legend_on=True, size=30, dot_color=None, outlier_color="r",
              alpha=0.8, color_rate=10, corr_on=None):
    """

    x: `array_like`, (1, )
        Values on x-axis
    y: `array_like`, (1, )
        Values on y-axis
    max_num: int
        Maximum number of dots to plotting by subsampling
    outlier: float
        The proportion of dots as outliers in different color
    line_on : bool
        If True, show the regression line
    method: 'spearman' or 'pearson'
        Method for coefficient R computation
    legend_on: bool
        If True, show the Pearson's correlation coefficient in legend. Replace
        of *corr_on*
    size: float
        The dot size
    dot_color: string
        The dot color. If None (by default), density color will be use
    outlier_color: string
        The color for outlier dot
    alpha : float
        The transparency: 0 (fully transparent) to 1
    color_rate: float
        Color rate for density
    :return:
    ax: matplotlib Axes
        The Axes object containing the plot.
    """
    if method == 'pearson':
        score = stats.pearsonr(x, y)
    if method == 'spearman':
        score = stats.spearmanr(x, y)
    np.random.seed(0)
    if len(x) > max_num:
        idx = np.random.permutation(len(x))[:max_num]
        x, y = x[idx], y[idx]
    outlier = int(len(x) * outlier)

    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)
    idx = z.argsort()
    idx1, idx2 = idx[outlier:], idx[:outlier]

    if dot_color is None:
        c_score = np.log2(z[idx] + color_rate * np.min(z[idx]))
    else:
        c_score = dot_color

    plt.set_cmap("Blues")
    plt.scatter(x[idx], y[idx], c=c_score, edgecolor=None, s=size, alpha=alpha)
    plt.scatter(x[idx2], y[idx2], c=outlier_color, edgecolor=None, s=size / 5,
                alpha=alpha / 3.0)

    if line_on:
        clf = linear_model.LinearRegression()
        clf.fit(x.reshape(-1, 1), y)
        xx = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)
        yy = clf.predict(xx)
        plt.plot(xx, yy, "k--", label="R=%.3f" % score[0])

    if legend_on or corr_on:
        plt.legend(loc="best", fancybox=True, ncol=1)

def global_plot(sample, pairs=None, figsize=(3,4), **kwarg):
    """
    overview of global selected pairs for a SpatialDM obj
    :param sample: SpatialDM obj
    :param pairs: list
    list of pairs to be highlighted in the volcano plot, e.g. ['SPP1_CD44'] or ['SPP1_CD44','ANGPTL4_SDC2']
    :param figsize: tuple
    default to (3,4)
    :param kwarg: plt.scatter arguments
    :return: ax: matplotlib Axes.
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    plt.scatter(np.log1p(sample.global_I), -np.log1p(sample.global_res.perm_pval),
                c=sample.global_res.selected, **kwarg)
    if pairs!=None:
        for i,pair in enumerate(pairs):
            plt.scatter(np.log1p(sample.global_I)[sample.ind==pair],
                        -np.log1p(sample.global_res.perm_pval)[sample.ind==pair],
                        c=color_codes[i])
    plt.xlabel('log1p Global I')
    plt.ylabel('-log1p(pval)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(np.hstack(([''], pairs)))

def differential_dendrogram(sample):
    _range = np.arange(1, sample.n_sub)
    ax = sns.clustermap(1-sample.p_df.loc[(sample.p_val<0.1) & (sample.tf_df.sum(1).isin(_range)),
                                     sample.subset])
    return ax

def differential_volcano(sample, pairs=None, legend=None, xmax = 25, xmin = -20):
    """
    Volcano plot for a differential obj
    :param sample: concatenated Spatial obj
    :param pairs: list
    list of pairs to be highlighted in the volcano plot, e.g. ['SPP1_CD44'] or ['SPP1_CD44','ANGPTL4_SDC2']
    :param legend: list
    list of specified names for each side of the volcano plot
    :param xmax: float
    max z-score difference
    :param xmin: float
    min z-score difference
    :return: ax: matplotlib Axes.
    """
    q1 = sample.q1
    q2 = sample.q2
    fdr_co = sample.fdr_co

    _range = np.arange(1, sample.n_sub)
    diff_cp = sample.diff.copy()
    diff_cp = np.where((diff_cp>xmax), xmax, diff_cp)
    diff_cp = np.where((diff_cp<xmin), xmin, diff_cp)

    plt.scatter(diff_cp[sample.tf_df.sum(1).isin(_range)],
                -np.log10(sample.diff_fdr)[sample.tf_df.sum(1).isin(_range)], s=10, c='grey')
    plt.xlabel('adult z - fetus z')
    plt.ylabel('diff_cperential fdr (log-likelihood, -log10)')
    plt.xlim([xmin-1,xmax+1])

    plt.scatter(diff_cp[(diff_cp>q1) & (sample.diff_fdr<fdr_co) & \
                           (sample.tf_df.sum(1).isin(_range))],
                -np.log10(sample.diff_fdr)[(diff_cp>q1) & (sample.diff_fdr<fdr_co) & \
                           (sample.tf_df.sum(1).isin(_range))], s=10,c='tab:orange')
    plt.scatter(diff_cp[(diff_cp<q2) & (sample.diff_fdr<fdr_co) & \
                           (sample.tf_df.sum(1).isin(_range))],
                -np.log10(sample.diff_fdr)[(diff_cp<q2) & (sample.diff_fdr<fdr_co)& \
                           (sample.tf_df.sum(1).isin(_range))], s=10,c='tab:green')
    if type(pairs)!=type(None):
        for i,pair in enumerate(pairs):
            plt.scatter(diff_cp[sample.p_df.index==pair],
                        -np.log10(sample.diff_fdr)[sample.p_df.index==pair], c=color_codes[i])
    plt.legend(np.hstack(([''], legend, pairs)))

