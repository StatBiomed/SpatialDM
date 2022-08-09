import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import linear_model
import scipy.stats as stats

def plt_util(title):
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.colorbar()

def plot_selected_pair(sample, pair, spots, selected_ind, figsize, markersize, cmap, cmap_l, cmap_r):
    i = pd.Series(selected_ind == pair).idxmax()
    L = sample.ligand[sample.ind == pair][0]
    R = sample.receptor[sample.ind == pair][0]
    l1, l2 = len(L), len(R)
    plt.figure(figsize=figsize)
    if (cmap!=None):
        plt.subplot(1, 5, 1)
        plt.scatter(sample.spatialcoord.x, sample.spatialcoord.y, marker='s', c=spots.loc[pair], cmap=cmap,
                    vmax=1, s=markersize, linewidth=0)
        plt_util('Moran: ' + str(sample.n_spots[i]) + ' spots')
    if (cmap_l!=None) & (cmap_r!=None):
        for l in range(l1):
            plt.subplot(1, 5, 2 + l)
            plt.scatter(sample.spatialcoord.x, sample.spatialcoord.y, marker='s', c=sample.exp.loc[:,L[l]].values, cmap=cmap_l,
                        s=markersize, linewidth=0)
            plt_util('Ligand: ' + L[l])
        for l in range(l2):
            plt.subplot(1, 5, 2 + l1 + l)
            plt.scatter(sample.spatialcoord.x, sample.spatialcoord.y, marker='s', c=sample.exp.loc[:,R[l]], cmap=cmap_r,
                        s=markersize, linewidth=0)
            plt_util('Receptor: ' + R[l])

def plot_pairs(sample, pairs_to_plot, pdf=None, figsize=(35, 5), markersize=18,
               cmap='Greens', cmap_l='coolwarm', cmap_r='coolwarm'):
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
                plot_selected_pair(sample, pair, spots, selected_ind, figsize, markersize, cmap=cmap,
                                   cmap_l=cmap_l, cmap_r=cmap_r)
                pdf.savefig()
                plt.show()
                plt.close()

    else:
        for pair in pairs_to_plot:
            plot_selected_pair(sample, pair, spots, selected_ind, figsize, markersize, cmap=cmap,
                               cmap_l=cmap_l, cmap_r=cmap_r)
            plt.show()
            plt.close()

def dot_path(sample, name, pdf=None, figsize=(3,5)):
    """
    plot pathway enrichment dotplot.
    :param sample: spatialdm onj
    :param name: str. Either 1) same to one of path_name from compute_pathway;
    2) 'P1', 'P2' for SpatialDE results
    :param pdf: str. save pdf with the specified filename
    :param figsize: figsize
    :return: pathway enrichment dotplot
    """
    plt.figure(figsize=figsize)
    cts = sample.__dict__['path_summary'][name]['counts']
    perc = sample.__dict__['path_summary'][name]['perc']
    plt.scatter(cts.values, cts.index, c=perc.loc[cts.index].values, cmap='Reds')
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
        # c_score = np.log2(z[idx]+100)
        c_score = np.log2(z[idx] + color_rate * np.min(z[idx]))
    else:
        # idx2 = []
        c_score = dot_color

    plt.set_cmap("Blues")
    plt.scatter(x[idx], y[idx], c=c_score, edgecolor=None, s=size, alpha=alpha)
    plt.scatter(x[idx2], y[idx2], c=outlier_color, edgecolor=None, s=size / 5,
                alpha=alpha / 3.0)  # /5

    # plt.grid(alpha=0.4)

    if line_on:
        clf = linear_model.LinearRegression()
        clf.fit(x.reshape(-1, 1), y)
        xx = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)
        yy = clf.predict(xx)
        plt.plot(xx, yy, "k--", label="R=%.3f" % score[0])
        # plt.plot(xx, yy, "k--")

    if legend_on or corr_on:
        plt.legend(loc="best", fancybox=True, ncol=1)
