import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

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