import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import linear_model
import scipy.stats as stats
import seaborn as sns
#from utils import compute_pathway
from .utils import *
import holoviews as hv
from holoviews import opts, dim
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.io import export_svg, export_png
from bokeh.layouts import gridplot
from scipy.sparse import csc_matrix

hv.extension('bokeh')
hv.output(size=200)

import math
from matplotlib.cm import hsv

def generate_colormap(number_of_distinct_colors, number_of_shades = 7):
#     https://stackoverflow.com/questions/42697933/colormap-with-maximum-distinguishable-colours
    number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors \
            / number_of_shades) * number_of_shades)

    linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / \
            number_of_distinct_colors_with_multiply_of_shades
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)
    arr_by_shade_columns = arr_by_shade_rows.T
    number_of_partitions = arr_by_shade_columns.shape[0]
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)
    initial_cm = hsv(nums_distributed_like_rising_saw)
    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier
    initial_cm = initial_cm[:,:3] * 255
    initial_cm = initial_cm.astype(int)
    initial_cm = np.array(['#%02x%02x%02x' % tuple(initial_cm[i]) for i in range(len(initial_cm))])
    return initial_cm

def ligand_ct(adata, pair):
    ct_L = adata.uns['local_stat']['local_I'][:,adata.uns['selected_spots'].index==pair] * adata.obs
    return ct_L

def receptor_ct(adata, pair):
    ct_R = adata.uns['local_stat']['local_I_R'][:,adata.uns['selected_spots'].index==pair] * adata.obs
    return ct_R

def chord_celltype(adata, pairs, color_dic=None, title=None, min_quantile=0.5, ncol=2, save=None):
    file_format = save.split('.')[-1]
    if color_dic is None:
        ct = adata.obs.columns.sort_values()
        l = len(ct)
        gen_col = generate_colormap(l)[:l]
        color_dic = {ct[i]: gen_col[i] for i in range(len(ct))}
    ls = []
    #     if type(min_link) != list:
    #         min_link = np.repeat(min_link, len(pairs))
    if type(min_quantile) is float:
        min_quantile = np.repeat(min_quantile, len(pairs))
    for i, pair in enumerate(pairs):
        if title is None:
            t = pair
        type_interaction = adata.uns['geneInter'].loc[pair, 'annotation']
        if type_interaction == 'Secreted Signaling':
            w = adata.obsp['weight']
        else:
            w = adata.obsp['nearest_neighbors']

        ct_L = ligand_ct(adata, pair)
        ct_R = receptor_ct(adata, pair)

        sparse_ct_sum = [[(csc_matrix(w).multiply(ct_L[n1].values).T.multiply(ct_R[n2].values)).sum() \
                          for n1 in ct_L.columns] for n2 in ct_R.columns]
        sparse_ct_sum = np.array(sparse_ct_sum)

        Links = pd.DataFrame({'source': np.tile(ct_L.columns, ct_R.shape[1]),
                              'target': np.repeat(ct_R.columns, ct_L.shape[1]),
                              'value': sparse_ct_sum.reshape(1, -1)[0]})

        Nodes = pd.DataFrame({'name': ct_L.columns})
        Nodes.index = Nodes.name.values
        nodes = hv.Dataset(Nodes, 'index')

        chord = hv.Chord((Links.loc[Links.value > 0], nodes)).select(  # Links.value>min_link[i]
            value=(Links.value.quantile(min_quantile[i]), None))
        cmap_ct = pd.Series(color_dic)[chord.nodes.data['index'].values].values.tolist()
        adata.uns[pair + '_link'] = Links
        chord.opts(
            opts.Chord(  # cmap='Category20',
                edge_cmap=cmap_ct,
                edge_color=dim('source').str(),
                labels='name', node_color=dim('index').str(),
                node_cmap=cmap_ct,
                title=t))
        ls.append(chord)

    ar = np.array([hv.render(fig) for fig in ls])
    for n in ar:
        n.output_backend = "svg"
    plots = ar.reshape(-1, ncol).tolist()
    grid = gridplot(plots)
    if file_format == 'svg':
        export_svg(grid, filename=save)
    elif file_format == 'png':
        export_png(grid, filename=save)
    show(grid)
    return grid

def chord_LR(adata, senders, receivers, color_dic=None,
             title=None, min_quantile=0.5, ncol=2, save=None):
    file_format = save.split('.')[-1]
    if color_dic is None:
        subgeneInter = adata.uns['geneInter'].loc[adata.uns['selected_spots'].index]
        type_interaction = subgeneInter.annotation
        n_short_lri = (type_interaction!='Secreted Signaling').sum()
        ligand_all = subgeneInter.interaction_name_2.str.split('-').str[0]
        receptor_all = subgeneInter.interaction_name_2.str.split('-').str[1]
        genes_all = np.hstack((ligand_all, receptor_all))
        genes_all = pd.Series(genes_all).drop_duplicates().values
        l = len(genes_all)
        gen_col = generate_colormap(l)[:l]
        color_dic = {genes_all[i]: gen_col[i] for i in range(l)}

    ls = []
    #     if type(min_link) != list:
    #         min_link = np.repeat(min_link, len(pairs))
    if type(min_quantile) is float:
        min_quantile = np.repeat(min_quantile, len(senders))

    for i, (sender, receiver) in enumerate(zip(senders, receivers)):
        if title is None:
            t = ('_').join((sender, receiver))

        ct_L = adata.obs.loc[:,sender].values * adata.uns['local_stat']['local_I'].T
        ct_R = adata.obs.loc[:,receiver].values * adata.uns['local_stat']['local_I_R'].T

        sparse_ct_sum = np.hstack(([csc_matrix(adata.obsp['nearest_neighbors']).multiply(n1).T.multiply(n2).sum() \
                      for n1,n2 in zip(ct_L[:n_short_lri], ct_R[:n_short_lri])],
                                  [csc_matrix(adata.obsp['weight']).multiply(n1).T.multiply(n2).sum() \
                      for n1,n2 in zip(ct_L[n_short_lri:], ct_R[n_short_lri:])]))


        Links = pd.DataFrame({'source':ligand_all,
                    'target':receptor_all,
                  'value': sparse_ct_sum})
        adata.uns[t+'_link'] = Links

        Nodes = pd.DataFrame({'name': genes_all.astype(str)})
        Nodes.index = Nodes.name.values

        Nodes=Nodes.drop_duplicates()

        nodes = hv.Dataset(Nodes, 'index')

        chord = hv.Chord((Links.loc[Links.value>0], nodes)).select(
            value=(Links.value.quantile(min_quantile).drop_duplicates().values, None))

        cmap_ct = pd.Series(color_dic)[chord.nodes.data['index'].values].values.tolist()

        chord.opts(
            opts.Chord(#cmap='Category20',
                        edge_cmap=cmap_ct,
                       edge_color=dim('source').str(),
                       labels='name', node_color=dim('index').str(),
                       node_cmap=cmap_ct,
                       title = 'Undifferentiated_Colonocytes'))
        ls.append(chord)

    ar = np.array([hv.render(fig) for fig in ls])
    for n in ar:
        n.output_backend="svg"
    plots = ar.reshape(-1,ncol).tolist()
    grid = gridplot(plots)

    if file_format=='svg':
        export_svg(grid, filename=save)
    elif file_format=='png':
        export_png(grid, filename=save)
    show(grid)
    return grid

def chord_celltype_allpairs(adata, color_dic=None,
                             min_quantile=0.9, ncol=3, save=None):
    file_format = save.split('.')[-1]
    if color_dic is None:
        ct = adata.obs.columns.sort_values()
        l = len(ct)
        gen_col = generate_colormap(l)[:l]
        color_dic = {ct[i]: gen_col[i] for i in range(len(ct))}

    long_pairs = adata.uns['geneInter'][adata.uns['geneInter'].annotation == \
                    'Secreted Signaling'].index.intersection(adata.uns['selected_spots'].index)
    short_pairs = adata.uns['geneInter'][adata.uns['geneInter'].annotation != \
                        'Secreted Signaling'].index.intersection(adata.uns['selected_spots'].index)
    ls=[]

    for by_range,pairs,w in zip(['long', 'short'],
                    [long_pairs, short_pairs],
                 [adata.obsp['weight'], adata.obsp['nearest_neighbors']]):
        sparse_ct_sum = [[[(csc_matrix(w).multiply(ligand_ct(adata, p)[n1].values).T.multiply(receptor_ct(adata, p)[n2].values)).sum() \
           for n1 in ct] for n2 in ct] for p in pairs]
        sparse_ct_sum = np.array(sparse_ct_sum).sum(0)

        Links = pd.DataFrame({'source':np.tile(ct, l),
                    'target':np.repeat(ct, l),
                  'value': sparse_ct_sum.reshape(1,-1)[0]})
        adata.uns[by_range]=Links

        Nodes = pd.DataFrame({'name': ct})
        Nodes.index = Nodes.name.values
        nodes = hv.Dataset(Nodes, 'index')

        chord = hv.Chord((Links.loc[Links.value>0], nodes)).select( #Links.value>min_link[i]
            value=(Links.value.quantile(min_quantile), None))
        cmap_ct = pd.Series(color_dic)[chord.nodes.data['index'].values].values.tolist()
        chord.opts(
            opts.Chord(#cmap='Category20',
                        edge_cmap=cmap_ct,
                       edge_color=dim('source').str(),
                       labels='name', node_color=dim('index').str(),
                       node_cmap=cmap_ct,
                       title = by_range))
        ls.append(chord)

    value = (len(long_pairs) * adata.uns['long'].value + len(short_pairs) * adata.uns['short'].value)/ \
            (len(long_pairs) + len(short_pairs))
    Links.value = value
    chord = hv.Chord((Links.loc[Links.value>0], nodes)).select( #Links.value>min_link[i]
            value=(Links.value.quantile(min_quantile), None))
    cmap_ct = pd.Series(color_dic)[chord.nodes.data['index'].values].values.tolist()
    chord.opts(
        opts.Chord(#cmap='Category20',
                    edge_cmap=cmap_ct,
                   edge_color=dim('source').str(),
                   labels='name', node_color=dim('index').str(),
                   node_cmap=cmap_ct,
                   title = 'Cell_type_interactions_between_all_identified_pairs'))
    ls.append(chord)

    ar = np.array([hv.render(fig) for fig in ls])
    for n in ar:
        n.output_backend="svg"
    plots = ar.reshape(-1,ncol).tolist()
    grid = gridplot(plots)

    if file_format=='svg':
        export_svg(grid, filename=save)
    elif file_format=='png':
        export_png(grid, filename=save)
    show(grid)


def plt_util(title):
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.colorbar()


def plot_selected_pair(sample, pair, spots, selected_ind, figsize, cmap, cmap_l, cmap_r, **kwargs):
    i = pd.Series(selected_ind == pair).idxmax()
    L = sample.uns['ligand'].loc[pair].dropna().values
    R = sample.uns['receptor'].loc[pair].dropna().values
    l1, l2 = len(L), len(R)
    plt.figure(figsize=figsize)
    plt.subplot(1, 5, 1)
    plt.scatter(sample.obsm['spatial'][:,0], sample.obsm['spatial'][:,1], c=spots.loc[pair], cmap=cmap,
                vmax=1, **kwargs)
    plt_util('Moran: ' + str(sample.uns['local_stat']['n_spots'][pair]) + ' spots')
    for l in range(l1):
        plt.subplot(1, 5, 2 + l)
        plt.scatter(sample.obsm['spatial'][:,0], sample.obsm['spatial'][:,1], c=sample[:,L[l]].X,
                    cmap=cmap_l, **kwargs)
        plt_util('Ligand: ' + L[l])
    for l in range(l2):
        plt.subplot(1, 5, 2 + l1 + l)
        plt.scatter(sample.obsm['spatial'][:,0], sample.obsm['spatial'][:,1], c=sample[:,R[l]].X,
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
    if sample.uns['local_stat']['local_method'] == 'z-score':
        selected_ind = sample.uns['local_z_p'].index
        spots = 1 - sample.uns['local_z_p']
    if sample.uns['local_stat']['local_method'] == 'permutation':
        selected_ind = sample.uns['local_perm_p'].index
        spots = 1 - sample.uns['local_perm_p']
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

from matplotlib import gridspec
def make_grid_spec(
    ax_or_figsize,
    nrows: int,
    ncols: int,
    wspace= None,
    hspace = None,
    width_ratios = None,
    height_ratios= None,
):
    kw = dict(
        wspace=wspace,
        hspace=hspace,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
    )
    if isinstance(ax_or_figsize, tuple):
        fig = plt.figure(figsize=ax_or_figsize)
        return fig, gridspec.GridSpec(nrows, ncols, **kw)
    else:
        ax = ax_or_figsize
        ax.axis('off')
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax.figure, ax.get_subplotspec().subgridspec(nrows, ncols, **kw)

def dot_path(adata, uns_key=None, dic=None, cut_off=1, groups=None, markersize = 50,
             legend_size=(8, 3), figsize=(6,8),
             **kwargs):
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
    if uns_key is not None:
        dic = {uns_key: adata.uns[uns_key]}
    pathway_res = compute_pathway(adata, dic=dic)
    pathway_res = pathway_res[pathway_res.selected >= cut_off]
    if groups is not None:
        pathway_res = pathway_res.loc[pathway_res.name.isin(groups)]
    n_subplot = len(pathway_res.name.unique())
    for i, name in enumerate(pathway_res.name.unique()):
        plt.figure(figsize=figsize)
        plt.subplot((n_subplot + 1) // 2, 2, i + 1)
        result1 = pathway_res.loc[pathway_res.name == name]
        result1 = result1.sort_values('selected', ascending=False)
        cts = result1.selected
        perc = result1.selected / result1.pathway_size
        value = -np.log10(result1.loc[:, 'fisher_p'].values)
        size = value * markersize
        plt.scatter(result1.selected.values, result1.index, c=perc.loc[result1.index].values,
                    s=size, cmap='Reds')
        plt.xlabel('Number of pairs')
        plt.xticks(np.arange(0, max(result1.selected.values) + 2))
        plt.tick_params(axis='y', labelsize=10)
        plt.title(name)
        plt.colorbar(location='bottom', label='percentage of pairs out of CellChatDB')
        plt.tight_layout()

        fig, legend_gs = make_grid_spec(
            legend_size,
            nrows=4, ncols=1
        )

        # plot size bar
        size_uniq = np.quantile(size, np.arange(1, 0, -0.1))
        value_uniq = np.quantile(value, np.arange(1, 0, -0.1))
        size_range = value_uniq
        size_legend_ax = fig.add_subplot(legend_gs[1])
        size_legend_ax.scatter(
            np.arange(len(size_uniq)) + 0.5,
            np.repeat(0, len(size_uniq)),
            s=size_uniq,
            color='gray',
            edgecolor='black',
            zorder=100,
        )
        size_legend_ax.set_xticks(np.arange(len(value_uniq)) + 0.5)
        # labels = [
        #     "{}".format(np.round((x * 100), decimals=0).astype(int)) for x in size_range
        # ]
        size_legend_ax.set_xticklabels(np.round(np.exp(-value_uniq), 3), fontsize='small')

        # remove y ticks and labels
        size_legend_ax.tick_params(
            axis='y', left=False, labelleft=False, labelright=False
        )

        # remove surrounding lines
        size_legend_ax.spines['right'].set_visible(False)
        size_legend_ax.spines['top'].set_visible(False)
        size_legend_ax.spines['left'].set_visible(False)
        size_legend_ax.spines['bottom'].set_visible(False)
        size_legend_ax.grid(False)

        ymax = size_legend_ax.get_ylim()[1]
        size_legend_ax.set_title('fisher exact p-value (right tile)', y=ymax + 0.9, size='small')

        xmin, xmax = size_legend_ax.get_xlim()
        size_legend_ax.set_xlim(xmin - 0.15, xmax + 0.5)


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
    if pairs is not None:
        color_codes = generate_colormap(max(10, len(pairs)+2))[2:]
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    plt.scatter(np.log1p(sample.uns['global_I']), -np.log1p(sample.uns['global_res'].perm_pval),
                c=sample.uns['global_res'].selected, **kwarg)
    if pairs!=None:
        for i,pair in enumerate(pairs):
            plt.scatter(np.log1p(sample.uns['global_I'])[sample.uns['ligand'].index==pair],
                        -np.log1p(sample.uns['global_res'].perm_pval)[sample.uns['ligand'].index==pair],
                        c=color_codes[i])
    plt.xlabel('log1p Global I')
    plt.ylabel('-log1p(pval)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(np.hstack(([''], pairs)))

def differential_dendrogram(sample):
    _range = np.arange(1, sample.uns['n_sub'])
    ax = sns.clustermap(1-sample.uns['p_df'].loc[(sample.uns['p_val']<0.1) & (sample.uns['tf_df'].sum(1).isin(_range)),
                                     sample.uns['subset']])
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
    q1 = sample.uns['q1']
    q2 = sample.uns['q2']
    fdr_co = sample.uns['fdr_co']

    _range = np.arange(1, sample.uns['n_sub'])
    diff_cp = sample.uns['diff'].copy()
    diff_cp = np.where((diff_cp>xmax), xmax, diff_cp)
    diff_cp = np.where((diff_cp<xmin), xmin, diff_cp)

    plt.scatter(diff_cp[sample.uns['tf_df'].sum(1).isin(_range)],
                -np.log10(sample.uns['diff_fdr'])[sample.uns['tf_df'].sum(1).isin(_range)], s=10, c='grey')
    plt.xlabel('adult z - fetus z')
    plt.ylabel('differential fdr (log-likelihood, -log10)')
    plt.xlim([xmin-1,xmax+1])

    plt.scatter(diff_cp[(diff_cp>q1) & (sample.uns['diff_fdr']<fdr_co) & \
                           (sample.uns['tf_df'].sum(1).isin(_range))],
                -np.log10(sample.uns['diff_fdr'])[(diff_cp>q1) & (sample.uns['diff_fdr']<fdr_co) & \
                           (sample.uns['tf_df'].sum(1).isin(_range))], s=10,c='tab:orange')
    plt.scatter(diff_cp[(diff_cp<q2) & (sample.uns['diff_fdr']<fdr_co) & \
                           (sample.uns['tf_df'].sum(1).isin(_range))],
                -np.log10(sample.uns['diff_fdr'])[(diff_cp<q2) & (sample.uns['diff_fdr']<fdr_co)& \
                           (sample.uns['tf_df'].sum(1).isin(_range))], s=10,c='tab:green')
    if type(pairs)!=type(None):
        for i,pair in enumerate(pairs):
            plt.scatter(diff_cp[sample.uns['p_df'].index==pair],
                        -np.log10(sample.uns['diff_fdr'])[sample.uns['p_df'].index==pair], c=color_codes[i])
    plt.legend(np.hstack(([''], legend, pairs)))

