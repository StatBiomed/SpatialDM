===============================================================
SpatialDM: Spatial Direct Messaging Detected by bivariate Moran
===============================================================

About
=====

SpatialDM (**Spatial** <u>D<u>irect **M**essaging, or **Spatial** co-expressed ligand and receptor **D**etected by **M**oran's bivariant extension), a statistical model and toolbox to identify the spatial co-expression (i.e., spatial association) between a pair of ligand and receptor. \

It comprises two main steps: \
1) global selection `spatialdm_global` to identify significantly interacting LR pairs; \
2) local selection `spatialdm_local` to identify local spots for each interaction.

Installation
============

SpatialDM is available through `PyPI <https://pypi.org/project/SpatialDM/>`_. 
To install, type the following command line and add ``-U`` for updates:

.. code-block:: bash

   pip install -U SpatialDM

Alternatively, you can install from this GitHub repository for latest (often 
development) version by the following command line:

.. code-block:: bash

   pip install -U git+https://github.com/leeyoyohku/SpatialDM

Installation time: < 1 min



Quick example
=============

Using the build-in melanoma dataset as an example, the following Python script
will compute the p-value indicating whether a certain Ligand-Receptor is 
spatially co-expressed. 


.. code-block:: python

  import spatialdm as sdm
  import spatialdm.plottings as pl  
  adata = sdm.datasets.melanoma()
  raw = pd.DataFrame(adata.raw.X, index=adata.obs_names, columns=adata.var_names)
  log = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
  my_sample = sdm.SpatialDM(log, raw, spatialcoord)     # load spatial data with simply log, raw, spatial input
  my_sample.extract_lr(species='human', min_cell=3)
  my_sample.weight_matrix(l=1.2, cutoff=0.2, single_cell=False)  # Not single-cell resolution
  my_sample.spatialdm_global(1000, select_num=None, method='permutation')  # complete in seconds
  my_sample.sig_pairs(method='permutation', fdr=True, threshold=0.1)     # select significant pairs
  pl.global_plot(my_sample)  # Overview of global selection
  my_sample.spatialdm_local(n_perm=1000, method='both', select_num=None, nproc=1)     # local spot selection complete in seconds
  my_sample.sig_spots(method='permutation', fdr=False, threshold=0.1)     # significant local spots
  pl.plot_pairs(my_sample, ['CSF1_CSF1R'], marker='s') # visualize known melanoma pair(s)



Detailed Manual
===============

The full manual is at https://spatialdm.readthedocs.io, including 
1) Permutation-based global & local selection in the melanoma data (293 spots, ST platform)
2) Interactions in adult intestine using SpatialDM z-score approach (Visium platform)
3) Differential analyses in the intestine data (8 samples)



References
==========

A BioRxiv preprint will be online soon.
