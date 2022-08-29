===============================================================
SpatialDM: Spatial Direct Messaging Detected by bivariate Moran
===============================================================

About
=====

SpatialDM (Spatial Direct Messaging, or Spatial co-expressed ligand and receptor Detected by Moran's bivariant extension), a statistical model and toolbox to identify the spatial co-expression (i.e., spatial association) between a pair of ligand and receptor. \

Uniquely, SpatialDM can distinguish co-expressed ligand and receptor pairs from spatially separating pairs, and identify the spots of interaction.

.. image:: https://github.com/StatBiomed/SpatialDM/blob/main/docs/.figs/AvsB-1.png?raw=true
   :width: 900px
   :align: center

With the analytical testing method, SpatialDM is scalable to 1 million spots within 12 min with only one core.

.. image:: https://github.com/StatBiomed/SpatialDM/blob/main/docs/.figs/runtime_aug16-1.png?raw=true
   :width: 600px
   :align: center
   
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
  spatialcoord = pd.DataFrame(adata.obsm['spatial'], index=adata.obs_names, columns=['x','y'])
  
  # Preprocessing
  my_sample = sdm.SpatialDM(log, raw, spatialcoord)     # load spatial data with simply log, raw, spatial input
  my_sample.extract_lr(species='human', min_cell=3)
  my_sample.weight_matrix(l=1.2, cutoff=0.2, single_cell=False)  # Not single-cell resolution
  
  # Global selection of significant pairs
  my_sample.spatialdm_global(1000, select_num=None, method='permutation')  # complete in seconds
  my_sample.sig_pairs(method='permutation', fdr=True, threshold=0.1)     # select significant pairs
  pl.global_plot(my_sample, pairs=['CSF1_CSF1R'])  # Overview of global selection
   
  # Local selection of significant spots
  my_sample.spatialdm_local(n_perm=1000, method='both', select_num=None, nproc=1)     # local spot selection complete in seconds
  my_sample.sig_spots(method='permutation', fdr=False, threshold=0.1)     # significant local spots
  pl.plot_pairs(my_sample, ['CSF1_CSF1R'], marker='s') # visualize known melanoma pair(s)
  
.. image:: https://github.com/StatBiomed/SpatialDM/blob/main/docs/.figs/global_plot.png?raw=true
   :width: 200px
   :align: center
   
.. image:: https://github.com/StatBiomed/SpatialDM/blob/main/docs/.figs/csf.png?raw=true
   :width: 600px
   :align: center



Detailed Manual
===============

The full manual is at https://spatialdm.readthedocs.io, including:  

* `Permutation-based SpatialDM (Recommended for small datasets, <10k spots)`_.

* `Analytical z-score-based SpatialDM`_.

* `Differential analyses of whole interactome among varying conditions`_.

.. _Permutation-based SpatialDM (Recommended for small datasets, <10k spots): docs/melanoma.ipynb

.. _Analytical z-score-based SpatialDM: docs/intestine_A1.ipynb

.. _Differential analyses of whole interactome among varying conditions: docs/differential_test_intestine.ipynb




References
==========

SpatialDM manuscript with more details is available on bioRxiv_ now and is currently under review.

.. _bioRxiv: https://www.biorxiv.org/content/10.1101/2022.08.19.504616v1/

