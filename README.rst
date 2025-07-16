===============================================================
SpatialDM: Spatial Direct Messaging Detected by bivariate Moran
===============================================================

About
=====

SpatialDM (Spatial Direct Messaging, or Spatial co-expressed ligand and 
receptor Detected by Moran's bivariant extension), a statistical model and 
toolbox to identify the spatial co-expression (i.e., spatial association) 
between a pair of ligand and receptor. \

Uniquely, SpatialDM can distinguish co-expressed ligand and receptor pairs from 
spatially separating pairs, and identify the spots of interaction.

.. image:: https://github.com/StatBiomed/SpatialDM/blob/main/docs/.figs/AvsB-1.png?raw=true
   :width: 900px
   :align: center

With the analytical testing method, SpatialDM is scalable to 1 million spots 
within 12 min with only one core.

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

   pip install -U git+https://github.com/StatBiomed/SpatialDM

Installation time: < 1 min



Quick example
=============

Using the build-in melanoma dataset as an example, the following Python script
will compute the p-value indicating whether a certain Ligand-Receptor is 
spatially co-expressed. 


.. code-block:: python

        import spatialdm as sdm
        adata = sdm.datasets.dataset.melanoma()
        sdm.weight_matrix(adata, l=1.2, cutoff=0.2, single_cell=False) # weight_matrix by rbf kernel
        sdm.extract_lr(adata, 'human', min_cell=3)      # find overlapping LRs from CellChatDB
        sdm.spatialdm_global(adata, 1000, specified_ind=None, method='both', nproc=1)     # global Moran selection
        sdm.sig_pairs(adata, method='permutation', fdr=True, threshold=0.1)     # select significant pairs
        sdm.spatialdm_local(adata, n_perm=1000, method='both', specified_ind=None, nproc=1)     # local spot selection
        sdm.sig_spots(adata, method='permutation', fdr=False, threshold=0.1)     # significant local spots

        # visualize global and local pairs
        import spatialdm.plottings as pl
        pl.global_plot(adata, pairs=['SPP1_CD44'])
        pl.plot_pairs(adata, ['SPP1_CD44'], marker='s')
 
.. image:: https://github.com/StatBiomed/SpatialDM/blob/main/docs/.figs/global_plot.png?raw=true
   :width: 200px
   :align: center
   
.. image:: https://github.com/StatBiomed/SpatialDM/blob/main/docs/.figs/SPP1_CD44.png?raw=true
   :width: 600px
   :align: center



Detailed Manual
===============

The full manual is at https://spatialdm.readthedocs.io, including:  

* `Permutation-based SpatialDM (Recommended for small datasets, <10k spots)`_.

* `Differential analyses of whole interactome among varying conditions`_.

.. _Permutation-based SpatialDM (Recommended for small datasets, <10k spots): tutorial/melanoma.ipynb

.. _Differential analyses of whole interactome among varying conditions: tutorial/differential_test_intestine.ipynb




References
==========

| Li, Z., Wang, T., Liu, P., & Huang, Y. (2023). SpatialDM for rapid 
  identification of spatially co-expressed ligand–receptor and revealing 
  cell–cell communication patterns. Nature communications, 14(1), 3995.
  https://www.nature.com/articles/s41467-023-39608-w

