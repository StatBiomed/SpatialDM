============================================================
SpatialDM: Spatail Direct Messaging Detected bivariate Moran
============================================================

About
=====



Installation
============

SpatialDM is available through `PyPI <https://pypi.org/project/spatialdm/>`_. 
To install, type the following command line and add ``-U`` for updates:

.. code-block:: bash

   pip install -U spatialdm

Alternatively, you can install from this GitHub repository for latest (often 
development) version by the following command line:

.. code-block:: bash

   pip install -U git+https://github.com/leeyoyohku/SpatialDM

Installation time: < 1 min



Quick example
=============

Using the buiding melanoma dataset as an example, the following Python script
will compute the p value inidicating whether a certain Ligand-Receptor is 
spatially co-expressed.

.. code-block:: python

  import spatialdm as sdm
  adata = sdm.datasets.melanoma()
  my_sample = sdm.SpatialDM(X=adata.X, spatialcoord=adata.obsm['spatial'])
  my_sample.extract_lr(species='human', min_cell=3)
  my_sample.weight_matrix(l=1.2, cutoff=0.2, single_cell=False)
  my_sample.spatialdm_global(1000, select_num=None, method='permutation')
  sdm.pl.volcano(my_sample)


Detailed Manual
===============

The full manual is at https://spatialdm.readthedocs.io


References
==========

A BioRxiv preprint will be online soon.