API
===

.. automodule:: spatialdm

Import SpatialDM as::

   import spatialdm as sdm


Functions
---------

.. autofunction:: spatialdm.stats.rbfweight

.. autofunction:: spatialdm.stats.Moran_R




Datasets
--------
The `spatialdm.datasets` module provides functions for loading and accessing spatial transcriptomics datasets. The following datasets are currently available:

* `dataset.melanoma()`: Sample 1 rep 2 human melanoma slide from Thrane's melanoma dataset.
* `dataset.SVZ()`: Mouse sub-ventricular zone (SVZ) from Eng's seqfish+ dataset.
* `dataset.A1()`: Adult colon with colorectal cancer or IBD, pcw: Adult.
* `dataset.A2()`: Adult colon with colorectal cancer or IBD, pcw: Adult.
* `dataset.A3()`: Fetal colon, pcw: 12PCW.
* `dataset.A4()`: Fetal colon, pcw: 19PCW.
* `dataset.A6()`: Fetal small intestine, pcw: 12PCW.
* `dataset.A7()`: Fetal small intestine, pcw: 12PCW.
* `dataset.A8()`: Fetal small intestine, pcw: 12PCW.
* `dataset.A9()`: Fetal small intestine, pcw: 12PCW.

Usage
-----

To use the `spatialdm.datasets` module, simply import it as follows:

.. code-block:: python

    from spatialdm.datasets import dataset

Then, you can load a dataset using the corresponding function. For example, to load the melanoma dataset:

.. code-block:: python

    adata = dataset.melanoma()

This will return an `anndata` object containing the expression data for the melanoma dataset in `.X`, the cell type decomposition values in `.obs`, and the spatial coordinates in `.obsm['spatial']`.

.. autosummary::
   :toctree: _autosummary
   weight_matrix()
   extract_lr()
   datasets.dataset.A1()
   datasets.dataset.A11()
   datasets.dataset.A2()
