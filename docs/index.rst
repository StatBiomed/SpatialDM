|PyPI| |Docs| |Build Status|

.. |PyPI| image:: https://img.shields.io/pypi/v/SpatialDM.svg
    :target: https://pypi.org/project/SpatialDM/
.. |Docs| image:: https://readthedocs.org/projects/spatialdm/badge/?version=latest
   :target: https://SpatialDM.readthedocs.io
.. |Build Status| image:: https://travis-ci.org/leeyoyohku/SpatialDM.svg?branch=main
   :target: https://travis-ci.org/leeyoyohku/SpatialDM
   
====
Home
====


About SpatialDM
===============

SpatialDM (Spatial Direct Messaging, or Spatial co-expressed ligand and receptor Detected by Moran's bivariant extension) is a statistical model and toolbox to identify the spatial co-expression (i.e., spatial association) between a pair of ligand and receptor.

Uniquely, SpatialDM can distinguish co-expressed ligand and receptor pairs from spatially separating pairs, and identify the spots of interaction.

.. image:: https://github.com/StatBiomed/SpatialDM/blob/main/docs/.figs/AvsB-1.png?raw=true
   :width: 900px
   :align: center

With the analytical testing method, SpatialDM is scalable to 1 million spots within 12 min with only one core.

.. image:: https://github.com/StatBiomed/SpatialDM/blob/main/docs/.figs/runtime_aug16-1.png?raw=true
   :width: 600px
   :align: center

SpatialDM comprises two main steps: \
 1) global selection with ``spatialdm_global`` to identify significantly interacting LR pairs; \
 2) local selection with ``spatialdm_local`` to identify local spots for each interaction.

Please refer to our tutorials for details:

* `Permutation-based SpatialDM (Recommended for small datasets, <10k spots)`_.

* `Analytical z-score-based SpatialDM`_.

* `Differential analyses of whole interactome among varying conditions`_.

.. _Permutation-based SpatialDM (Recommended for small datasets, <10k spots): melanoma.ipynb

.. _Analytical z-score-based SpatialDM: intestine_A1.ipynb

.. _Differential analyses of whole interactome among varying conditions: differential_test_intestine.ipynb



References
==========
SpatialDM manuscript with more details is available on bioRxiv_ now and is currently under review.

.. _bioRxiv: https://www.biorxiv.org/content/10.1101/2022.08.19.504616v1/



.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:
   
   index
   install
   quick_start
   release

.. toctree::
   :caption: Examples
   :maxdepth: 1
   :hidden:

   melanoma
   intestine_A1
   differential_test_intestine
