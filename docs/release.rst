Release History
===============

TODO
----
- support cache for downloaded h5ad data
- rename local_permI to local_permI_L (also global)
- tentatively: make new adata with LR genes only (if raw exist, other make raw)

Development version
-------------------
- Minor fix with supporting sparse matrix for `adata.X`
- Disabled the output of local permutaiton data in notebook
- More efficient KNN graph construction, with obsp elements into sparse matrix
- Added LR data into package
- Minor fix concat_obj() function
- Minor updates on notebooks: SpatialDE limited one CPU; diff-test only z-score
- Suggest adding `adata.obsm['celltypes']` dataframe to replace `adata.obs`

Version 0.0.8 (14/03/2023)
--------------------------

- SpatialDM wrapped into AnnData object, fixed typos

Version 0.0.1 (11/08/2022)
--------------------------

- Alpha version of SpatialDM released
