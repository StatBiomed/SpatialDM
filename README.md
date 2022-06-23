## Title


### Install

```bash
conda create -n pyname python=3.8

pip install -r requirements.txt
```

### Data Preparation

Prepare the data at `/path/to/src_data`:
```
src_data
├── 0_CellChatDB
│   ├── complex_input_CellChatDB.csv
│   └── t.csv
├── A1.csv
├── A1_celltype.csv
└── A1_tissue_positions_list.csv
```

### Customize weight parameters l and cutoff
The range of signaling is often of hundred μm magnitude. Users should determine reasonable l and cutoff (min) values given the spatial coordinates and spot-spot distance. Users can run the following exemplar steps iteratively to visualize and determine the parameters.

```
def weight_matrix_rbf(spatialcoods, l, min, single_cell):
    dis = (spatialcoods.x.values.reshape(-1, 1) - spatialcoods.x.values) ** 2 + \
          (spatialcoods.y.values.reshape(-1, 1) - spatialcoods.y.values) ** 2
    rbf_d = np.exp(-dis / (2 * l ** 2))  # RBF Distance
    rbf_d[rbf_d < min] = 0
    if single_cell:
        np.fill_diagonal(rbf_d, 0)  # TODO: Optional
    else:
        pass
    return rbf_d

# Intestine data, 100mm center-to-center distance
l = 75 # To be customized
min = 0.2 # To be customized
single_cell = False # set True to exclude ligand & receptor from the same spot
rbf_d = weight_matrix_rbf(spatialcoods, l, min, single_cell)

# Visualize the signaling distance (neighbors)
plt.scatter(spatialcoods.x,spatialcoods.y,c=rbf_d[100],s=80,vmax=0.8,cmap=‘Reds’)
plt.colorbar()
plt.xlim([7800,8200])
plt.ylim([-2100,-1500])
```
![image](https://user-images.githubusercontent.com/52441289/175238071-6d16c234-9508-4d60-a28c-5420b5757501.png)

### Global
```bash
python main.py --data /path/to/src_data --output /path/to/desired/global_result
```


### Local
```bash
python main.py --data /path/to/src_data --output /path/to/desired/local_result --is_local
```

### Option
```
--nproc INT_NUM: using INT_NUM parallel processes (default 50)
--select_num INT_NUM: select first INT_NUM data for fast verification (default 2)
```


### Acknowledge

