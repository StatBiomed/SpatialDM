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

