import os
import argparse
import pandas as pd
import numpy as np

from utils import load_db, coarse_selection


def get_args_parser():
    parser = argparse.ArgumentParser('TODO: Add description here', add_help=False)
    parser.add_argument('--data', default=".", type=str, help="data location")
    parser.add_argument('--output', default=".", type=str, help="output dir")
    parser.add_argument('--sample', default="sample", type=str, help="sample")
    parser.add_argument('--rbf_l', default=75, type=int, help="rbf parameter l: decide based on spot distance, \
                                                              range of signanling, and spatial coords")
    parser.add_argument('--rbf_co', default=0.2, type=float, help="rbf parameter cut-off: same as r")
    parser.add_argument('--single_cell', default=False, action='store_true', help="whether single cell or not")
    parser.add_argument('--select_num', default=2, type=int, help='selected number of data')

    parser.add_argument('--nproc', default=20, type=int, help="number multi-processing process")
    parser.add_argument('--num_permutation', default=1000, type=int, help="number of permutation")
    parser.add_argument('--dmean', default=True, help="TODO: ")
    parser.add_argument('--is_local', default=False, action='store_true', help="permutation for coarse pair selections & local spots")
    return parser.parse_args()

# compute weight matrix
def weight_matrix_rbf(spatialcoods, l, min, single_cell):
    dis = (spatialcoods.x.values.reshape(-1, 1) - spatialcoods.x.values) ** 2 + \
          (spatialcoods.y.values.reshape(-1, 1) - spatialcoods.y.values) ** 2
    rbf_d = np.exp(-dis / (2 * l ** 2))  # RBF Distance
    rbf_d[rbf_d < min] = 0
    rbf_d = rbf_d * spatialcoods.shape[0] / rbf_d.sum()

    if single_cell:
        np.fill_diagonal(rbf_d, 0)  # TODO: Optional
    else:
        pass
    return rbf_d


def create_output(output_dir, sample):
    result_dir = os.path.join(output_dir, "final_results", sample)
    perm_dir = os.path.join(result_dir, 'perm1k')
    z_dir = os.path.join(result_dir, 'z_score')
    try:
        os.makedirs(perm_dir)
        os.makedirs(z_dir)
    except OSError as e:
        if e.errno != e.errno:
            raise
    return result_dir, perm_dir, z_dir


def main(result_dir, perm_dir, z_dir):
    exp = pd.read_csv(os.path.join(args.data, "{}.csv".format(args.sample)), header=0, index_col=0)
    spatialcoods = pd.read_csv(os.path.join(args.data, '{}_tissue_positions_list.csv'.format(args.sample)), header=None,
                               index_col=0)
    spatialcoods = spatialcoods.loc[spatialcoods[1] == 1, [4, 5]]  # TODO: purpose here
    spatialcoods.columns = ['x', 'y']
    exp = exp.transpose()
    exp = exp.reindex(index=spatialcoods.index)

    # preprocessing
    rbf_d = weight_matrix_rbf(spatialcoods, args.rbf_l, args.rbf_co, single_cell=args.single_cell)
    ligand, receptor, ind = load_db(exp, 'human', min_cell=10, data_root=args.data)

    # TODO: why only select first 100 number
    ligand, receptor, ind = ligand[:args.select_num], receptor[:args.select_num], ind[:args.select_num]

    # no_pairs, no_spots = len(ind), exp.shape[0]
    num_pairs, num_spots = len(ind), exp.shape[0]
    # s1, s2, EI, W, s4, s5 = Moran_var_constant(num_spots, rbf_d)
    rbf_d = rbf_d.astype(np.float16)

    result = coarse_selection(num_pairs, num_spots, rbf_d, ind, z_dir, exp, ligand, receptor, args)
    if args.is_local:
        local_result(result, perm_dir, result_dir)
    else:
        global_result(result, ind, perm_dir)

    return


def global_result(result, ind, perm_dir):
    global_I, Global_PermI = result

    p = 1 - (global_I > Global_PermI).sum(0) / 1000
    pairs = ind[p < 0.05]
    selected = (p < 0.05)

    checkpoint = dict(global_I=global_I, Global_PermI=Global_PermI, p=p,
                      pairs=pairs, selected=selected)

    for k, v in checkpoint.items():
        np.save(perm_dir + '/{}.npy'.format(k), np.array(v, dtype=object))
        print('Successfully save perm_dir/{} ...'.format(k))


def local_result(result, perm_dir, result_dir):
    global_I, Global_PermI, pos, constant, local_I, local_I_R, geary_C, Local_PermI, Local_PermI_R, Geary_Perm = result
    Geary_spots = np.sum((np.expand_dims(geary_C, 1) < Geary_Perm), axis=1)
    Moran_spots = np.sum((np.expand_dims(local_I, 1) > Local_PermI), axis=1)
    Moran_spots_R = np.sum((np.expand_dims(local_I_R, 1) > Local_PermI_R), axis=1)

    Geary_spots = Geary_spots * pos / 2
    Geary_spots[Geary_spots < 900] = 0
    Geary_spots = Geary_spots.astype(float)
    Geary_spots = np.where(np.isnan(Geary_spots), 0, Geary_spots)
    no_Geary_spots = (Geary_spots > 0).sum(axis=1)

    Moran_spots = Moran_spots * pos / 2
    Moran_spots[Moran_spots < 900] = 0
    Moran_spots = Moran_spots.astype(float)
    Moran_spots = np.where(np.isnan(Moran_spots), 0, Moran_spots)
    no_Moran_spots = (Moran_spots > 0).sum(axis=1)

    Moran_spots_R = Moran_spots_R * pos / 2
    Moran_spots_R[Moran_spots_R < 900] = 0
    Moran_spots_R = Moran_spots_R.astype(float)
    Moran_spots_R = np.where(np.isnan(Moran_spots_R), 0, Moran_spots_R)
    no_Moran_spots_R = (Moran_spots_R > 0).sum(axis=1)

    checkpoint = dict(
        global_I=global_I,
        Global_PermI=Global_PermI,
        pos=pos,
        constant=constant,
        local_I=local_I,
        local_I_R=local_I_R,
        geary_C=geary_C,
        Local_PermI=Local_PermI,
        Local_PermI_R=Local_PermI_R,
        Geary_Perm=Geary_Perm,
    )
    for k, v in checkpoint.items():
        np.save(perm_dir + '/{}.npy'.format(k), np.array(v, dtype=object))
        print('Successfully save perm_dir/{} ...'.format(k))

    checkpoint = dict(
        Geary_spots=Geary_spots,
        Moran_spots=Moran_spots,
        Moran_spots_R=Moran_spots_R,
        no_Geary_spots=no_Geary_spots,
        no_Moran_spots=no_Moran_spots,
        no_Moran_spots_R=no_Moran_spots_R
    )
    for k, v in checkpoint.items():
        np.save(result_dir + '/{}.npy'.format(k), np.array(v, dtype=object))
        print('Successfully save result_dir/{} ...'.format(k))


if __name__ == '__main__':
    args = get_args_parser()
    # print(args)
    result_dir, perm_dir, z_dir = create_output(args.output, args.sample)
    # result_dir, perm_dir, z_dir = create_output('output', 'A1')

    # save settings
    with open(result_dir + "settings.txt", "w") as fh:
        msg_setting = f"rbf parameters: l={args.rbf_l} co={args.rbf_co}\n" \
                      f"diagnal weight made zero: {args.single_cell}\n" \
                      f"permutation times: {args.num_permutation}\n" \
                      f"Local: {args.is_local}\n" \
                      f"dmean: {args.dmean}\n" \
                      f"spatial_cols: pixel\n"
        fh.write(msg_setting)

    main(result_dir, perm_dir, z_dir)
