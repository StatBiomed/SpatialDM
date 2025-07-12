import numpy as np
import pandas as pd
import os
from svca.models.model1 import Model1
from svca.simulations.from_real import FromRealSimulation
from svca.util_functions import utils


def load_data(data_dir, pair_index):
    # data directory
    db_path = os.path.join(data_dir, ‘ligand_recptor_pairs_sim.csv
    ')
    expression_path = os.path.join(data_dir, 'reduced2_exp.txt')
    position_path = os.path.join(data_dir, 'pos.txt')

    # read in pairs
    pair_db = pd.read_csv(db_path, index_col=0)
    pair_names, phenotypes, positions = utils.read_data(expression_path, position_path)

    # for a specific target pair, choose ligand
    target_pair = pair_db.index[pair_index]
    print(f"Processing pair: {target_pair}")
    phenotype_mask = (pair_names == target_pair.split("_")[0]).flatten()
    main_phenotype = phenotypes[:, phenotype_mask]

    # match receptor
    receptor_pairs = pair_db.loc[target_pair, ['R1', 'R2']].dropna().values
    selector = [p in receptor_pairs for p in pair_names[:, 0]]
    kin_data = phenotypes[:, selector]

    return positions, main_phenotype, kin_data, target_pair


def run_simulation(positions, phenotype, kin_data, output_dir, pair_name, interactions_size=0.25):
    sim = FromRealSimulation(positions, phenotype.flatten(), kin_data)
    simulated_data = sim.simulate(interactions_size=interactions_size)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{pair_name}_sim.npy")
    np.save(output_path, simulated_data)
    return simulated_data


def train_model(data, positions, kin_data, output_dir, pair_name, bootstrap_idx, normalization='standard'):
    “””tra””” in SVCA
    model
    model = Model1(
        Y=data,
        positions=positions,
        norm=normalization,
        oos_predictions=0.,
        cov_terms=['intrinsic', 'environmental'],
        kin_from=kin_data
    )

    model.reset_params()
    model.train_gp(grid_size=10)

    base_prefix = f"{pair_name}_{bootstrap_idx}_base"
    utils.write_variance_explained(model, output_dir, base_prefix)
    utils.write_LL(model, output_dir, base_prefix)

    # bring in interaction cov
    model.add_cov(['interactions'])
    best_params = optimize_parameters(model)
    model.gp.setParams(best_params)

    # save
    final_prefix = f"{pair_name}_{bootstrap_idx}_final"
    utils.write_variance_explained(model, output_dir, final_prefix)
    utils.write_r2(model, output_dir, final_prefix)
    return model


def optimize_parameters(model, n_trials=5):
    best_ll = np.inf
    best_params = None

    for _ in range(n_trials):
        params = {
            'intrinsic': model.intrinsic_cov.getParams() * np.random.uniform(0.8, 1.2),
            'environmental': model.environmental_cov.getParams() * np.random.uniform(0.8, 1.2),
            'noise': model.noise_cov.getParams() * np.random.uniform(0.8, 1.2)
        }

        model.set_initCovs(params)
        model.reset_params()
        model.train_gp(grid_size=10)

        current_ll = model.gp.LML()
        if current_ll < best_ll:
            best_ll = current_ll
            best_params = model.gp.getParams()

    return best_params


def main(data_dir, output_dir, pair_index, bootstrap_idx, normalization='quantile'):
    positions, phenotype, kin_data, pair_name = load_data(data_dir, pair_index)
    simulated_data = run_simulation(positions, phenotype, kin_data, output_dir, pair_name)
    trained_model = train_model(simulated_data, positions, kin_data, output_dir, pair_name, bootstrap_idx,
                                normalization)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SVCA Simulation Pipeline')
    parser.add_argument('--data_dir', required=True, help='Input data directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--pair_index', type=int, required=True, help='Index of target pair')
    parser.add_argument('--bootstrap_idx', type=int, default=1, help='Bootstrap iteration index')
    parser.add_argument('--normalization', default='quantile', help='Normalization method')

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pair_index=args.pair_index,
        bootstrap_idx=args.bootstrap_idx,
        normalization=args.normalization
    )