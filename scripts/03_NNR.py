
import os
from time import perf_counter
from independent_null_models import run_group_nulls
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg', force=True)
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as multi
from statsmodels.stats.multitest import multipletests
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from node_neighbor_correlation import calculate_neighbor_correlation_for_subject
from plotting_functions import create_all_raincloud_plots


def save_csv_atomic(dataframe, output_file, index=False, compression=None):
    temporary_file = f"{output_file}.tmp"
    dataframe.to_csv(
        temporary_file, index=index,
        compression=compression if compression is not None else 'infer',
    )
    os.replace(temporary_file, output_file)


def sparsify_fc_by_density(fc_matrix, density):
    upper_rows, upper_cols = np.triu_indices_from(fc_matrix, k=1)
    upper_weights = fc_matrix[upper_rows, upper_cols]
    positive_indices = np.flatnonzero(upper_weights > 0)

    n_keep = int(np.ceil(positive_indices.size * density))
    ranked_indices = positive_indices[
        np.argsort(upper_weights[positive_indices])[::-1]
    ]
    kept_indices = ranked_indices[:n_keep]

    sparse_fc = np.zeros_like(fc_matrix)
    kept_rows = upper_rows[kept_indices]
    kept_cols = upper_cols[kept_indices]
    sparse_fc[kept_rows, kept_cols] = fc_matrix[kept_rows, kept_cols]
    sparse_fc[kept_cols, kept_rows] = fc_matrix[kept_cols, kept_rows]
    np.fill_diagonal(sparse_fc, 0)

    return sparse_fc


def calculate_all_network_NNR_for_subject(subject_data, networks):

    results = {}
    for network_name, network_matrix in networks.items():
        correlation, p_value = calculate_neighbor_correlation_for_subject( subject_data, network_matrix)
        results[f'{network_name}_NNR'] = correlation
        results[f'{network_name}_pvalue'] = p_value
    return results

def plot_empirical_spin_rewired_comparison(summary_df, null_df, networks, group_labels, output_path):
    front_size = 17

    colors = {'Spin': '#3979AD', 'BrainSMASH': '#4AC532', 'Rewired': '#D543D7'}
    offsets = {'Spin': -0.25, 'BrainSMASH': 0, 'Rewired': 0.25}
    model_labels = {'Spin': 'Null-spin', 'BrainSMASH': 'Null-smash', 'Rewired': 'Null-rewired'}
    names = [name for name in networks if name != 'Euclidean']
    desired_order = ['Aβ+ CN-NC', 'Aβ+ CN-C', 'Aβ+ MCI-NC', 'Aβ+ MCI-C', 'Aβ+ AD']

    label_to_group = {label: group for group, label in group_labels.items()}
    ordered_groups = [(label_to_group[label], label) for label in desired_order]

    fig, axes = plt.subplots(len(ordered_groups), 1, figsize=(max(14, len(names) * 0.65), 16),
                             squeeze=False, sharey=True)

    for ax, (group, label) in zip(axes[:, 0], ordered_groups):
        group_nulls = null_df.loc[
            (null_df.Group == group) & (null_df.Network != 'Euclidean'), 'Correlation'
        ].dropna()
        y_min = group_nulls.min() if not group_nulls.empty else -0.9
        star_y = y_min - 0.05

        for pos, name in enumerate(names):
            row = summary_df[(summary_df.Group == group) & (summary_df.Network == name)]
            if row.empty:
                continue

            empirical_NNR = row.Correlation.iloc[0]
            empirical_x = []

            for model, color in colors.items():
                box_x = pos + offsets[model]
                values = null_df.loc[
                    (null_df.Group == group) & (null_df.Network == name)
                    & (null_df.Null_Type == model), 'Correlation'
                ].dropna()
                if values.empty:
                    continue

                ax.boxplot([values], positions=[box_x], widths=0.20, showfliers=False,
                           manage_ticks=False, boxprops={'color': color},
                           whiskerprops={'color': color}, capprops={'color': color},
                           medianprops={'color': color})
                empirical_x.append(box_x)

                p_column = f'Pvalue_{model}_FDR'
                if p_column in row.columns:
                    p_value = row[p_column].iloc[0]
                    if np.isfinite(p_value) and p_value < 0.05:
                        ax.text(box_x, star_y, '*', ha='center', va='bottom', color=color,
                                fontsize=front_size, fontweight='bold', clip_on=True, zorder=6)

            if empirical_x:
                ax.plot(empirical_x, [empirical_NNR] * len(empirical_x), color='red',
                        marker='o', markersize=4.5, linewidth=1.2, zorder=5)

        ax.set_title(label, loc='center', fontsize=front_size)
        ax.set_ylabel('Node-neighbor\nrelationship', fontsize=front_size)
        ax.set_xticks(range(len(names)), names)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylim(-0.9, 0.9)
        ax.yaxis.set_major_locator(MultipleLocator(0.4))
        ax.tick_params(axis='both', labelsize=front_size)
        ax.xaxis.get_offset_text().set_fontsize(front_size)
        ax.yaxis.get_offset_text().set_fontsize(front_size)

    legend_handles = [Patch(facecolor='white', edgecolor=color, label=model_labels[model])
                      for model, color in colors.items()]
    legend_handles.append(Line2D([0], [0], color='black', marker='o', markersize=4,
                                 linewidth=1.2, label='Empirical'))
    axes[0, 0].legend(handles=legend_handles, loc='upper right', ncol=4, fontsize=front_size)

    fig.tight_layout()
    fig.savefig(os.path.join(output_path, 'group_empirical_three_nulls_comparison.png'),
                dpi=1000, bbox_inches='tight')
    plt.close(fig)


def main():
    path = r''
    filewscore = os.path.join(path, 'results', '2_Wscore', 'W_scores_threshold0.csv')
    filedemo = os.path.join(path, 'results', '2_Wscore', 'W_scores_demo.csv')
    Dir_net = os.path.join(path, 'resources', 'connect')
    
    Dir_Atlas = os.path.join(path, 'resources', 'atlas')
    LH_ANNOT = os.path.join(Dir_Atlas, 'lh.Schaefer2018_400Parcels_7Networks_order.annot')
    RH_ANNOT = os.path.join(Dir_Atlas, 'rh.Schaefer2018_400Parcels_7Networks_order.annot')
    coords_file = os.path.join(Dir_Atlas, 'Schaefer400_coords.txt')

    outpath = os.path.join(path, 'results', '3_NNR', '1_cross_section')
    group_level_outpath = os.path.join(outpath, '1_group_level_analysis')
    subject_level_outpath = os.path.join(outpath, '2_subject_level_analysis')
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(group_level_outpath, exist_ok=True)
    os.makedirs(subject_level_outpath, exist_ok=True)

    SPIN_N_PERM = 5000
    BRAINSMASH_N_PERM = 5000
    REWIRED_N_PERM = 5000
    REWIRED_N_BINS = 10
    REWIRED_SWAPS_PER_EDGE = 1
    NULL_MODEL_SEEDS = 42
    N_JOBS = 160
    BRAINSMASH_OPTIONS = dict(deltas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                             kernel='exp', pv=25, nh=25, resample=False, n_jobs=N_JOBS)

    group_labels = {
        1: 'Aβ+ CN-NC',
        2: 'Aβ+ CN-C',
        3: 'Aβ+ MCI-NC',
        4: 'Aβ+ MCI-C',
        5: 'Aβ+ AD'}

    # load wscore
    data_wscore_df = pd.read_csv(filewscore, index_col=0, header=0)
    data_demo = pd.read_csv(filedemo, header=0)

    # Basic consistency checks
    w_ids = data_wscore_df.index.astype(str).str.strip()
    d_ids = data_demo['SubjectID'].astype(str).str.strip()

    if not w_ids.equals(d_ids):
        data_wscore_df = data_wscore_df.loc[d_ids].copy()
        print("Reordered W-scores to match the SubjectID order in the demographic table")
    else:
        print("SubjectIDs and their order match exactly; no adjustment is needed")

    subject_ids_all = data_wscore_df.index.to_numpy() 
    data_wscore = data_wscore_df.to_numpy()
    data_demo['SubjectID'] = data_demo['SubjectID'].astype(str)
    group_index = data_demo['Group_Code_new'].to_numpy().flatten()


    custom_colors = [
        '#5D5D5D',  
        '#F28E2B',  
        '#E15759',  
        '#76B7B2', 
        '#59A14F',  
        '#EDC948', 
        '#B07AA1',  
        '#FF9DA7',  
        '#8C564B',  
        '#1E4E79', 
        '#275884',
        '#30628F',
        '#396C9A',
        '#4276A5',
        '#4C80AF',
        '#588AB8',
        '#6594C1',
        '#739ECA',
        '#82A9D3',
        '#91B4DC',  
    ]

    # load connectomes
    euclidean = np.load(os.path.join(Dir_net, 'euclidean.npy'))
    gc = np.load(os.path.join(Dir_net, 'gc.npy'))
    rs = np.load(os.path.join(Dir_net, 'rs.npy'))
    ls = np.load(os.path.join(Dir_net, 'ls.npy'))
    mc = np.load(os.path.join(Dir_net, 'mc.npy'))
    hc = np.load(os.path.join(Dir_net, 'hc.npy'))
    ec = np.load(os.path.join(Dir_net, 'ec.npy'))
    ts = np.load(os.path.join(Dir_net, 'ts.npy'))
    wsc = np.load(os.path.join(Dir_net, 'sc.npy'))
    bsc = (wsc > 0).astype(float)
    np.fill_diagonal(bsc, 0.0)
    
    
    coords = np.genfromtxt(coords_file)[:, -3:]
    distance_matrix = squareform(pdist(coords, metric='euclidean'))

    fc_densities = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    fc_networks = {
        f"FC{int(round(density * 100))}": sparsify_fc_by_density(hc, density)
        for density in fc_densities
    }

    networks = {
        "Euclidean": euclidean, "bSC": bsc, "wSC": wsc, "GC": gc, "RS": rs, "LS": ls,
        "MC": mc, "EC": ec, "TS": ts, "FC": hc,
        **fc_networks
    }

    unique_groups = np.unique(group_index)

    # ==================== Group mean volume analysis ====================
    print("\n=== Starting group mean volume analysis ===")

    group_indices = {}
    group_mean_volumes = {}  # Store the mean volume for each group
    for group in unique_groups:
        group_indices[group] = np.where(group_index == group)[0]
        group_mean_volumes[group] = np.mean(data_wscore[group_indices[group], :], axis=0)
        label = group_labels.get(group, f'Group {group}')
        print(f"Group {group} ({label}): {len(group_indices[group])} subjects")

    group_mean_NNR_df, group_null_distributions_df = run_group_nulls(
        networks, group_mean_volumes, group_labels, distance_matrix, group_level_outpath,
        spin_n=SPIN_N_PERM, smash_n=BRAINSMASH_N_PERM, rewire_n=REWIRED_N_PERM,
        smash_options=BRAINSMASH_OPTIONS,
        n_jobs=N_JOBS, rewire_bins=REWIRED_N_BINS, rewire_swaps=REWIRED_SWAPS_PER_EDGE,LH_ANNOT=LH_ANNOT,RH_ANNOT= RH_ANNOT,
        model_seed=NULL_MODEL_SEEDS
    )

    network_order = {name: order for order, name in enumerate(networks)}
    group_mean_NNR_df['_network_order'] = group_mean_NNR_df['Network'].map(network_order)
    group_mean_NNR_df = group_mean_NNR_df.sort_values(
        ['_network_order', 'Group']
    ).drop(columns='_network_order').reset_index(drop=True)


    group_mean_NNR_output_file = os.path.join(group_level_outpath, 'group_mean_NNR.csv')
    group_null_output_file = os.path.join(group_level_outpath, 'group_null_distributions.csv.gz')
    save_csv_atomic(group_mean_NNR_df, group_mean_NNR_output_file, index=False)
    save_csv_atomic(group_null_distributions_df, group_null_output_file, index=False, compression='gzip')

    for model in ('Spin', 'BrainSMASH', 'Rewired'):
        p_column = f'Pvalue_{model}'
        mask = np.isfinite(group_mean_NNR_df[p_column])
        group_mean_NNR_df[p_column + '_FDR'] = np.nan
        if mask.any():
            group_mean_NNR_df.loc[mask, p_column + '_FDR'] = multi.multipletests(
                group_mean_NNR_df.loc[mask, p_column], method='fdr_bh')[1]

    save_csv_atomic(group_mean_NNR_df, group_mean_NNR_output_file, index=False)

    plot_empirical_spin_rewired_comparison(
        summary_df=group_mean_NNR_df,null_df=group_null_distributions_df,
        networks=networks, group_labels=group_labels, output_path=group_level_outpath,
    )

    # ==================== Individual-level analysis ====================
    print("\n=== Starting individual-level analysis of the original data ===")
    output_file = os.path.join(subject_level_outpath, 'nnr_results_each_subject.csv')
    required_result_cols = [
        column for net_name in networks
        for column in (f'{net_name}_NNR', f'{net_name}_pvalue')
    ]
    existing_result_is_complete = False
    existing_subject_results = pd.DataFrame()
    if os.path.exists(output_file):
        existing_subject_results = pd.read_csv(output_file)
        existing_columns = existing_subject_results.columns
        existing_result_is_complete = all(
            col in existing_columns for col in required_result_cols
        ) and set(existing_subject_results['SubjectID'].astype(str)) == set(
            np.asarray(subject_ids_all).astype(str)
        )

    if not existing_result_is_complete:
        base_columns = ['SubjectID', 'group', 'group_label', *required_result_cols]
        existing_subject_results = existing_subject_results[
            [col for col in base_columns if col in existing_subject_results.columns]
        ]
        all_group_results = []
        completed_groups = set()
        if not existing_subject_results.empty and all(
            col in existing_subject_results.columns for col in required_result_cols
        ):
            for completed_group, rows in existing_subject_results.groupby('group'):
                expected_ids = subject_ids_all[group_indices[completed_group]]
                if set(rows['SubjectID'].astype(str)) == set(np.asarray(expected_ids).astype(str)):
                    completed_groups.add(completed_group)
                    all_group_results.append(rows)

        for group in unique_groups:
            if group in completed_groups:
                print(f"Skipping completed group {group} from checkpoint.")
                continue
            group_label = group_labels.get(group, f'Group {group}')
            print(f"Processing group {group} ({group_label})")

            group_subject_indices = group_indices[group]
            group_data = data_wscore[group_subject_indices, :]
            group_subject_ids = subject_ids_all[group_subject_indices]

            group_results = Parallel(n_jobs=N_JOBS)(
                delayed(calculate_all_network_NNR_for_subject)(
                    group_data[sub_idx, :], networks,
                )
                for sub_idx in range(group_data.shape[0])
            )
            subject_result_df = pd.DataFrame(group_results)
            subject_result_df.insert(0, 'group_label', group_label)
            subject_result_df.insert(0, 'group', group)
            subject_result_df.insert(0, 'SubjectID', group_subject_ids)
            group_results = [subject_result_df]

            group_final_result = pd.DataFrame({
                'SubjectID': group_subject_ids, 'group': group, 'group_label': group_label  # Add group labels
            })

            for result_df in group_results:
                group_final_result = pd.merge(group_final_result, result_df,
                                              on=['SubjectID', 'group', 'group_label'], how='left')

            all_group_results.append(group_final_result)
            intermediate_result = pd.concat(all_group_results, ignore_index=True)
            save_csv_atomic(intermediate_result, output_file, index=False)
            print(f"Checkpoint saved after group {group}: {output_file}")

    
        final_result_all_groups = pd.concat(all_group_results, ignore_index=True)
        save_csv_atomic(final_result_all_groups, output_file, index=False)
        print(f"\nAll groups results saved to: {output_file}")
    else:
        final_result_all_groups = pd.read_csv(output_file)

    # ==================== Global FDR correction (all groups x all networks) ====================
    if not any('_fdr' in col for col in final_result_all_groups.columns):
        pvalue_cols = [col for col in final_result_all_groups.columns if col.endswith('_pvalue')]

        all_pvals_list = []
        all_indices = [] 

        for col in pvalue_cols:
            pvals = final_result_all_groups[col]
            mask = ~pvals.isna()
            all_pvals_list.extend(pvals[mask].values)
            all_indices.extend([(idx, col) for idx in pvals[mask].index])

        all_pvals_array = np.array(all_pvals_list)

        _, pvals_fdr, _, _ = multipletests(all_pvals_array, alpha=0.05, method='fdr_bh')

        for (row_idx, col), fdr_val in zip(all_indices, pvals_fdr):
            final_result_all_groups.loc[row_idx, col + '_fdr'] = fdr_val

        save_csv_atomic(final_result_all_groups, output_file, index=False)


    create_all_raincloud_plots(
        final_result_all_groups=final_result_all_groups, networks=networks,
        output_path=subject_level_outpath,
        group_labels=group_labels,
        y_label='Node-neighbor relationship',
        y_bottom=-0.4, y_top=0.9, custom_colors=custom_colors
    )

    print("\nAll calculations and plotting completed!")

if __name__ == "__main__":
    main()
