import os.path
from itertools import combinations
from patsy import build_design_matrices, dmatrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, ttest_ind
from scipy import stats
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

def sparsify_fc_by_density(fc_matrix, density):
    """Keep the strongest proportion of positive FC edges symmetrically."""

    # Count each undirected edge once and sparsify the existing positive edges.
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

def run_ancova_for_network(df, network_name):

    NNR_col = f'{network_name}_NNR'

    needed_cols = ['group', 'Age', 'Sex_Code', 'PTEDUCAT', 'PTHAND', NNR_col]
    analysis_df = df[needed_cols].copy()
    analysis_df = analysis_df.dropna()

    if analysis_df.shape[0] < 10:
        return None, np.nan, np.nan, analysis_df

    # Fisher z transformation
    eps = 1e-8
    analysis_df['FisherZ'] = np.arctanh(np.clip(analysis_df[NNR_col], -1 + eps, 1 - eps))

    analysis_df['group'] = analysis_df['group'].astype('category')
    analysis_df['Sex_Code'] = analysis_df['Sex_Code'].astype('category')
    analysis_df['PTHAND'] = analysis_df['PTHAND'].astype('category')

    formula = 'FisherZ ~ C(group) + C(Sex_Code) + Age + PTEDUCAT + C(PTHAND)'

    model = smf.ols(formula, data=analysis_df).fit()
    anova_res = anova_lm(model, typ=2)

    group_f = anova_res.loc['C(group)', 'F'] if 'C(group)' in anova_res.index else np.nan
    group_p = anova_res.loc['C(group)', 'PR(>F)'] if 'C(group)' in anova_res.index else np.nan

    return model, group_f, group_p, analysis_df


def perform_posthoc_tests_covariate_adjusted(analysis_df):
    significant_comparisons = []

    model = smf.ols(
        'FisherZ ~ C(group) + Age + PTEDUCAT + C(Sex_Code) + C(PTHAND)',
        data=analysis_df
    ).fit()

    def test_scalar(value):
        values = np.asarray(value).reshape(-1)
        return float(values[0])

    all_p_values = []
    all_comparisons = []

    groups = sorted(analysis_df['group'].dropna().unique())
    design_info = dmatrix(
        'C(group) + Age + PTEDUCAT + C(Sex_Code) + C(PTHAND)',
        analysis_df, return_type='dataframe'
    ).design_info
    for group1, group2 in combinations(groups, 2):

        reference = analysis_df.iloc[[0, 0]].copy().reset_index(drop=True)
        reference['group'] = pd.Categorical(
            [group1, group2], categories=analysis_df['group'].cat.categories
        )
        design = np.asarray(build_design_matrices([design_info], reference)[0])
        test = model.t_test(design[1] - design[0])
        p_value = test_scalar(test.pvalue)
        if np.isfinite(p_value):
            all_p_values.append(p_value)
            all_comparisons.append((group1, group2, test_scalar(test.effect)))

    if len(all_p_values) == 0:
        return significant_comparisons

    rejected, corrected_pvals, _, _ = multipletests(all_p_values, alpha=0.05, method='fdr_bh')

    for idx, (group1, group2, effect) in enumerate(all_comparisons):
        if rejected[idx]:
            significant_comparisons.append({
                'group1': group1, 'group2': group2,
                'p_value': corrected_pvals[idx], 'original_p': all_p_values[idx], 'effect': effect
            })

    return significant_comparisons


Path = r''
Dir_in = os.path.join(Path, 'results', '3_NNR', '1_cross_section', '2_subject_level_analysis')
file_NNR = os.path.join(Dir_in,'nnr_results_each_subject.csv')
outpath = os.path.join(Dir_in, 'plot_by_group')

filewscore = os.path.join(Path, 'results', '2_Wscore', 'W_scores_threshold0.csv')
filedemo = os.path.join(Path, 'results', '2_Wscore','W_scores_demo.csv')
Dir_net = os.path.join(Path, 'resources', 'connect')
outpath = os.path.join(Dir_in, 'plot_by_group')

os.makedirs(outpath, exist_ok=True)

group_labels = {
1: 'Aβ+ CN-NC',
2: 'Aβ+ CN-C',
3: 'Aβ+ MCI-NC',
4: 'Aβ+ MCI-C',
5: 'Aβ+ AD'}

final_result_all_groups = pd.read_csv(file_NNR)

# load wscore
data_wscore = pd.read_csv(filewscore, index_col=0, header=0).to_numpy()
numSub = data_wscore.shape[0]
numRegions = data_wscore.shape[1]

# load demo
data_demo = pd.read_csv(filedemo, header=0)
data_demo['Subject ID'] = data_demo['SubjectID'].astype(str)
group_index = data_demo['Group_Code_new'].to_numpy().flatten()

unique_groups = np.unique(group_index)

# load connectomes
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
euclidean = np.load(os.path.join(Dir_net, 'euclidean.npy'))

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


network_colors = [
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

covariate_cols = ['SubjectID', 'Age', 'Sex_Code', 'PTEDUCAT', 'PTHAND']
data_demo_sub = data_demo[covariate_cols].drop_duplicates(subset='SubjectID')

final_result_all_groups = final_result_all_groups.merge(
    data_demo_sub, on='SubjectID', how='left')

ancova_pvalues = {}
ancova_fvalues = {}

network_models = {}
network_analysis_dfs = {}

network_names = list(networks.keys())
for network_name in network_names:
    model, group_f, group_p, analysis_df = run_ancova_for_network(final_result_all_groups, network_name)

    ancova_pvalues[network_name] = group_p
    ancova_fvalues[network_name] = group_f

    network_models[network_name] = model
    network_analysis_dfs[network_name] = analysis_df

# ==================== FDR ====================
network_list = list(ancova_pvalues.keys())
pvals = np.array([ancova_pvalues[n] for n in network_list], dtype=float)

valid_mask = ~np.isnan(pvals)
pvals_fdr = np.full_like(pvals, np.nan, dtype=float)

if np.sum(valid_mask) > 0:
    rejected, corrected_pvals, _, _ = multipletests(
        pvals[valid_mask], alpha=0.05, method='fdr_bh'
    )
    pvals_fdr[valid_mask] = corrected_pvals

ancova_pvalues_fdr = dict(zip(network_list, pvals_fdr))

# ==================== Save statistical results ====================
stats_rows = []
for network_name in network_names:
    stats_rows.append({
        'Network': network_name,
        'Group_F': ancova_fvalues.get(network_name, np.nan),
        'Group_P': ancova_pvalues.get(network_name, np.nan),
        'Group_P_FDR': ancova_pvalues_fdr.get(network_name, np.nan),
    })

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(os.path.join(outpath, 'ancova_results.csv'), index=False)

# =================== Plotting ====================
print("\n=== Starting plotting ===")

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

for network_name in network_names:
    print(f"\nCreating a raincloud plot for network {network_name}...")

    plot_data = {}
    pvalue_data = {}

    for group in unique_groups:
        group_data = final_result_all_groups[final_result_all_groups['group'] == group]
        NNR_col = f'{network_name}_NNR'
        pval_col = f'{network_name}_pvalue_fdr'

        valid_cols = [NNR_col]
        if pval_col in group_data.columns:
            valid_cols.append(pval_col)

        valid_data = group_data[valid_cols].dropna(subset=[NNR_col])

        plot_data[group] = valid_data[NNR_col]

        if pval_col in valid_data.columns:
            pvalue_data[group] = valid_data[pval_col]
        else:
            pvalue_data[group] = pd.Series(np.ones(len(valid_data)), index=valid_data.index)

    p_anova_fdr = ancova_pvalues_fdr[network_name]
    f_stat = ancova_fvalues[network_name]

    significant_comparisons = []
    if not np.isnan(p_anova_fdr) and p_anova_fdr < 0.05:
        print(f"Significant ANCOVA (FDR p = {p_anova_fdr:.4f}); performing covariate-adjusted post hoc tests...")
        significant_comparisons = perform_posthoc_tests_covariate_adjusted(
            network_analysis_dfs[network_name]
        )

    fig, ax = plt.subplots(figsize=(max(6, len(unique_groups) * 1.5), 11))

    positions = np.arange(len(unique_groups))
    box_width = 0.15
    violin_width = 0.5

    for i, group in enumerate(unique_groups):
        correlations = plot_data[group].values
        p_values = pvalue_data[group].values

        if len(correlations) == 0:
            continue

        violin_color = network_colors[network_names.index(network_name)]

        ax.scatter(
            np.random.normal(positions[i] - box_width, 0.04, len(correlations)),
            correlations,
            color=violin_color, alpha=0.8, s=40, edgecolor='white',
            linewidth=0.5, zorder=2
        )

        box_pos = positions[i] - box_width / 100

        ax.boxplot(
            correlations,
            positions=[box_pos], widths=box_width,
            patch_artist=True, showfliers=False, notch=True,
            medianprops={'color': 'black', 'linewidth': 2, 'zorder': 4},
            boxprops={'facecolor': violin_color, 'edgecolor': violin_color, 'linewidth': 2, 'zorder': 3},
            whiskerprops={'color': violin_color, 'linewidth': 2, 'zorder': 3},
            capprops={'color': violin_color, 'linewidth': 2, 'zorder': 3}
        )

        violin_pos = positions[i] + box_width / 50

        violin = ax.violinplot(
            correlations,
            positions=[violin_pos],
            widths=violin_width,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )

        for pc in violin['bodies']:
            pc.set_facecolor(violin_color)
            pc.set_edgecolor(violin_color)
            pc.set_alpha(0.4)
            pc.set_zorder(2)

            vertices = pc.get_paths()[0].vertices
            vertices[:, 0] = np.where(
                vertices[:, 0] > violin_pos,
                vertices[:, 0],
                violin_pos
            )

    annotation_top = 0.9
    if significant_comparisons:
        group_positions = dict(zip(unique_groups, positions))
        finite_arrays = [np.asarray(v)[np.isfinite(np.asarray(v))]
                         for v in plot_data.values()]
        all_values = np.concatenate([v for v in finite_arrays if len(v)])
        data_span = max(float(np.ptp(all_values)), 0.2)
        step = max(0.065, data_span * 0.09)
        base = float(np.max(all_values)) + step * 0.6
        blocks_by_right_group = {}
        for comparison in significant_comparisons:
            anchor = max(
                (comparison['group1'], comparison['group2']),
                key=lambda group: group_positions[group]
            )
            blocks_by_right_group.setdefault(anchor, []).append(comparison)

        comparison_blocks = []
        for anchor in sorted(blocks_by_right_group, key=lambda group: group_positions[group]):
            block = sorted(
                blocks_by_right_group[anchor],
                key=lambda c: abs(group_positions[c['group2']] - group_positions[c['group1']])
            )
            comparison_blocks.append((anchor, block))

        comparisons = []
        levels = []
        next_level = 0.0
        for anchor, block in comparison_blocks:
            for comparison in block:
                comparisons.append(comparison)
                levels.append(next_level)
                next_level += 1.0
            next_level += 0.5

        for level, comparison in zip(levels, comparisons):
            x1, x2 = sorted([group_positions[comparison['group1']],
                             group_positions[comparison['group2']]])
            y = base + level * step
            bracket_height = step * 0.25
            ax.plot([x1, x1, x2, x2],
                    [y, y + bracket_height, y + bracket_height, y],
                    color='black', linewidth=1.5, zorder=6)
            p_fdr = comparison['p_value']
            if p_fdr<0.05:
                stars = '*'

            ax.annotate(
                stars,
                xy=((x1 + x2) / 2, y + bracket_height), xytext=(0, -5),              
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=27, fontweight='bold', zorder=7
            )
        annotation_top = max(0.9, base + next_level * step + step * 0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels([group_labels[group] for group in unique_groups], fontsize=28)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=45, ha='right')

    ax.set_ylabel('Node-neighbor relationship', fontsize=28)

    ax.autoscale(enable=False)
    y_ticks = np.arange(-0.8, 1, 0.2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks], fontsize=28)
    ax.set_ylim(bottom=-0.4, top=annotation_top)

    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color('black')

    plt.title(f'{network_name}', pad=8, fontsize=28, fontweight='bold')

    text_x = 0.35
    if network_name in ['FC', 'EC', 'TS', 'GC', 'Euclidean', 'FC60','FC70','FC80','FC90']:
        text_y = 0.9
    else:
        text_y = 0.05

    if np.isnan(p_anova_fdr):
        text_str = r'$\mathit{F}=\mathrm{NaN}$' + '\n' + r'$\mathit{p}_{\mathrm{FDR}}=\mathrm{NaN}$'
    elif p_anova_fdr < 0.001:
        text_str = (
            r'$\mathit{F}=%.1f$ ' % f_stat +
            r'$\mathit{p}_{\mathrm{FDR}}<0.001$'
        )
    else:
        text_str = (
            r'$\mathit{F}=%.1f$ ' % f_stat +
            r'$\mathit{p}_{\mathrm{FDR}}=%.3f$' % p_anova_fdr
        )

    ax.text(
        text_x, text_y, text_str,
        transform=ax.transAxes, fontsize=26, verticalalignment='bottom',
        bbox=dict(boxstyle='square', facecolor='lightgray', alpha=0.7, edgecolor='none'),
        zorder=4
    )

    plt.tight_layout()

    save_path = os.path.join(outpath, f'raincloud_NNR_{network_name}.png')
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close()

print("\nAll calculations and plotting completed!")