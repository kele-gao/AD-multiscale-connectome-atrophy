'''
Plotting functions module
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

def get_star(p):
    if pd.isna(p):
        return ''
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

def fisher_z_transform(correlations):
    corr_clipped = np.clip(correlations, -0.9999, 0.9999)

    z_scores = np.arctanh(corr_clipped)

    return z_scores


def compare_with_euclidean(group_data, networks, alpha=0.05):
    network_names = list(networks.keys())
    if 'Euclidean' in network_names:
        network_names.remove('Euclidean')

    comparison_results = []

    for network in network_names:
        paired = group_data[['Euclidean_NNR', f'{network}_NNR']].to_numpy(dtype=float)
        paired = paired[np.isfinite(paired).all(axis=1)]
        euclidean_corrs = paired[:, 0]
        network_corrs = paired[:, 1]

        if len(paired) < 2:
            comparison_results.append({
                'Network': network,
                't_statistic': np.nan,'p_value': np.nan,
                'p_value_corrected': np.nan,
                'significant': False
            })
            continue

        euclidean_z = fisher_z_transform(euclidean_corrs)
        network_z = fisher_z_transform(network_corrs)

        t_stat, p_value = stats.ttest_rel(euclidean_z, network_z)

        comparison_results.append({
            'Network': network, 't_statistic': t_stat, 'p_value': p_value, 'significant': False
        })

    results_df = pd.DataFrame(comparison_results)

    return results_df

def create_raincloud_plot(group_data, group_label, comparison_results,
                          networks, output_path, y_label, y_bottom, y_top, custom_colors=None,
                          figsize=(9, 8), dpi=1000, text=None):

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']

    network_names = list(networks.keys())

    fig, ax = plt.subplots(figsize=figsize)

    positions = np.arange(len(network_names))
    box_width = 0.15
    violin_width = 0.5


    for i, network_name in enumerate(network_names):
        NNR_col = f'{network_name}_NNR'
        pval_col = f'{network_name}_pvalue'  

        valid_data = group_data[[NNR_col, pval_col]].dropna()

        if len(valid_data) == 0:
            continue

        NNR = valid_data[NNR_col].values
        color_idx = i % len(custom_colors)

        ax.scatter(
            np.random.normal(positions[i] - box_width, 0.04, len(NNR)),
            NNR,color=custom_colors[color_idx],
            alpha=0.8, s=40, edgecolor='white',linewidth=0.5, zorder=2,
        )

        box_pos = positions[i] - box_width / 100

        box = ax.boxplot(
            NNR,
            positions=[box_pos],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            notch=True,
            medianprops={'color': 'black', 'linewidth': 2, 'zorder': 4},
            boxprops={'facecolor': custom_colors[color_idx],
                      'edgecolor': custom_colors[color_idx],
                      'linewidth': 2, 'zorder': 3},
            whiskerprops={'color': custom_colors[color_idx],
                          'linewidth': 2, 'zorder': 3},
            capprops={'color': custom_colors[color_idx],
                      'linewidth': 2, 'zorder': 3}
        )

        violin_pos = positions[i] + box_width / 50

        violin = ax.violinplot(
            NNR,positions=[violin_pos],
            widths=violin_width,
            showmeans=False, showmedians=False, showextrema=False
        )

    
        for pc in violin['bodies']:
            pc.set_facecolor(custom_colors[color_idx])
            pc.set_edgecolor(custom_colors[color_idx])
            pc.set_alpha(0.4)
            pc.set_zorder(2)

            vertices = pc.get_paths()[0].vertices
            vertices[:, 0] = np.where(
                vertices[:, 0] > violin_pos,
                vertices[:, 0],
                violin_pos
            )

        if comparison_results is not None:
            if network_name != 'Euclidean':

                network_comparison = comparison_results[comparison_results['Network'] == network_name]

                if not network_comparison.empty and network_comparison['significant'].iloc[0]:

                    y_position = np.max(NNR) + 0.03
                    ax.text(positions[i], y_position, '*',
                            ha='center', va='bottom', fontsize=30,
                            fontweight='bold', color='black')


    ax.set_xticks(positions)
    ax.set_xticklabels(network_names, fontsize=25)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=45, ha='right')
    ax.set_ylabel(y_label, fontsize=25)

  
    y_ticks = np.arange(-0.4, 0.8, 0.2) 
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks], fontsize=25)

    ax.set_ylim(bottom=y_bottom, top=y_top)

    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color('black')

    plt.title(f'{group_label}',pad=7, fontsize=25, fontweight='bold')

    if text is not None:
        ax.text(
            0.6, 0.98, text,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=13, fontstyle='italic',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black', alpha=0.85)
        )


    plt.tight_layout()

    save_name = group_label.replace(" ", "_").replace("+", "pos").replace("/", "_")
    save_path_png = os.path.join(output_path, f'raincloud_NNR_{save_name}_with_comparison.png')
    plt.savefig(save_path_png, dpi=dpi, bbox_inches='tight', transparent=True)

    plt.close()
    return save_path_png


def create_all_raincloud_plots(final_result_all_groups, networks, output_path,
                               group_labels, y_label, y_bottom, y_top, custom_colors=None):

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']

    unique_groups = final_result_all_groups['group'].unique()

    all_comparison_results = []

    for group in unique_groups:

        group_data = final_result_all_groups[final_result_all_groups['group'] == group] 

        comparison_results = compare_with_euclidean(group_data, networks)

        comparison_results['group'] = group 

        all_comparison_results.append(comparison_results)  

    
    all_comparison_df = pd.concat(all_comparison_results, ignore_index=True)

    pvals = all_comparison_df['p_value'].values
    _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    all_comparison_df['p_value_fdr'] = pvals_corrected
    all_comparison_df['significant'] = (pvals_corrected < 0.05) & (all_comparison_df['t_statistic'] < 0)
    
    comparison_file = os.path.join(output_path, f'comparison_with_euclidean.csv')
    all_comparison_df.to_csv(comparison_file, index=False)

   
    for group in unique_groups:
        group_label = group_labels.get(group, f'Group {group}')
        print(f"\nCreating a raincloud plot for group {group} ({group_label})...")

        orig_group_data = final_result_all_groups[final_result_all_groups['group'] == group]
        comparison_results = all_comparison_df[all_comparison_df['group'] == group]

        create_raincloud_plot(
            group_data=orig_group_data, group_label=group_label,
            comparison_results=comparison_results, networks=networks,
            output_path=output_path, custom_colors=custom_colors,
            y_label=y_label, y_bottom=y_bottom, y_top=y_top,
            text=None,figsize=(14, 8)
        )

    print("\nAll raincloud plots have been created!")
