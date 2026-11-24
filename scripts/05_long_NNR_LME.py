from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import os.path
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway, chi2_contingency
import statsmodels.stats.multitest as multi
import statsmodels.formula.api as smf
from patsy import build_design_matrices, dmatrix
from itertools import combinations
from node_neighbor_correlation import calculate_neighbor_correlation_for_subject

warnings.filterwarnings('ignore')


plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

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


def calculate_individual_slopes(results_df):
    print("\n计算个体斜率...")

    all_slopes = []

    # 获取所有网络
    networks = results_df['Network'].unique()

    # 遍历每个被试
    for subject_id in tqdm(results_df['SubjectID'].unique(), desc="处理被试"):
        # 获取该被试的所有数据
        subject_data = results_df[results_df['SubjectID'] == subject_id]

        # 获取组别（假设组别不变）
        group_code = subject_data['Group_Code'].iloc[0] if not subject_data.empty else None

        # 对每个网络计算斜率
        for network in networks:
            # 获取该被试在该网络上的数据
            network_data = subject_data[subject_data['Network'] == network].copy()
            network_data = network_data.dropna(subset=['r_value', 'scan_interval_years'])

            # 至少需要2个时间点才能计算斜率
            if len(network_data) >= 2:
                x = network_data['scan_interval_years']
                y = network_data['r_value']

                # 计算线性回归斜率
                slope, intercept, r_value, p_value, _ = stats.linregress(x, y)

                all_slopes.append({
                    'SubjectID': subject_id, 'Group_Code': group_code, 'Network': network,
                    'slope': slope, 'intercept': intercept,
                    'r_value': r_value, 'p_value': p_value,
                    'n_points': len(network_data)
                })

    slopes_df = pd.DataFrame(all_slopes)

    return slopes_df


def _apply_fdr(results_df, p_column='p_value'):
    results_df = results_df.copy()
    valid = results_df[p_column].notna()
    results_df['Pvalue_FDR'] = np.nan
    results_df['significant_FDR'] = False
    if valid.any():
        reject, corrected, _, _ = multi.multipletests(
            results_df.loc[valid, p_column].to_numpy(),
            alpha=0.05, method='fdr_bh'
        )
        results_df.loc[valid, 'Pvalue_FDR'] = corrected
        results_df.loc[valid, 'significant_FDR'] = reject
    return results_df


def _estimate_linear_contrast(fit, contrast):
    fe_names = list(fit.fe_params.index)
    contrast = np.asarray(contrast, dtype=float).reshape(-1)

    beta = float(contrast @ fit.fe_params.to_numpy())
    fe_cov = fit.cov_params().loc[fe_names, fe_names].to_numpy()

    variance = float(contrast @ fe_cov @ contrast)
    se = float(np.sqrt(max(variance, 0.0)))

    z_value = beta / se if se > 0 else np.nan
    p_value = float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
    return {
        'beta': beta, 'se': se,
        'ci_lower': beta - 1.96 * se,
        'ci_upper': beta + 1.96 * se,
        'z_value': z_value, 'p_value': p_value
    }


def _time_slope_contrast(fit, design_info, network, group_code, network_order,
                         group_order, covariate_values):
    """Return the fixed-effect contrast representing one network/group time slope."""
    rows = pd.DataFrame({
        'scan_interval_years': [0.0, 1.0],
        'Network': pd.Categorical(
            [network, network], categories=network_order, ordered=True
        ),
        'Group_Code': pd.Categorical(
            [group_code, group_code], categories=group_order, ordered=True
        ),
        **{name: [value, value] for name, value in covariate_values.items()}
    })
    design = np.asarray(build_design_matrices([design_info], rows)[0])
    return design[1] - design[0]


def _groupwise_time_slope_contrast(fit, design_info, network,
                                   network_order, covariate_values):
    """Fixed-effect contrast for one network's time slope in one group model."""
    rows = pd.DataFrame({
        'scan_interval_years': [0.0, 1.0],
        'Network': pd.Categorical(
            [network, network], categories=network_order, ordered=True
        ),
        **{name: [value, value] for name, value in covariate_values.items()}
    })
    design = np.asarray(build_design_matrices([design_info], rows)[0])
    return design[1] - design[0]


def fit_groupwise_time_network_lme(results_df, group_labels,
                                   network_order, group_order):
    """
    Fit one joint time×network LME within each clinical group:

    Fisher-z(NNR) ~ time * network + baseline covariates
                    + (1 + time | SubjectID)
    """
    print("\n按组别拟合 time × network LME...")
    all_slopes = []
    all_network_contrasts = []
    diagnostics = []
    summaries = []
    reference_network = 'Euclidean'

    for group_code in group_order:
        group_label = group_labels[group_code]
        required = ['NNR_value', 'scan_interval_years', 'Network', 'SubjectID',
                    'BaselineAge', 'Sex_Code', 'PTEDUCAT', 'PTHAND']
        model_data = results_df.loc[results_df['Group_Code_new'] == group_code].dropna(subset=required).copy()
        model_data = model_data.loc[model_data['NNR_value'].between(-1, 1, inclusive='neither')]
        model_data['Fisher_z_NNR'] = np.arctanh( model_data['NNR_value'].clip(-0.999999, 0.999999))
        model_data['Network'] = pd.Categorical(
            model_data['Network'], categories=network_order, ordered=True
        )

        fixed_effect_rhs = (
            "scan_interval_years * "
            f"C(Network, Treatment(reference='{reference_network}')) + "
            "BaselineAge + Sex_Code + PTEDUCAT + PTHAND"
        )
        formula = f'Fisher_z_NNR ~ {fixed_effect_rhs}'
        fixed_design = dmatrix(fixed_effect_rhs, model_data, return_type='dataframe')
        design_info = fixed_design.design_info
        model = smf.mixedlm(
            formula, model_data, groups=model_data['SubjectID'],
            re_formula='1 + scan_interval_years',use_sparse=True
        )

        method_used = 'lbfgs'
        fit = model.fit(reml=False, method=method_used, maxiter=3000, disp=False)

        covariate_values = {
            name: float(model_data[name].mean())
            for name in ['BaselineAge', 'Sex_Code', 'PTEDUCAT', 'PTHAND']
        }

        # 提取每个网络自身的时间斜率
        slope_vectors = {}
        for network in network_order:
            contrast = _groupwise_time_slope_contrast(
                fit, design_info, network, network_order, covariate_values
            )
            slope_vectors[network] = contrast
            estimate = _estimate_linear_contrast(fit, contrast)  # 计算斜率的标准误、置信区间与 P 值
            all_slopes.append({
                'Group_Code': group_code, 'Group_Label': group_label,
                'Network': network, 'beta_time': estimate['beta'],
                **{key: value for key, value in estimate.items() if key != 'beta'}
            })

        # 计算每个网络相对于参考网络的时间斜率差异
        reference_vector = slope_vectors[reference_network]
        for network in network_order:
            if network == reference_network:
                continue
            
            estimate = _estimate_linear_contrast(fit, slope_vectors[network] - reference_vector)
            all_network_contrasts.append({
                'Group_Code': group_code, 'Group_Label': group_label,
                'Network': network,
                'Reference_Network': reference_network,
                'beta_difference': estimate['beta'],
                **{key: value for key, value in estimate.items() if key != 'beta'}
            })

        random_cov = fit.cov_re
        intercept_var = float(random_cov.iloc[0, 0])
        slope_var = float(random_cov.iloc[1, 1])
        intercept_slope_cov = float(random_cov.iloc[0, 1])
        re_corr = (
            intercept_slope_cov / np.sqrt(intercept_var * slope_var)
            if intercept_var > 0 and slope_var > 0 else np.nan
        )
        diagnostics.append({
            'Group_Code': group_code, 'Group_Label': group_label,
            'formula': formula, 'optimizer': method_used,
            'converged': bool(fit.converged),
            'n_observations': int(fit.nobs),
            'n_subjects': int(model_data['SubjectID'].nunique()),
            'n_scans': int(model_data[['SubjectID', 'ScanDate']].drop_duplicates().shape[0]),
            'aic': float(fit.aic), 'bic': float(fit.bic),
            'log_likelihood': float(fit.llf),
            'random_intercept_variance': intercept_var,
            'random_time_slope_variance': slope_var,
            'random_intercept_time_covariance': intercept_slope_cov,
            'random_intercept_time_correlation': re_corr,
            'residual_variance': float(fit.scale)
        })
        summaries.append(f'### {group_label}\n\n{fit.summary().as_text()}')
        print(f'  {group_label}: converged with {method_used}')

    slopes_df = _apply_fdr(pd.DataFrame(all_slopes))
    network_contrasts_df = _apply_fdr(pd.DataFrame(all_network_contrasts))
    return (slopes_df, network_contrasts_df,
            pd.DataFrame(diagnostics), '\n\n'.join(summaries))


def plot_dual_significance_heatmap(slopes_df, network_contrasts_df,
                                   output_file, group_order,
                                   network_order, dpi=600, figsize=(15, 17)):
    """
    Show time slopes. Black *: slope differs from zero after BH-FDR.
    Purple *: slope is greater than Euclidean after BH-FDR.
    """
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'

    values = slopes_df.pivot(
        index='Network', columns='Group_Label', values='beta_time'
    ).loc[network_order, group_order]
    slope_p = slopes_df.pivot(
        index='Network', columns='Group_Label', values='Pvalue_FDR'
    ).loc[network_order, group_order]
    network_p = network_contrasts_df.pivot(
        index='Network', columns='Group_Label', values='Pvalue_FDR'
    ).reindex(index=network_order, columns=group_order)
    network_difference = network_contrasts_df.pivot(
        index='Network', columns='Group_Label', values='beta_difference'
    ).reindex(index=network_order, columns=group_order)

    limit = 0.04
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.heatmap(
        values, cmap='coolwarm', center=0, vmin=-limit, vmax=limit,
        linewidths=0.5, square=True, annot=False, cbar=False,
        ax=ax
    )
    norm = plt.Normalize(vmin=-limit, vmax=limit)
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.3, aspect=15, pad=0.02,
                        ticks=np.linspace(-limit, limit, 5))
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(r'Time slope $\beta$ (Fisher z/year)', fontsize=15)

    comparison_color = '#7B1FA2'
    for i, network in enumerate(values.index):
        for j, group_label in enumerate(values.columns):
            ax.text(
                j + 0.5, i + 0.5, f'{values.loc[network, group_label]:.3f}',
                ha='center', va='center', color='black', fontsize=15
            )
            if slope_p.loc[network, group_label] < 0.05:
                ax.text(
                    j + 0.45, i + 0.45, '*', ha='center', va='bottom',
                    color='black', fontsize=20, fontweight='bold'
                )
            if (pd.notna(network_p.loc[network, group_label])
                    and network_p.loc[network, group_label] < 0.05
                    and network_difference.loc[network, group_label] > 0):
                ax.text(
                    j + 0.6, i + 0.45, '*', ha='center', va='bottom',
                    color=comparison_color, fontsize=20, fontweight='bold'
                )

    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xticklabels(values.columns, fontsize=15, rotation=45, ha='right')
    ax.set_yticklabels(values.index, fontsize=15, rotation=0)
    legend_items = [
        Line2D([0], [0], marker='$*$', color='black', linestyle='None',
               markersize=8, label='Time slope FDR p < 0.05'),
        Line2D([0], [0], marker='$*$', color=comparison_color,
               linestyle='None', markersize=8,
               label='Slope > Euclidean, FDR p < 0.05')
    ]
    

    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Dual-significance heatmap saved to: {output_file}')

    legend_file = os.path.splitext(output_file)[0] + '_legend.png'
    legend_fig = plt.figure(figsize=(4.5, 0.9))
    legend = legend_fig.legend(
        handles=legend_items, loc='center',
        frameon=False, ncol=1, fontsize=9
    )
    legend_fig.canvas.draw()
    legend_bbox = legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted())
    legend_fig.savefig(legend_file, dpi=dpi, bbox_inches=legend_bbox, pad_inches=0.1)
    plt.close(legend_fig)
    print(f'Dual-significance legend saved to: {legend_file}')


def plot_contrast_heatmap(results_df, value_column, column_name, output_file,
                          colorbar_label, column_order=None,
                          network_order=None, figsize=(16, 11), dpi=600):
    """Plot estimates with a star for BH-FDR-adjusted p < 0.05."""
    values = results_df.pivot(
        index='Network', columns=column_name, values=value_column
    )
    pvalues = results_df.pivot(
        index='Network', columns=column_name, values='Pvalue_FDR'
    )
    if network_order is not None:
        present_networks = [n for n in network_order if n in values.index]
        values = values.loc[present_networks]
        pvalues = pvalues.loc[present_networks]
    if column_order is not None:
        values = values[column_order]
        pvalues = pvalues[column_order]
    limit = max(float(np.nanmax(np.abs(values.to_numpy()))), 0.001)
    annotations = values.copy().astype(object)
    for row in values.index:
        for column in values.columns:
            value = values.loc[row, column]
            star = '*' if pvalues.loc[row, column] < 0.05 else ''
            annotations.loc[row, column] = f'{value:.3f}{star}'

    fig, ax = plt.subplots(figsize=figsize)
    heatmap = sns.heatmap(
        values, cmap='coolwarm', center=0, vmin=-limit, vmax=limit,
        linewidths=0.5, annot=annotations, fmt='',
        annot_kws={'fontsize': 9, 'color': 'black'},
        cbar_kws={'label': colorbar_label}, ax=ax
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title('* BH-FDR adjusted p < 0.05', pad=12)
    heatmap.collections[0].colorbar.set_label(colorbar_label)
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'Heatmap saved to: {output_file}')

def main():
    path = r''
    file_long_wscore = os.path.join(path, 'results', '2_Wscore', 'long_data', 'W_scores_threshold0.csv')
    file_baseline_demo = os.path.join(path, 'results', '2_Wscore','W_scores_demo.csv')

    Dir_net = os.path.join(path, 'resources', 'connect')
    Dir_out = os.path.join(path, 'results', '3_NNR', '2_longitudinal')
    
    os.makedirs(Dir_out, exist_ok=True) 

    # 1. 加载数据
    print("\n1. 加载数据...")
    data_wscore = pd.read_csv(file_long_wscore)
    data_baseline_demo = pd.read_csv(file_baseline_demo)

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

    networks = {"Euclidean": euclidean, "bSC": bsc, "wSC": wsc, "GC": gc, "RS": rs, "LS": ls,
        "MC": mc, "EC": ec, "TS": ts, "FC": hc, **fc_networks}

    group_labels = {
    1: 'Aβ+ CN-NC',
    2: 'Aβ+ CN-C',
    3: 'Aβ+ MCI-NC',
    4: 'Aβ+ MCI-C',
    5: 'Aβ+ AD'}
    group_order = [1, 2, 3, 4, 5] # 定义绘图时的组别顺序
    group_name_order = ['Aβ+ CN-NC', 'Aβ+ CN-C', 'Aβ+ MCI-NC', 'Aβ+ MCI-C', 'Aβ+ AD']

    # 2. 数据预处理
    print("\n2. 数据预处理...")
    wscore_id_column = data_wscore.columns[0]
    data_wscore = data_wscore.rename(columns={wscore_id_column: 'ScanKey'})
    brain_columns = data_wscore.columns[1:].tolist()
    parts = data_wscore['ScanKey'].str.split('_', n=3, expand=True)
    data_wscore['SubjectID'] = parts[0] + '_' + parts[1] + '_' + parts[2]
    data_wscore['ScanDate'] = pd.to_datetime(parts[3], format='%Y-%m-%d')

    data_baseline_demo['StudyDate'] = pd.to_datetime(data_baseline_demo['Study Date']).dt.normalize()


    # 4. 计算节点-邻居相关
    print("\n4. 计算节点-邻居相关...")
    brain_data = data_wscore[brain_columns].to_numpy(dtype=float)
    all_results = []
  
    for scan_idx in tqdm(range(len(data_wscore)), desc="Processing scans"):
        scan = data_wscore.iloc[scan_idx]
        w_score_vector = brain_data[scan_idx]

        for net_name, net_matrix in networks.items():

            NNR_value, p_value = calculate_neighbor_correlation_for_subject(w_score_vector, net_matrix)

            all_results.append({
            'SubjectID': scan['SubjectID'],'ScanDate': scan['ScanDate'],
            'Network': net_name,'NNR_value': NNR_value, 'p_value': p_value
            })

    results_df = pd.DataFrame(all_results)
    

    baseline_info = data_baseline_demo[['SubjectID', 'Study Date', 'Group_Code_new','Age', 'Sex_Code', 
                                        'PTEDUCAT', 'PTHAND']].copy()
    results_df = results_df.merge(
        baseline_info,
        on='SubjectID', how='left',
        validate='many_to_one', indicator=True
    )

    results_df = results_df.sort_values(
        ['SubjectID', 'ScanDate'], kind='mergesort'
    ).reset_index(drop=True)

    # 计算scan_interval，以首次扫描作为分析基线。
    results_df['scan_interval'] = (
        results_df['ScanDate']
        - results_df.groupby('SubjectID')['ScanDate'].transform('min')
    ).dt.days.astype(float)
    results_df['scan_interval_years'] = results_df['scan_interval']/ 365.0

    results_df= results_df.rename(columns={
        'Study Date': 'BaselineDate',
        'Age': 'BaselineAge'
    })
    results_df.to_csv(os.path.join(Dir_out, 'Long_NNR_results.csv'), index=False)

    # 5. 计算个体斜率和组水平趋势
    print("\n5. 计算随时间变化...")

    network_order = list(networks.keys())
    (group_trends_df, network_contrasts_df,
     model_diagnostics_df, model_summaries) = fit_groupwise_time_network_lme(
        results_df, group_labels, network_order, group_order
    )

    group_trends_file = os.path.join(Dir_out, 'groupwise_LME_network_time_slopes.csv')
    network_contrasts_file = os.path.join(Dir_out, 'groupwise_LME_network_vs_Euclidean_contrasts.csv')

    group_trends_df.to_csv(group_trends_file, index=False)
    network_contrasts_df.to_csv(network_contrasts_file, index=False)
    model_diagnostics_df.to_csv(os.path.join(Dir_out, 'groupwise_LME_model_diagnostics.csv'), index=False)
    with open(os.path.join(Dir_out, 'groupwise_LME_model_summaries.txt'), 'w',
              encoding='utf-8') as summary_file:
        summary_file.write(model_summaries)

    # 6. 绘图
    print("\n6. 创建双色星号时间效应热图...")
    plot_dual_significance_heatmap(
        group_trends_df, network_contrasts_df,
        output_file=os.path.join(Dir_out, 'groupwise_LME_time_slopes_dual_significance.png'),
        group_order=group_name_order, network_order=network_order, dpi=600
    )

    print(f"分析完成")


if __name__ == "__main__":
    main()
