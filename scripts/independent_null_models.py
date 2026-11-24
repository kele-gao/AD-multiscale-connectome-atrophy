"""Independent checkpoints for group node-neighbor nulls.

Spin and BrainSMASH keep Bx fixed and correlate it with surrogate node maps.
Rewired keeps x fixed and recomputes B_random x. All tests are right-tailed
Spearman tests with +1 correction. Bx = sum(positive edge * x) / degree.
"""
import os
from pathlib import Path
from functools import lru_cache, partial
from neuromaps import nulls
from neuromaps.images import annot_to_gifti
from brainsmash.mapgen.base import Base
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from netneurotools.networks import match_length_degree_distribution
from node_neighbor_correlation import (
    calculate_neighborhood_deviation_for_subject,
    calculate_neighbor_correlation_for_subject,
)

def node_neighbor_relationship(x, network, maps):
    
    neighbor_value = calculate_neighborhood_deviation_for_subject(x, network)
    correlations = np.full(maps.shape[1], np.nan)


    for j in range(maps.shape[1]):
        surrogate = maps[:, j]
        valid = np.isfinite(surrogate) & np.isfinite(neighbor_value)

        if np.count_nonzero(valid) < 2:
            continue

        correlations[j],_ = spearmanr(surrogate[valid], neighbor_value[valid])

    return correlations



def cached(directory, name, producer):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filename = directory / f'{name}.npz'

    if filename.exists():
        with np.load(filename, allow_pickle=False) as data:
            print(f'[checkpoint hit] {directory.name}/{name}', flush=True)
            return {k: data[k] for k in data.files}
    
    result = producer()
    temporary = filename.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **result)
    os.replace(temporary, filename)
    print(f'[checkpoint saved] {directory.name}/{name}', flush=True)
    return result

def right_tail(observed, nulls):
    valid = np.asarray(nulls)[np.isfinite(nulls)]
    if not np.isfinite(observed) or not valid.size:
        return np.nan
    return (1 + np.count_nonzero(valid >= observed)) / (1 + len(valid))


def produce_brainsmash_maps(x, distances, n_perm, seed, options):
    
    options = dict(options)
    if 'deltas' in options:
        options['deltas'] = np.asarray(options['deltas'], dtype=float)
    generator = Base(x=x, D=distances, seed=seed, **options)
    maps = np.asarray(generator(n=n_perm)).reshape(n_perm, len(x)).T

    return {'maps': maps}

@lru_cache(maxsize=4)
def _get_schaefer400_spin_indices(n_perm, random_state, LH_ANNOT, RH_ANNOT):
    parcellation = annot_to_gifti((str(LH_ANNOT), str(RH_ANNOT)))

    spin_indices = nulls.alexander_bloch(
        data=None, atlas="fsaverage", density="10k",
        parcellation=parcellation, n_perm=n_perm, seed=random_state
    )

    spin_indices = np.asarray(spin_indices, dtype=int)

    return spin_indices


def produce_spin_maps(x, n_perm, seed, n_jobs,LH_ANNOT,RH_ANNOT):
    """Generate spin indices in independent batches using multiple processes."""

    n_jobs = min(n_jobs, n_perm)
    batch_sizes = np.full(n_jobs, n_perm // n_jobs, dtype=int)
    batch_sizes[:n_perm % n_jobs] += 1 # 把除不尽的剩余置换次数，分配给前几个并行任务，保证总次数等于 n_perm
    batches = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_get_schaefer400_spin_indices)(int(size), seed + batch, LH_ANNOT, RH_ANNOT)
        for batch, size in enumerate(batch_sizes)
    )
    spin_indices = np.concatenate(batches, axis=1)
    return {'maps': x[spin_indices]}


def produce_rewired(name, network, distances, groups, rewire_n, n_jobs,
                    rewire_bins, rewire_swaps, seed=42):
    """Generate rewired nulls using explicit inputs"""

    adjacency = np.asarray(network, dtype=float).copy()
    distances = np.asarray(distances, dtype=float)
    adjacency[adjacency <= 0] = 0.0
    np.fill_diagonal(adjacency, 0.0)
    n_edges = np.count_nonzero(np.triu(adjacency > 0, k=1))
    target_swaps = max(1, int(round(rewire_swaps * n_edges)))

    group_items = list(groups.items())
    result = np.full((len(group_items), rewire_n), np.nan)
    swaps = np.full(rewire_n, np.nan)
    network_seed = sum((i+1)*ord(ch) for i, ch in enumerate(name))

    # 依次生成重连网络，同时每次都计算node-neighbor relashionship
    outputs = Parallel(n_jobs=n_jobs, return_as='generator_unordered')(
        delayed(calculate_one_rewired_permutation)(p, adjacency, distances, groups,
            rewire_bins, target_swaps, seed+network_seed) for p in range(rewire_n))
    
    for count, (p, correlations, fraction) in enumerate(outputs, 1):
        for row, (g, _) in enumerate(group_items):
            result[row, p] = correlations[g]
        swaps[p] = fraction
        if count == 1 or count % 100 == 0 or count == rewire_n:
            print(f'[{name}] Rewired {count}/{rewire_n}', flush=True)
    return {'nulls': result, 'swaps': swaps}


def calculate_one_rewired_permutation(
    permutation, net_matrix, distance_matrix, group_mean_volumes,
    n_bins, target_swaps, base_seed,
):
    """使用预处理矩阵和预先计算的target_swaps生成重连网络并计算各组相关。"""

    _, rewired_matrix, successful_swaps = match_length_degree_distribution(
        W=net_matrix, D=distance_matrix, nbins=n_bins, nswap=target_swaps,
        replacement=True, weighted=True, seed=base_seed + permutation,
    )
    
    correlations = {
        group: calculate_neighbor_correlation_for_subject(
            mean_volume, rewired_matrix
        )[0]
        for group, mean_volume in group_mean_volumes.items()
    }
    swap_fraction = successful_swaps / target_swaps if target_swaps > 0 else np.nan
    return permutation, correlations, swap_fraction


def run_group_nulls(networks, groups, labels, distances, output,
                    spin_n, smash_n, rewire_n, smash_options,
                    n_jobs, rewire_bins, rewire_swaps, LH_ANNOT,RH_ANNOT,
                    model_seed=None):
    """Cache each model separately and reuse a checkpoint whenever it exists."""
    distances = np.asarray(distances, dtype=float)
    
    root = Path(output) / 'null_models'
    for model in ('Spin', 'BrainSMASH', 'Rewired'):
        (root / model).mkdir(parents=True, exist_ok=True)

    group_items = [(g, np.asarray(x, dtype=float).squeeze()) for g, x in groups.items()]

    # Generate spin and smash node map nulls before expensive network rewiring.
    maps_by_group = {}
    for g, x in group_items:
        # spin test
        spun = cached(root/'Spin', f'maps_group{g}',
                      lambda x=x: produce_spin_maps(
                          x, spin_n, model_seed, n_jobs,LH_ANNOT,RH_ANNOT,
                      ))['maps']
        # smash
        smashed = cached(root/'BrainSMASH', f'maps_group{g}',
                         lambda x=x: produce_brainsmash_maps(x, distances, smash_n, model_seed, smash_options))['maps']
        maps_by_group[g] = {'Spin': spun, 'BrainSMASH': smashed}

    summaries, null_tables = [], []
    for name, network in networks.items():
        
        # calculate smash and spin null node_neighbor_relationship
        model_nulls = {}
        for model in ('Spin', 'BrainSMASH'):
            values = []
            for g, x in group_items:
                maps = maps_by_group[g][model]
                values.append(cached(root/model, f'{name}_group{g}',
                    lambda x=x, maps=maps: {'nulls': node_neighbor_relationship(x, network, maps)})['nulls'])
            model_nulls[model] = np.array(values)
        
        # rewired
        swap_fraction = np.nan
        if name != 'Euclidean':
            producer = partial(
                produce_rewired, name=name, network=network, distances=distances,
                groups=dict(group_items), rewire_n=rewire_n, n_jobs=n_jobs,
                rewire_bins=rewire_bins, rewire_swaps=rewire_swaps, seed=model_seed,
            )
            rewired = cached(root/'Rewired', name, producer)
            model_nulls['Rewired'] = rewired['nulls']
            swap_fraction = float(np.nanmean(rewired['swaps']))
        
        for row, (g, x) in enumerate(group_items):
            # calculate observed correlation
            r,p = calculate_neighbor_correlation_for_subject(x, network)
            summary = dict(Group=g, Group_Label=labels[g], Network=name,
                           Correlation=r, Pvalue_Spearman=p,
                           Num_Regions=len(x), Rewired_Mean_Swap_Fraction=swap_fraction)
            
            for model in ('Spin', 'BrainSMASH', 'Rewired'):
                values = model_nulls[model][row] if model in model_nulls else np.array([])
                summary[f'Pvalue_{model}'] = right_tail(r, values)
                if values.size:
                    null_tables.append(pd.DataFrame(dict(Group=g, Group_Label=labels[g], Network=name,
                        Null_Type=model, Permutation=np.arange(len(values)), Correlation=values)))
            summaries.append(summary)

    return pd.DataFrame(summaries), pd.concat(null_tables, ignore_index=True)
