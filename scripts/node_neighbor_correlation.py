
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def calculate_neighborhood_deviation_for_subject(subject_data, connectivity_matrix):

    numRegions = subject_data.shape[0]
    neighbor_weighted_avg = np.zeros(numRegions)

    for i in range(numRegions):
        
        neighbor_indices = np.where((connectivity_matrix[i, :] > 0) &
                                    (np.arange(numRegions) != i))[0]

        N_i = len(neighbor_indices)  

        if N_i > 0:
            weighted_sum = 0
            for j in neighbor_indices:
                p_j = subject_data[j]  
                C_ij = connectivity_matrix[i, j]  
                weighted_sum += p_j * C_ij

            neighbor_weighted_avg[i] = weighted_sum / N_i
        else:

            neighbor_weighted_avg[i] = np.nan

    return neighbor_weighted_avg


def calculate_neighbor_correlation_for_subject(subject_data, connectivity_matrix):

    
    neighbor_weighted_avg = calculate_neighborhood_deviation_for_subject(
        subject_data, connectivity_matrix
    )

    
    valid_indices = ~np.isnan(neighbor_weighted_avg)

    if np.sum(valid_indices) < 2:
        
        return np.nan, np.nan

    
    valid_subject_data = subject_data[valid_indices]
    valid_neighbor_avg = neighbor_weighted_avg[valid_indices]

    
    correlation, p_value = spearmanr(valid_subject_data, valid_neighbor_avg)

    return correlation, p_value


def calculate_network_correlations_for_group(network_name, network_matrix, group_data, group_name, subject_indices, group_labels):

    group_label = group_labels.get(group_name, f'Group {group_name}')

    num_subjects_in_group = group_data.shape[0]
    correlations = np.zeros(num_subjects_in_group)
    p_values = np.zeros(num_subjects_in_group)

    for i, sub_idx in enumerate(range(num_subjects_in_group)):
        correlation, p_value = calculate_neighbor_correlation_for_subject(
            group_data[sub_idx, :], network_matrix
        )

        correlations[i] = correlation
        p_values[i] = p_value

    result_df = pd.DataFrame({
        'SubjectID': subject_indices,
        'group': group_name, 'group_label': group_label, 
        f'{network_name}_NNR': correlations,
        f'{network_name}_pvalue': p_values
    })

    return result_df
