import os.path
import neuroCombat
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


path = r''
File_img = os.path.join(path, 'data', 'sub_GrayVol.csv')
File_eTIV = os.path.join(path, 'data', 'sub_eTIV.csv')
File_demo = os.path.join(path, 'data', 'sub_demo.csv')

Dir_out = os.path.join(path, 'results', '1_combat')

batch_col = 'SITEID'
categorical_cols = ['Sex', 'PTHAND', 'Group_Code_new']  
continuous_cols = ["Age", "PTEDUCAT", "eTIV"]

os.makedirs(Dir_out, exist_ok=True)


Data_img = pd.read_csv(File_img)
Data_demo = pd.read_csv(File_demo)
Data_eTIV = pd.read_csv(File_eTIV)
Data_demo = Data_demo.rename(columns={"Subject ID": "SubjectID"})


# Reorder the three data matrices to ensure consistent row order
order = Data_demo["SubjectID"]

Data_img = (
    Data_img
    .set_index("Subject ID")
    .loc[order]
    .reset_index()
)

Data_eTIV = (
    Data_eTIV
    .set_index("Subject ID")
    .loc[order]
    .reset_index()
)

same_order_1 = Data_img['Subject ID'].equals(Data_eTIV['Subject ID'])
same_order_2 = Data_img['Subject ID'].equals(Data_demo['SubjectID'])

if not same_order_1 or not same_order_2:
    raise ValueError("Data_img、Data_eTIV、Data_demo 的 Subject ID 顺序不一致。")
else:
    print("Subject IDs are in the same order in Data_img, Data_eTIV, and Data_demo.")    

Data_demo.to_csv(os.path.join(Dir_out, 'T1_demoData.csv'), index=False, encoding='utf-8-sig')
Data_img.to_csv(os.path.join(Dir_out, 'T1_GrayVol.csv'), index=False)
Data_eTIV.to_csv(os.path.join(Dir_out, 'T1_eTIV.csv'), index=False)

Data_demo['eTIV'] = Data_eTIV['eTIV'].values

features = Data_img.iloc[:, 1:].values.T  # Transpose to (features, subjects)

# Specify the batch column and covariates
covar_cols = [batch_col] + categorical_cols + continuous_cols
covars = Data_demo[covar_cols].copy()# Prepare covariate data

# Check the number of subjects at each site
site_counts = covars[batch_col].value_counts()
single_subject_sites = site_counts[site_counts == 1].index.tolist()
multi_subject_sites = site_counts[site_counts > 1].index.tolist()


# Create the results matrix, initialized with the original data
result_features = features.copy()

# Apply ComBat correction if at least two sites have multiple subjects
print("Starting ComBat batch effect correction for sites with multiple subjects...")

# Select data from sites with multiple subjects
multi_site_indices = covars[batch_col].isin(multi_subject_sites)
multi_site_features = features[:, multi_site_indices]
multi_site_covars = covars[multi_site_indices].copy()

# Apply ComBat correction to sites with multiple subjects
combat_result = neuroCombat.neuroCombat(
    dat=multi_site_features,
    covars=multi_site_covars,
    batch_col=batch_col,
    categorical_cols=categorical_cols,
    continuous_cols=continuous_cols
)

# Retrieve the corrected data
corrected_multi_site_features = combat_result["data"]

# Insert the corrected data into the corresponding positions in the results matrix
result_features[:, multi_site_indices] = corrected_multi_site_features

# Convert the results matrix back to the original format (subjects x features)
result_data = result_features.T

# Create the corrected DataFrame
Data_img_corrected = Data_img.copy()
Data_img_corrected.iloc[:, 1:] = result_data

# Save the corrected data
output_file = os.path.join(Dir_out, 'T2_GrayVol_combat.csv')
Data_img_corrected.to_csv(output_file, index=False)
print(f"Corrected data saved to: {output_file}")
