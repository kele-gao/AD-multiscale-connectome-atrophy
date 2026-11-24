import os
import numpy as np
import pandas as pd

# ================= Input settings =================
Path = r''
file_demo = os.path.join(Path, 'results','1_combat', 'T1_demoData.csv')
file_image = os.path.join(Path, 'results','1_combat', 'T2_GrayVol_combat.csv')
file_eTIV = os.path.join(Path, 'results','1_combat', 'T1_eTIV.csv')
dir_out =  os.path.join(Path, 'results','2_Wscore')
threshold = 0 

os.makedirs(dir_out, exist_ok=True)

covariates = ['Sex_Code', 'Age', 'PTHAND', 'PTEDUCAT', 'eTIV']

data_GMV = pd.read_csv(file_image)
data_demo = pd.read_csv(file_demo)
data_eTIV = pd.read_csv(file_eTIV)

# Reorder the three data matrices to ensure consistent row order
order = data_demo["SubjectID"]
data_demo_sorted = data_demo.copy()

data_GMV_sorted = (
    data_GMV
    .set_index("Subject ID")
    .loc[order]
    .reset_index()
)

data_eTIV_sorted = (
    data_eTIV
    .set_index("Subject ID")
    .loc[order]
    .reset_index()
)

same_order_1 = data_GMV_sorted['Subject ID'].equals(data_eTIV_sorted['Subject ID'])
same_order_2 = data_GMV_sorted['Subject ID'].equals(data_demo_sorted['SubjectID'])

print("GMV vs eTIV order matches:", same_order_1)
print("GMV vs demo order matches:", same_order_2)

numROI = data_GMV.shape[1] - 1  # The first column is assumed to contain subject_id

# ================= Construct variables =================
# Y_all: GMV matrix for all ROIs
Y_all = data_GMV_sorted.iloc[:, 1:].to_numpy(dtype=float)

# X_all: Demographic table with eTIV added
X_all = data_demo_sorted.copy()
X_all['eTIV'] = data_eTIV_sorted['eTIV'].values

# ================= Identify healthy control subjects =================
# Logic: A subject is considered HC only if Group_Code_New is 0 in all rows for the same subject_id
HC_index = X_all['Group_Code_new'] == 0 # 
print(f"Number of healthy control subjects: {HC_index.sum()}")
print(f"Number of patient subjects: {(~HC_index).sum()}")


# ================= Estimate W-score coefficients using healthy control data =================
# Healthy control design matrix
X_HC = X_all.loc[HC_index, covariates].to_numpy(dtype=float)
X_HC = np.column_stack([np.ones(X_HC.shape[0]), X_HC])  # Add an intercept
Y_HC = Y_all[HC_index, :]

# Store regression coefficients and residual standard deviations for each ROI
b_HC = np.zeros((X_HC.shape[1], numROI))
dev_HC = np.zeros(numROI)

n_hc, p = X_HC.shape

for roi_idx in range(numROI):
    y = Y_HC[:, roi_idx]

    # Least-squares fitting, equivalent to MATLAB glmfit(..., 'normal')
    beta, residuals, rank, s = np.linalg.lstsq(X_HC, y, rcond=None)
    b_HC[:, roi_idx] = beta

    # Calculate the residual standard deviation
    y_pred = X_HC @ beta
    resid = y - y_pred
    sigma = np.sqrt(np.sum(resid ** 2) / (n_hc - p))
    dev_HC[roi_idx] = sigma

# ================= Calculate patient W-scores =================
X_patients = X_all.loc[~HC_index, covariates].to_numpy(dtype=float)
X_patients = np.column_stack([np.ones(X_patients.shape[0]), X_patients])
Y_patients = Y_all[~HC_index, :]
Patients_index = X_all.loc[~HC_index, 'Group_Code_new'].to_numpy() # _New

# Predicted values: n_patients x numROI
Y_predict = X_patients @ b_HC

# W-score = (predicted value - observed value) / healthy control residual standard deviation
W_scores = (Y_predict - Y_patients) / dev_HC

# ================= Apply threshold =================
W_scores_thre = W_scores.copy()
W_scores_thre[W_scores < threshold] = 0

# ================= Save results =================
column_names = data_GMV_sorted.columns[1:].tolist()
row_names = X_all.loc[~HC_index, "SubjectID"].values

W_scores_table = pd.DataFrame(W_scores_thre, columns=column_names)
W_scores_table.insert(0, "Subject ID", row_names)

W_scores_demo = X_all.loc[~HC_index, :].copy()

W_scores_table.to_csv(os.path.join(dir_out, 'W_scores_threshold'+str(threshold)+'.csv'), index=False)
W_scores_demo.to_csv(os.path.join(dir_out, 'W_scores_demo.csv'), index=False, encoding='utf-8-sig')

ids_wscore = W_scores_table["Subject ID"].astype(str).str.strip().reset_index(drop=True)
ids_demo = W_scores_demo["SubjectID"].astype(str).str.strip().reset_index(drop=True)

print("Subject counts match:", len(ids_wscore) == len(ids_demo))
print("Subject order matches exactly:", ids_wscore.equals(ids_demo))
print("Number of shared subjects:", len(set(ids_wscore) & set(ids_demo)))

print('Great!')

