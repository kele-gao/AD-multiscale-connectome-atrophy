# Multiscale connectome-based analysis of regional brain atrophy

## Repository layout

```text
public_code/
├── data/
│   ├── sub_demo.csv
│   ├── sub_eTIV.csv
│   ├── sub_GrayVol.csv
│   └── RegionName.txt
├── resources/
│   ├── atlas/                 # Schaefer400 annotations and coordinates
│   └── connect/               # Connectome matrices (.npy)
├── scripts/
│   ├── 01_combat.py
│   ├── 02_Wscore.py
│   ├── 03_NNR.py
│   ├── 04_NNR_compare_across_group.py
│   ├── 05_long_NNR_LME.py
│   ├── node_neighbor_correlation.py
│   ├── independent_null_models.py
│   └── plotting_functions.py
└── results/ 
```

## Environment

All code was written in Python. The scripts import the following third-party packages:

```bash
python -m pip install numpy pandas scipy matplotlib seaborn scikit-learn \
    statsmodels patsy joblib tqdm neuroCombat neuromaps brainsmash \
    netneurotools nibabel
```

## Data

The data in the repository are demonstration data with simulated measurements. They are intended to illustrate input formats and analysis workflows. 

Data used in the study are available from the Alzheimer’s Disease Neuroimaging Initiative (ADNI; https://adni.loni.usc.edu) and Open Access Series of Imaging Studies (OASIS-3; https://www.oasis-brains.org) repositories. 

## Resources

Schaefer400 annotations and coordinates are available from https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal. Connectome matrices are available from https://github.com/netneurolab/hansen_many_networks.


## Running the workflow

Run commands from the repository root. The main scripts currently use an empty path/Path setting. Inspect each script's configuration before running.

### 01 ComBat harmonization

Uses SITEID as the batch variable; Sex, PTHAND, and Group_Code_new as categorical covariates; and Age, PTEDUCAT, and eTIV as continuous covariates. 

Outputs in results/1_combat/ include T1_demoData.csv, T1_GrayVol.csv, T1_eTIV.csv, and T2_GrayVol_combat.csv.

### 02 W-scores

Fits regional reference regressions using group 0 and the covariates Sex_Code, Age, PTHAND, PTEDUCAT, and eTIV. The implemented direction is:

```text
W-score = (predicted volume - observed volume) / reference residual SD
```

Values below the configured threshold are set to zero. Outputs are results/2_Wscore/W_scores_threshold0.csv and W_scores_demo.csv. 

### 03 Cross-sectional node-neighbor relationship (NNR)

Computes Spearman correlations between regional W-scores and their connectome-derived neighbor values. 

Group-level analyses use Spin, BrainSMASH, and distance/degree-constrained rewiring nulls. Current defaults include 5,000 permutations per model and N_JOBS=160; reduce parallelism to suit available CPU and memory.

Outputs are under results/3_NNR/1_cross_section/, with group-level and subject-level subdirectories. 

Null checkpoints are reused by filename. Changing inputs, parameters, seeds, or group ordering may require moving the affected caches aside.

### 04 Cross-sectional group comparisons

Applies Fisher-z transformation and ANCOVA. Outputs include ancova_results.csv and network raincloud plots under 2_subject_level_analysis/plot_by_group/.

### 05 Longitudinal LME

The LME model is fitted separately within each group:

```text
Fisher-z(NNR) ~ time * network + baseline covariates
                + (1 + time | SubjectID)
```

 Outputs are under results/3_NNR/2_longitudinal/.

