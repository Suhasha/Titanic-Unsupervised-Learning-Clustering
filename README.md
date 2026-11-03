# Week 3 – Unsupervised Learning and Clustering Analysis

This project applies K-Means clustering to the cleaned Titanic dataset.

## Method
- Select passenger-level numerical features
- Exclude `Survived` from clustering because the task is unsupervised
- Standardize features with StandardScaler
- Test K=2 through K=6
- Compare inertia and silhouette scores
- Select the best K
- Visualize clusters using PCA
- Profile cluster characteristics
- Examine survival only after clustering for interpretation

## Main files
- `code/week3_clustering.py`
- `data/titanic_clustered_data.csv`
- `data/k_selection_metrics.csv`
- `data/cluster_summary.csv`
- `data/cluster_survival_rates.csv`
- Four visualization PNG files
- `report/Week3_Unsupervised_Learning_Clustering_Report.docx`

Dataset: Kaggle Titanic – Machine Learning from Disaster
https://www.kaggle.com/competitions/titanic/data
