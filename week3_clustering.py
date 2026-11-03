# Week 3: Unsupervised Learning and Clustering Analysis
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

df = pd.read_csv("data/titanic_eda_dataset.csv")

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
X_df = df[features].apply(pd.to_numeric, errors="coerce").dropna()

# Scale features because K-Means is distance-based.
scaler = StandardScaler()
X = scaler.fit_transform(X_df)

# Compare K=2 to K=6 using inertia and silhouette score.
scores = []
models = {}
for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    scores.append((k, model.inertia_, silhouette_score(X, labels)))
    models[k] = model

scores_df = pd.DataFrame(scores, columns=["K", "Inertia", "Silhouette"])
best_k = int(scores_df.loc[scores_df["Silhouette"].idxmax(), "K"])

# Final K-Means model
model = models[best_k]
X_df["Cluster"] = model.labels_

print(scores_df)
print(X_df.groupby("Cluster")[features].mean())

# PCA visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=model.labels_)
plt.title(f"K-Means Clusters Visualized with PCA (K={best_k})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
