import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
data = load_breast_cancer()
X,y = data.data,data.target
X_scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=2,random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
print(confusion_matrix(y,y_kmeans))
print(classification_report(y,y_kmeans))
pca = PCA(2)
X_pca = pca.fit_transform(X_scaled)
df = pd.DataFrame(X_pca,columns=['PC1','PC2'])
df['Cluster'] = y_kmeans
df['True Label'] = y
sns.scatterplot(data=df,x='PC1',y='PC2',hue='Cluster',palette='Set1')
plt.title("Clusters")
plt.show()
sns.scatterplot(data=df,x='PC1',y='PC2',hue='True Label',palette='coolwarm')
plt.title("True Labels")
plt.show()
centers = pca.transform(kmeans.cluster_centers_)
sns.scatterplot(data=df,x='PC1',y='PC2',hue='Cluster',palette='Set1')
plt.scatter(centers[:,0],centers[:,1],c='red',marker='X',s=200,label='Centroid')
plt.legend()
plt.title("Clusters with Centroids")
plt.show()    
