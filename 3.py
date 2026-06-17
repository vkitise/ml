import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
iris = load_iris()
data = iris.data
labels = iris.target
label_names = iris.target_names
data_reduced = PCA(n_components=2).fit_transform(data)
reduced_df = pd.DataFrame(data_reduced,columns=['Principal Component 1','Principal Component 2'])
reduced_df['Label'] = labels
colors = ['r','g','b']
for i in np.unique(labels):
    plt.scatter(
        reduced_df[reduced_df['Label']==i]['Principal Component 1'],
        reduced_df[reduced_df['Label']==i]['Principal Component 2'],
        color=colors[i], label=label_names[i]
    )
plt.title('PCA on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.show() 
