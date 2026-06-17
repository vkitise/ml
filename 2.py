import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True).frame

correlation_matrix = data.corr()

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

sns.pairplot(data, diag_kind='kde', plot_kws={'alpha':0.5})
plt.suptitle("Pair Plot", y=1.02)
plt.show()  
