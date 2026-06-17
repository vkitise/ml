import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True).frame
num_cols = housing.select_dtypes(include=np.number).columns
for i, col in enumerate(num_cols):
    plt.subplot(3,3,i+1)
    sns.histplot(housing[col], kde=True, bins=30)
    plt.title(col)
plt.tight_layout()
plt.show()
for i, col in enumerate(num_cols):
    plt.subplot(3,3,i+1)
    sns.boxplot(x=housing[col])
    plt.title(col)
plt.tight_layout()
plt.show()
print("Outliers Detection:")
for col in num_cols:
    Q1 = housing[col].quantile(0.25)
    Q3 = housing[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = housing[(housing[col] < Q1-1.5*IQR) |
                       (housing[col] > Q3+1.5*IQR)]
    print(col, ":", len(outliers), "outliers")
print("\nDataset Summary:")
print(housing.describe())  
