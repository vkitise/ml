import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
data = np.random.rand(100)
train_data = data[:50]
train_labels = ["Class1" if x <= 0.5 else "Class2" for x in train_data]
test_data = data[50:]
def knn(train_x, train_y, test_x, k):
    dist = [(abs(test_x - train_x[i]), train_y[i]) for i in range(len(train_x))]
    dist.sort()
    return Counter([label for _, label in dist[:k]]).most_common(1)[0][0]
k_values = [1, 2, 3, 4, 5, 20, 30]
results = {}
for k in k_values:
    results[k] = [knn(train_data, train_labels, x, k) for x in test_data]
for k in k_values:
    class1 = [test_data[i] for i in range(len(test_data))
              if results[k][i] == "Class1"]
    class2 = [test_data[i] for i in range(len(test_data))
              if results[k][i] == "Class2"]
    plt.scatter(train_data, [0]*50,
                c=["blue" if y=="Class1" else "red" for y in train_labels])
    plt.scatter(class1, [1]*len(class1), c="blue", marker="x")
    plt.scatter(class2, [1]*len(class2), c="red", marker="x")
    plt.title(f"k-NN Result (k={k})")
    plt.show()   
