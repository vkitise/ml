import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
X = fetch_california_housing(as_frame=True).data[["AveRooms"]]
y = fetch_california_housing(as_frame=True).target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
plt.scatter(X_test,y_test,label="Actual")
plt.plot(X_test,y_pred,'r',label="Predicted")
plt.xlabel("Average Rooms")
plt.ylabel("House Value")
plt.title("Linear Regression")
plt.legend()
plt.show()
print("MSE:",mean_squared_error(y_test,y_pred))
print("R²:",r2_score(y_test,y_pred))
url="https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
cols=["mpg","cyl","disp","hp","wt","acc","year","org"]
data=pd.read_csv(url,sep="\s+",names=cols,na_values="?").dropna()
X,y=data[["disp"]],data["mpg"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
poly=make_pipeline(PolynomialFeatures(2),StandardScaler(),LinearRegression())
poly.fit(X_train,y_train)
y_pred=poly.predict(X_test)
plt.scatter(X_test,y_test,label="Actual")
plt.scatter(X_test,y_pred,c='r',label="Predicted")
plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.title("Polynomial Regression")
plt.legend()
plt.show()
print("MSE:",mean_squared_error(y_test,y_pred))
print("R²:",r2_score(y_test,y_pred))  
