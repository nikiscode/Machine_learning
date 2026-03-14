import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
data = np.loadtxt(open("/content/sample_data/california_housing_train.csv","rb"),delimiter=',', skiprows=1)
-------------------------------------------------------------------------------------------------------------
data=np.loadtxt(open("/content/sample_data/california_housing_train.csv","rb"),delimiter=',', skiprows=1)
-------------------------------------------------------------------------------------------------------------
print(data.shape)
-------------------------------------------------------------------------------------------------------------
x=data[:,:-1]
y=data[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
------------------------------------------------------------------------------------------------------------
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
------------------------------------------------------------------------------------------------------------
from sklearn import linear_model
model =linear_model.LinearRegression(fit_intercept=True)
------------------------------------------------------------------------------------------------------------
model.fit(x_train,y_train)
------------------------------------------------------------------------------------------------------------
