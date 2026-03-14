import numpy as np
fromsklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook

class univariateLinearRegression:
    def __init__(self):
        self.theta_0 = 0.0
        self.theta_1 = 0.0

    def hypothesis(self, x):
        return self.theta_0 + self.theta_1 * x

    def grad_theta_0(self, x, y):
        return self.hypothesis(x) - y

    def grad_theta_1(self, x, y):
        return (self.hypothesis(x) - y) * x

    def fit(self, X, Y, epochs=10, learning_rate=0.01):
        n_samples = X.shape[0]

        for i in range(epochs):
            dtheta_0 = 0.0
            dtheta_1 = 0.0
            for x_val, y_val in zip(X.flatten(), Y.flatten()):
                dtheta_0 += self.grad_theta_0(x_val, y_val)
                dtheta_1 += self.grad_theta_1(x_val, y_val)

            self.theta_0 = self.theta_0 - learning_rate * (dtheta_0 / n_samples)
            self.theta_1 = self.theta_1 - learning_rate * (dtheta_1 / n_samples)

    def predict(self, X):
        y_pred = []
        for x_val in X.flatten():
            y_pred.append(self.hypothesis(x_val))
        return np.array(y_pred)
 ----------------------------------------------------------------------------------------------------------
model= univariateLinearRegression()
-----------------------------------------------------------------------------------------------------------
import numpy as np
data = np.loadtxt(open("/content/data_train_sv (1).csv","rb"),delimiter=",")
-----------------------------------------------------------------------------------------------------------
x=data[:,0]
x=x.reshape((-1,1))
y=data[:,-1]
y=y.reshape((-1,1))
-----------------------------------------------------------------------------------------------------------
model.fit(x,y)
------------------------------------------------------------------------------------------------------------
test_data = np.loadtxt(open("/content/data_train_sv (1).csv","rb"),delimiter=",")
------------------------------------------------------------------------------------------------------------
