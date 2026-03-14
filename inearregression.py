class univariateLinearRegression:
    def __init__(self):
        self.theta_0=None
        self.theta_1=None

    def hypothesis(self,x):
        return self.theta_0 + self.theta_1 * x
        def grad_theta_0(self,x,y):
          return self.hypothesis(x)-y
          def grad_theta_1(self,x,y):
            return (self.hypothesis(x)-y)*x
        def fit(self,x,y epochs = 10,learning_rate =0.01):
         m=shape.X[0]
          for i in range(epochs):
            dtheta_0=0.0
            dtheta_1=0.0
            for x,y in zip(x,y):
                dtheta_0= dtheta_0+self.grad_theta_0
                dtheta_1= dtheta_1+self.grad_theta_1
            self.theta_0= self.theta_0-learning_rate*(dtheta_0)/m
            self.theta_1= self.theta_1-learning_rate*(dtheta_1)/m
