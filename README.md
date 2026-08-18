# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import NumPy, Pandas, and StandardScaler.
2.Read the 50_Startups.csv dataset using Pandas.
3.Select the independent variables X and dependent variable y.
4.Convert the input features into floating-point values.
5.Standardize the input features and target variable.
6.Add a column of ones to the feature matrix for the intercept.
7.Initialize the parameter vector theta with zeros.
8.Set the learning rate and number of iterations.
9.Calculate predictions using X × theta.
10.Calculate the error between predicted and actual values.
11.Calculate the gradient using (1/m) × Xᵀ × error.
12.Update theta using theta = theta − learning_rate × gradient.
13.Repeat the gradient calculation and parameter update for the specified number of iterations.
14.Use the final theta values to predict the output for new input data.
15.Standardize the new input data before prediction.
16.Convert the predicted standardized value back to the original scale.
17.Display the predicted value.
## Program:
```
Program to implement the linear regression using gradient descent.
Developed by:N.SWEDHA
RegisterNumber:212225230280
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression (X1, y, learning_rate=0.1, num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)

    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors= (predictions-y).reshape(-1,1)
    theta -learning_rate*(1/len(X1))*X.T.dot(errors)

    return theta

data=pd.read_csv(r"C:\Users\acer\Downloads\50_Startups.csv")
data.head(11)

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)

print("X =",X)

print("X1_Scaled =",X1_Scaled)

theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data= np.array([165349.2, 136897.8, 471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_scaled), theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform (prediction)
print("prediction =",prediction)
print(f"Predicted value: {pre}")

```

## Output:
<img width="445" height="887" alt="image" src="https://github.com/user-attachments/assets/ce13a787-8915-44de-a044-e9a07143b9ff" />

<img width="488" height="842" alt="image" src="https://github.com/user-attachments/assets/877d70c5-a7f8-46c6-b7dc-12bda8b481bd" />

<img width="893" height="376" alt="image" src="https://github.com/user-attachments/assets/4fbb7cc5-6e99-4d16-9ed6-80bd691229a1" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
