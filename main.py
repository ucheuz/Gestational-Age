import pandas as pd

# Read the data
df = pd.read_csv("dataset-fetal-brain-reg.csv")
df
--------
import numpy as np
from sklearn.preprocessing import StandardScaler

# convert to numpy
data = df.to_numpy()

# Create label/target vector
y = data[:,9] 

# Extract and scale features
features = StandardScaler().fit_transform(data[:,:9])

# Create feature matrix with extra column of ones
X = np.ones(data.shape);
X[:,1:10]=features

# Print the first few lines of the feature matrix
print(np.around(X[:4,:],1))
--------
# Print dimensions X
print('Dimensions X: ', X.shape) 

# Print dimensions y
print('Dimensions y: ', y.shape) 
-------
# training set, the first 120 samples
X_train= X[:120,:]
y_train= y[:120]

# test set, the remaining samples
X_test= X[120:,:]
y_test= y[120:]

# print dimensions
print('Number of samples in training set: feature matrix {}, target vector {}.'.format(X_train.shape[0],y_train.shape[0]))
print('Number of samples in test set: feature matrix {}, target vector {}.'.format(X_test.shape[0],y_test.shape[0]))
--------
# Create matrix Lambda
lambda_value = 0.1
D = X_train.shape[1]
Lambda = np.diag([0] + [lambda_value] * (D - 1))

print(Lambda)
-------
# Implement equation
w = np.linalg.inv(X_train.T @ X_train + Lambda) @ X_train.T @ y_train

# Print dimension of w 
print('Dimension of w is ', w.shape) 

# print intercept 
print('Intercept: ', np.round(w[0],1)) 

# print slopes
print('Slopes: ', np.round(w[1:],1)) 
--------
# Evaluate GA on the first subject in training set and print the result
predicted_ga_subject_0 = X_train[0,:] @ w
print('Predicted GA of subject 0: ', round(predicted_ga_subject_0, 1))

# Print true age 
print('True GA of subject 0: ', round(y_train[0], 1))
--------
# Implement function RMSE

def RMSE(y,y_pred):
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    print('RMSE: ',round(rmse,2)) 

# predict on training set
y_train_pred = X_train @ w

# calculate error on training set
print('Training error:')
rmse_train = RMSE(y_train, y_train_pred)

# predict on test set
y_test_pred = X_test @ w

# calculate error on test set
print('Test error: ',)
rmse_test = RMSE(y_test, y_test_pred)

print('As the root mean squared error of the test set is greater than that of the training set we can say the model performs better on the training set than on the test set, suggesting it may be slightly overfitting.')
----------
from matplotlib import pyplot as plt

# plot predicted against expected target values
plt.scatter(y_test, y_test_pred, label= 'original target values')
plt.plot([22, 38], [22, 38], 'r', label="$y=\\hat{y}$")
plt.xlabel("Original Target Values (y)")
plt.ylabel("Predicted Target Values (ŷ)")
plt.title("Predictions on test set")
_=plt.legend(fontsize = 12)

# Tuning the penalty weight λ to help control the balance between bias and variance can improve the model fitting
---------
# create training and test feature matrices
X1_train = X_train[:, 1:]
X1_test = X_test[:, 1:]

# print the first few lines of the train matrix
print('Train:')
print(np.around(X1_train[:3,:],1))

# print the first few lines of the test matrix
print('\nTest:')
print(np.around(X1_test[:3,:],1))
---------
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse

# select model 
model = Ridge(alpha = 0.1)

# fit to the training set 
model.fit(X1_train, y_train)

# calculate RMSE on training set 
y_train_pred = model.predict(X1_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
print('Training RMSE: ',round(rmse_train,2))

# calculate RMSE on test set - 2 marks (0 marks if not on test set)
y_test_pred = model.predict(X1_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
print('Test RMSE: ',round(rmse_test,2))
