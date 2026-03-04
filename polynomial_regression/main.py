import least_squares as ls
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x_train = np.random.uniform(0, 2 * np.pi, 50)
noise_train = np.random.normal(0, 0.1, 50) 
y_train = np.sin(x_train) + noise_train

x_test = np.linspace(0, 2 * np.pi, 100)
noise_test = np.random.normal(0, 0.1, 100)
y_test = np.sin(x_test) + noise_test

d = 10

A_train = ls.Vandermonde(x_train, d)
theta = ls.solver(A_train, y_train)
train_model = A_train @ theta

print(theta)

A_test = ls.Vandermonde(x_test, d)
test_model = A_test @ theta

plt.scatter(x_train, y_train)
plt.scatter(x_train, train_model, color='g')
plt.show()

plt.scatter(x_test, y_test)
plt.scatter(x_test, test_model, color='g')
plt.show()

train_error = np.sqrt(np.mean((y_train - train_model)**2))
test_error = np.sqrt(np.mean((y_test - test_model)**2))

print(train_error, '\n', test_error)

