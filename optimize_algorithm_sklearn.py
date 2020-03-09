
"""
This is a demo of how you can optimize machine learning algorithm results with
only linear programming tools like scipy.optimize
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import train_test_split


def mape(y, y_pred, weight=1):
    """Custom metric with bias possibilities. Weight parameter will multiply
    every prediction by it's value thus changing the mape error.

    Arguments:
        y {array} -- ground truth
        y_pred {array} -- prediction made by estimator

    Keyword Arguments:
        weight {int} -- custom bias for y_pred (default: {1})

    Returns:
        [float] -- mape error with added bias (default = 1)
    """
    mape = round(mean_absolute_error(y, y_pred*weight) / np.mean(y), 4)
    return mape


def objective(x):
    """Simple optimization function for sklearn.optimize

    Arguments:
        x {float} -- initial value to start optimization from

    Returns:
        float -- error minimized by sklearn.optimize function
    """
    return mean_absolute_error(y_train, y_pred_train * x)


# Let's load dataset and select only numerical data for simplicity sake
df = pd.read_csv('datasets/weatherHistory.csv')
num = df.select_dtypes(include='float64').columns
df = df[num]

# cut data between target value and features
y = df['Pressure (millibars)']
X = df.loc[:, df.columns != 'Pressure (millibars)']

# this requires no explanation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

# define model and predict train and test data
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# let's further minimize error by calling minimize on our predictions
x0 = 1.0
sol = minimize(objective, x0, method='Nelder-Mead')
print('\n'+str(sol)+'\n')

# weight is the optimized value that will further reduce error on our prediction
# you have to multiply weight value by y_train to see how good it optimizes
# result. After that test it on test data
weight = sol.x[0]

print('mape on train', mape(y_train, y_pred_train))
print('mape on train scipy optimized', mape(y_train, y_pred_train, weight))
print('mape on test', mape(y_test, y_pred_test))
print('mape on test scipy optimized', mape(y_test, y_pred_test, weight))

# We can see that after optimizing our predictions error went down. This is done
# on toy dataset so reduction is small. Test it on real business problem to se
# better results
