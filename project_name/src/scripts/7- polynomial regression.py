import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from Functions import *


# ----------------- Load the training and testing datasets
train_data = pd.read_csv('cleaned_train_data.csv')
test_data = pd.read_csv('normalized_test_data.csv')

# ------------------ let's consider columns 'Column1' and 'Column2'
column1 = 'input20'
column2 = 'output'

# ----------------- Define independent (X) & dependent variable (Y)
X_train = train_data[column1].values.reshape(-1, 1)
y_train = train_data[column2].values


# ------------------ Select data for testing
X_test = test_data[column1].values.reshape(-1, 1)
y_test = test_data[column2].values


# ------------------ Create a .csv file to save the results
a = open(f'PR_grid search_{column1} vs {column2}.csv','w')
a.write(f'{column1} vs {column2},MSE, RMSE, STD\n')

# ----------------- Create polynomial features
# You can change the polynomial degree
for i in [2,3,4]:

    degree = i  
    poly = PolynomialFeatures(degree=degree)

    # Transform data
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Fit a polynomial regression model to the training data
    model = LinearRegression().fit(X_train_poly, y_train)

    # Print the coefficients
##    print('Coefficient: ', model.coef_)
##    print('Intercep0.t: ', model.intercept_)


    # ----------------- Make predictions on the testing data
    y_pred = model.predict(X_test_poly) 
    ##print(f"predicted response:\n{y_pred}")


    # -------------------- Evaluation (Error)
    # Calculate the Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    a.write('deg %d,%f,'%(i,mse))

    # Calculate the Root Mean Squared Error
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    a.write('%f,'%rmse)

    # Calculate R-squared(coefficient of determination)
    r2 = r2_score(y_test, y_pred)

##    print(f'Mean Squared Error: {mse}')
##    print(f"R-squared: {r2}")

    # Calculate residuals
    residuals = y_test - y_pred
    
    # Calculate standard deviation of residuals
    std_deviation = np.std(residuals)
    a.write('%f\n'%std_deviation)

    # -------------------- Create a new figure for each iteration
    plt.figure()  # Create a new figure for each polynomial degree plot

    # -------------------- Create the scatter plot
    colors = ['blue', 'green', 'grey']
    plt.scatter(X_train, y_train, color=colors[i-2], label='Data points')
    ##plt.title(f'Scatter Plot of {column1} vs {column2} with Polynomial Regression')
    plt.xlabel(column1)
    plt.ylabel(column2)

    # -------------------- Plotting the polynomial regression line
    plt.plot(X_test, y_pred, 'o', color='red', label=
             f'Polynomial Regression line (deg={degree})')

    # Show legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11),
              fancybox=True, shadow=True, ncol=5)

    props = dict(boxstyle='round', facecolor='red', alpha=0.15)
    # Displaying Error values on the plot
    plt.text(0.85, -0.1,
             f'MSE = {mse:.3f} / std ={std_deviation:.3f}',
             verticalalignment='top', horizontalalignment='center',
             transform=plt.gca().transAxes,
             bbox = props, fontsize=10)

    # save the plot
    plt.savefig(f'PR_deg{degree}_{column1} vs {column2}.png', dpi=150,
                bbox_inches='tight')

    # Show the plot
a.close()


##plt.scatter(X_train[:, 0], y_train, label='Training Data')
##plt.scatter(X_test[:, 0], y_test, label='Testing Data')
###x_range = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
###y_range = model.predict(poly.transform(x_range.reshape(1, -1)))
####plt.plot(x_range, y_range, color='red', label='Polynomial Regression')







