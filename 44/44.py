# read raw data 
import numpy as np 
data = np.loadtxt('./44/quadratic_raw_data.csv', delimiter=',') 
x = data[:,0,None] 
y = data[:,1,None]


# generate quadratic features 
from sklearn.preprocessing import PolynomialFeatures 
poly = PolynomialFeatures(degree=2, include_bias=False) 
X_poly = poly.fit_transform(x) # X_poly: 1st column is x, 2nd column is x^2

# scale model inputs 
from sklearn.preprocessing import StandardScaler 
# fit linear model & predict 
from sklearn.linear_model import LinearRegression 
model = LinearRegression() 
model.fit(X_poly, y) 
y_predicted = model.predict(X_poly)


# plot  
import matplotlib.pyplot as plt #Used for model assessment and plotting
plt.figure(figsize=(4, 2)) 
plt.plot(x, y, 'o', label='raw data') 
plt.plot(x, y_predicted, label='quadratic fit') 
plt.legend() 
plt.xlabel('x'),plt.ylabel('y')

# Display the plot in alternative window
plt.show()