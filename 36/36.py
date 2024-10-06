# import libraries 
import numpy as np 
from sklearn.preprocessing import PolynomialFeatures  # Used for data transformations   
from sklearn.preprocessing import StandardScaler # Used for data transformations 
from sklearn.linear_model import LinearRegression  #Used for building linear models
from sklearn.metrics import r2_score #Used for model assessment and plotting
import matplotlib.pyplot as plt #Used for model assessment and plotting

# #For debugging purposes
# import os
# print(os.getcwd()," Current directory")  # Prints the current working directory
# print(os.listdir()," Current  directory files")  # Lists the files in the current directory
# #Debugging ends

#read data 
data = np.loadtxt('./36/quadratic_raw_data.csv', delimiter=',') 
x = data[:,0:1]; 
y = data[:,1:] # equivalent to y = data[:,1,None] which returns 2D array


# Pre-process / Feature engineering 
poly = PolynomialFeatures(degree=2, include_bias=False) 
X_poly = poly.fit_transform(x)  # X_poly: 1st column is x; 2nd column is x^2  

# scale model input variables 
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X_poly) 


# fit linear model & predict 
model = LinearRegression() 
model.fit(X_poly, y) 
y_predicted = model.predict(X_poly) 

# Assess model accuracy 
print('Fit accuracy = ', r2_score(y, y_predicted))

# plot predictions 
plt.figure(figsize=(4, 2)) 
plt.plot(x, y, 'o', label='raw data') 
plt.plot(x, y_predicted, label='quadratic fit') 
plt.legend() 
plt.xlabel('x'),plt.ylabel('y')

# Display the plot in alternative window
plt.show()

#plt.savefig('./36/plot.png')  # Save the plot as a PNG file