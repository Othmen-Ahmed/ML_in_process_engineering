# read raw data 
import numpy as np 
data = np.loadtxt('./44/quadratic_raw_data.csv', delimiter=',') 
x = data[:,0,None] 
y = data[:,1,None]



# create pipeline for quadratic fit via linear model  
# import relevant classes 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression  

# add transformers and estimators sequentially as list of tuples 
# the names ‘poly’, ‘scaler’, ‘model’ can be used to access the individual elements of pipeline later  
pipe = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)),                                
                ('scaler', StandardScaler()),                              
                ('model', LinearRegression())])  

# fit pipeline and predict 
pipe.fit(x, y) 
y_predicted = pipe.predict(x) 


# plot  
import matplotlib.pyplot as plt #Used for model assessment and plotting
plt.figure(figsize=(4, 2)) 
plt.plot(x, y, 'o', label='raw data') 
plt.plot(x, y_predicted, label='quadratic fit') 
plt.legend() 
plt.xlabel('x'),plt.ylabel('y')

# Display the plot in alternative window
plt.show()