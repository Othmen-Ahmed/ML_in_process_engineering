# import library, fit, and transform 
import numpy as np 
from sklearn.preprocessing import StandardScaler 

X = np.array([[ 1000, 0.01,  300], [ 1200,  0.06,  350], [ 1500,  0.1, 320]]) 
scaler = StandardScaler().fit(X) # compute mean & std column-wise 
X_scaled = scaler.transform(X)  # transform using computed mean and std 

print(X)
print(X_scaled)
print(scaler.mean_) 



#Min Max Scaling 