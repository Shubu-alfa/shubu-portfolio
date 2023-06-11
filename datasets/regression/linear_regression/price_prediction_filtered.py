# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


car_data = pd.read_csv("D:/Data Science with Shubham/Datasets/regression/filtered_data.csv")

print(car_data.head())
#------------------IMPLEMENT LINEAR REGRESSION MODEL-----------------------------------------------------

from sklearn import linear_model

from sklearn.model_selection import train_test_split

print(car_data.columns)

y = car_data.iloc[:,5]
print(y.head())
X = car_data.drop(['price'], axis = 1)
#Model Building

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25)

regr = linear_model.LinearRegression().fit(X_train,y_train)

print(regr.coef_)

print(regr.intercept_)

predt = regr.predict(X_test)

regr.score(X_test,y_test)

from sklearn.metrics import mean_squared_error, r2_score

print("Mean Squared Error=",mean_squared_error(y_test,predt))

print("R Squared:",r2_score(y_test,predt))

import matplotlib.pyplot as plt

plt.hist(y_test)
plt.hist(predt)

plt.show()

#list = [[4,9,111,5000,27,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,1,0,0],[4,9,111,5000,27,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,1,0,0]]
#predicted_score = regr.predict(list)
#print("predicted_score:",predicted_score[0])


# Saving model to disk
pickle.dump(regr, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
list = [[4,9,111,5000,27,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,1,0,0],[4,9,111,5000,27,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,1,0,0]]
#predicted_score = regr.predict(list)
print(model.predict(list))
