import pandas as pd

car_data = pd.read_csv("D:/Data Science with Shubham/Datasets/regression/Automobile price data _Raw.csv")

#print(car_data.head())

print(car_data.info())

print(car_data.corr())
#----------------------------------------encoding make--------
one_hot = pd.get_dummies(car_data['make'])
car_data = car_data.drop('make',axis = 1)
car_data = car_data.join(one_hot)
#print(car_data.head())
#output_excel = car_data1.to_excel("D:/Data Science with Shubham/Datasets/regressionout_excel.xlsx")
#-----------------------------------------encoding fuel_type-----
one_hot1 = pd.get_dummies(car_data['fuel-type'])
car_data = car_data.drop('fuel-type',axis = 1)
car_data = car_data.join(one_hot1)
print(car_data.head())
#----------------------------------------encoding aspiration--------
one_hot2 = pd.get_dummies(car_data['aspiration'])
car_data = car_data.drop('aspiration',axis = 1)
car_data = car_data.join(one_hot2)
print(car_data.head())
#-----------------------------------------encoding no-of-doors-------
one_hot2 = pd.get_dummies(car_data['num-of-doors'])
car_data = car_data.drop('num-of-doors',axis = 1)
car_data = car_data.join(one_hot2)
print(car_data.head())
#----------------------------------------encoding body-style--------
one_hot2 = pd.get_dummies(car_data['body-style'])
car_data = car_data.drop('body-style',axis = 1)
car_data = car_data.join(one_hot2)
print(car_data.head())
#----------------------------------------encoding drive wheels------
one_hot2 = pd.get_dummies(car_data['drive-wheels'])
car_data = car_data.drop('drive-wheels',axis = 1)
car_data = car_data.join(one_hot2)
print(car_data.head())
#---------------------------------------encoding engine location----
one_hot2 = pd.get_dummies(car_data['engine-location'])
car_data = car_data.drop('engine-location',axis = 1)
car_data = car_data.join(one_hot2)
print(car_data.head())
#output_excel = car_data.to_excel("D:/Data Science with Shubham/Datasets/regression/out_excel.xlsx")
#----------------------------------------Encoding  engine-type---------
one_hot2 = pd.get_dummies(car_data['engine-type'])
car_data = car_data.drop('engine-type',axis = 1)
car_data = car_data.join(one_hot2)
print(car_data.head())
#---------------------------------------Encoding Num of Cylinders------
car_data['num-of-cylinders'] = car_data['num-of-cylinders'].replace("three", 3)
car_data['num-of-cylinders'] = car_data['num-of-cylinders'].replace("four", 4)
car_data['num-of-cylinders'] = car_data['num-of-cylinders'].replace("five", 5)
car_data['num-of-cylinders'] = car_data['num-of-cylinders'].replace("six", 6)
car_data['num-of-cylinders'] = car_data['num-of-cylinders'].replace("eight", 8)
car_data['num-of-cylinders'] = car_data['num-of-cylinders'].replace("twelve", 12)
print(car_data.head())
#---------------------------------------Encoding fuel-system------
one_hot2 = pd.get_dummies(car_data['fuel-system'])
car_data = car_data.drop('fuel-system',axis = 1)
car_data = car_data.join(one_hot2)
print(car_data.head())
#output_excel = car_data.to_excel("D:/Data Science with Shubham/Datasets/regression/out_excel.xlsx")
#out_corr_mat = car_data.corr()
#out_corr_mat.to_excel("D:/Data Science with Shubham/Datasets/regression/out_excel_corr.xlsx")
car_data_final = car_data
#------------------IMPLEMENT LINEAR REGRESSION MODEL-----------------------------------------------------

from sklearn import linear_model

from sklearn.model_selection import train_test_split

print(car_data_final.columns)

y = car_data_final.iloc[:,13]
print(y.head())
X = car_data_final.drop(['price'], axis = 1)

#---------------------------------------------------------------------------------------------------------
#Model Building

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)

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
#-----------------------------------------------------------------------------------------------------
print("Data:",X_test.iloc[4:6,:])

lst = [[88.6,168.8,64.1,48.8,2548,4,130,3.47,2.68,9,111,5000,27,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,1,0,0],[88.6,168.8,64.1,48.8,2548,4,130,3.47,2.68,9,111,5000,27,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,1,0,0]]
predicted_score = regr.predict(lst)
print("predicted_score:",predicted_score[0])
