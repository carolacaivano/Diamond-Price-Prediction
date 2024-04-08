import numpy as np
import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from lineartree import LinearTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

#load the dataset 
data=pd.read_csv('sys.argv[1]')
num_data = data.shape[0]
print("The dataset consists of {} data points.".format(num_data))

#we delete the diamonds that are not 3D and that have a price that is equal or less than 0
data = data.drop(data[data["price"]==-1].index)
data = data.drop(data[data["x"]==0].index)
data = data.drop(data[data["y"]==0].index)
data = data.drop(data[data["z"]==0].index)
num_data = data.shape[0]
print("The dataset consists of {} after the data cleaning.".format(num_data))

#identifying and removing outliers
features=['depth','table','x','y','z','carat']
lower_bounds=[58.75,51.5,1.92,1.9450000000000016,1.1700000000000004,-0.5850000000000001]
upper_bounds=[64.75,63.5,9.28,9.264999999999999,5.729999999999999,2.015]
for feature, lb_feature, ub_feature in zip(features, lower_bounds, upper_bounds):
    filter_value = (data[feature] >= lb_feature) & (data[feature] <= ub_feature)
    data = data.loc[filter_value, :]
num_data = data.shape[0]
print("The dataset consists of {} after removing the outliers.".format(num_data))

#Label Encoding Categorical Feature
# Define the order for each categorical variable (cut, color, clarity) from worst to best
cut_order= {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
clarity_order={'I1':0,'SI2':1, 'SI1':2, 'VS2':3, 'VS1':4, 'VVS2':5, 'VVS1':6, 'IF':7}
color_order={'J':0, 'I':1, 'H':2, 'G':3, 'F':4, 'E':5,'D':6}
# Label encoding
data['encoded_cut'] = data['cut'].map(cut_order)
data['encoded_clarity'] = data['clarity'].map(clarity_order)
data['encoded_color'] = data['color'].map(color_order)
#We drop the columns cut, clarity and color
data=data.drop(['cut'],axis=1)
data=data.drop(['clarity'],axis=1)
data=data.drop(['color'],axis=1)
print(" We succesfully encoded the categorical features")

#ML model selection 
scaler = StandardScaler()
data_X=data[["carat","depth", "table", "x", "y", "z","encoded_cut", "encoded_clarity",	"encoded_color"]]
data_y=data[["price"]]
#we define the two models 
lr = LinearRegression()
regr = LinearTreeRegressor(base_estimator=LinearRegression())

kf = KFold(n_splits=5)
X=np.array(data_X)
y=np.array(data_y)
r2_linear=[]
mae_linear=[]
r2_regr=[]
mae_regr=[]
#training and evaluation of linear regression
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train= scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    r2=r2_score(y_test, y_pred)
    mae=mean_absolute_error(y_test,y_pred)
    r2_linear.append(r2)
    mae_linear.append(mae)
print("r2 score:", r2_linear)
print("mae score:", mae_linear)
print("R² Score: {:.2f}".format(np.mean(r2_linear)))
print("MAE Score: {:.2f}".format(np.mean(mae_linear)))

#training and evaluation of Linear Tree Regressor
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train= scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)
    regr.fit(X_train,y_train)
    y_pred = regr.predict(X_test)
    r2=r2_score(y_test, y_pred)
    mae=mean_absolute_error(y_test,y_pred)
    r2_regr.append(r2)
    mae_regr.append(mae)
print(r2_regr)
print(mae_regr)
print("R² Score: {:.2f}".format(np.mean(r2_regr)))
print("MAE Score: {:.2f}".format(np.mean(mae_regr)))
