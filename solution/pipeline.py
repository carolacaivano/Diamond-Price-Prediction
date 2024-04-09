import numpy as np
import pandas as pd
import seaborn as sns
import sys
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from lineartree import LinearTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

#load the dataset 
path=sys.argv[1]
print(path)
data=pd.read_csv(path)
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
def remove_outliers_numeric(df,feature,delta=1.5):
    '''This function takes as input DataFrame and the feature name to consider and remove outliers 
    based on IQR and a delta (by default delta = 1.5)'''
    Q1_value = df[feature].quantile(0.25)
    Q3_value = df[feature].quantile(0.75)
    IQR_value = Q3_value - Q1_value    #IQR is interquartile range.
    lower_bound=Q1_value - delta * IQR_value
    upper_bound= Q3_value + delta *IQR_value
    print("lower bound for ", feature, "is {}".format(lower_bound))
    print("upper bound for ", feature, "is {}".format(upper_bound))
    filter_value = (df[feature] >= lower_bound) & (df[feature] <= upper_bound)
    df = df.loc[filter_value,:]
    return df
data=remove_outliers_numeric(data,'depth',delta=1.5)
data=remove_outliers_numeric(data,'table',delta=1.5)
data=remove_outliers_numeric(data,'x',delta=1.5)
data=remove_outliers_numeric(data,'y',delta=1.5)
data=remove_outliers_numeric(data,'z',delta=1.5)
data=remove_outliers_numeric(data,'carat',delta=1.5)
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
X=data_X
y=data_y
r2_linear=[]
mae_linear=[]
r2_regr=[]
mae_regr=[]
#training and evaluation of linear regression
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
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
#save the model for Linear Regression
lr_pkl_file = r'../diamond_linear_regression.pkl' 
with open(lr_pkl_file, 'wb') as file:  
    pickle.dump(lr, file)

#training and evaluation of Linear Tree Regressor
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
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
#save the model for Linear Tree Regressor
regr_pkl_file =r'../diamond_linear_tree_regression.pkl'
with open(regr_pkl_file, 'wb') as file:  
    pickle.dump(regr, file)

#save the scores 
data = {
    'Linear Regression': {
        'R2 Scores': r2_linear,
        'MAE Scores': mae_linear,
        'Mean R2 Score': np.mean(r2_linear),
        'Mean MAE Score': np.mean(mae_linear)
    },
    'Linear Tree Regression': {
        'R2 Scores': r2_regr,
        'MAE Scores': mae_regr,
        'Mean R2 Score': np.mean(r2_regr),
        'Mean MAE Score': np.mean(mae_regr)
    }
}
scores_df = pd.DataFrame(data)
scores_df.to_csv(r'../metrics_scores.csv')