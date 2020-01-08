import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
import os
np.random.seed(42)
Name="heart.csv"
path="/home/samsung-ub/Desktop/Pyhton/Datasets/"
dataNameandPath = os.path.join("r",path, Name)
df = pd.read_csv("../Datasets/"+Name)

# %%
features=df.columns[0:-1]
X_o=df[features]
y_org=df.target
print(X_o.shape)
print(y_org.shape)

# %%
### Change categorical variables with dummy varibles columns
# 'cp', 'thal' and 'slope' 
df['cp'] = df['cp'].astype('category')
df['thal'] = df['thal'].astype('category')
df['slope'] = df['slope'].astype('category')

a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")

frm = [X_o,a, b, c]
X_wCatVar = pd.concat(frm, axis = 1)

X_org = X_wCatVar.drop(columns = ['cp', 'thal', 'slope']).values

#%%
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#%%
m = len(X_org)
shuffled_indices = np.random.permutation(m)
X = X_org[shuffled_indices]
y  = y_org [shuffled_indices]

X_train, X_val, y_train, y_val = train_test_split(X, y,random_state=42)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_tr_poly = poly_features.fit_transform(X_train)
X_val_poly = poly_features.fit_transform(X_val)

scaler = StandardScaler()
X_tr_poly_std = scaler.fit_transform(X_tr_poly.astype(np.float64))
X_val_poly_std = scaler.fit_transform(X_val_poly.astype(np.float64))

#%%
data={'LogReg': [0], 
     'SGD_log_None': [0],
     'SGD_log_l2': [0],
     'SGD_log_l1': [0],
     'SGD_hinge_l2': [0],
     'SGD_hinge_l1': [0]}

ind =['train_MSE', 'val_MSE', 'train_Cls_Error', 'val_Cls_Error']

Results = pd.DataFrame(data, index =ind) 

#%% Log regression
log_reg = LogisticRegression()
log_reg.fit(X_tr_poly_std, y_train)

y_train_predict = log_reg.predict(X_tr_poly_std)
y_val_predict = log_reg.predict(X_val_poly_std)

train_MSE=(mean_squared_error(y_train, y_train_predict))
val_MSE=(mean_squared_error(y_val, y_val_predict))
train_Cls_Error=1 - np.mean(y_train == y_train_predict)
val_Cls_Error=1 - np.mean(y_val == y_val_predict)

Results.loc[:,"LogReg"]=train_MSE, val_MSE, train_Cls_Error,val_Cls_Error

#%%
from sklearn.linear_model import SGDClassifier
sgd_log = SGDClassifier(loss="log", penalty=None, random_state=42)
sgd_log.fit(X_tr_poly_std, y_train)

y_train_predict = sgd_log.predict(X_tr_poly_std)
y_val_predict = sgd_log.predict(X_val_poly_std)

train_MSE=(mean_squared_error(y_train, y_train_predict))
val_MSE=(mean_squared_error(y_val, y_val_predict))
train_Cls_Error=1 - np.mean(y_train == y_train_predict)
val_Cls_Error=1 - np.mean(y_val == y_val_predict)

Results.loc[:,"SGD_log_None"]=train_MSE, val_MSE, train_Cls_Error,val_Cls_Error
#%%
from sklearn.linear_model import SGDClassifier
sgd_log = SGDClassifier(loss="log", penalty="l2", random_state=42)
sgd_log.fit(X_tr_poly_std, y_train)

y_train_predict = sgd_log.predict(X_tr_poly_std)
y_val_predict = sgd_log.predict(X_val_poly_std)

train_MSE=(mean_squared_error(y_train, y_train_predict))
val_MSE=(mean_squared_error(y_val, y_val_predict))
train_Cls_Error=1 - np.mean(y_train == y_train_predict)
val_Cls_Error=1 - np.mean(y_val == y_val_predict)

Results.loc[:,"SGD_log_l2"]=train_MSE, val_MSE, train_Cls_Error,val_Cls_Error

#%%
from sklearn.linear_model import SGDClassifier
sgd_log = SGDClassifier(loss="log", penalty="l1", random_state=42)
sgd_log.fit(X_tr_poly_std, y_train)

y_train_predict = sgd_log.predict(X_tr_poly_std)
y_val_predict = sgd_log.predict(X_val_poly_std)

train_MSE=(mean_squared_error(y_train, y_train_predict))
val_MSE=(mean_squared_error(y_val, y_val_predict))
train_Cls_Error=1 - np.mean(y_train == y_train_predict)
val_Cls_Error=1 - np.mean(y_val == y_val_predict)

Results.loc[:,"SGD_log_l1"]=train_MSE, val_MSE, train_Cls_Error,val_Cls_Error

#%% loss="hinge", penalty="l2"
sgd_hinge = SGDClassifier(loss="hinge", penalty="l2", random_state=42)
sgd_hinge.fit(X_tr_poly_std, y_train)

y_train_predict = sgd_hinge.predict(X_tr_poly_std)
y_val_predict = sgd_hinge.predict(X_val_poly_std)

train_MSE=(mean_squared_error(y_train, y_train_predict))
val_MSE=(mean_squared_error(y_val, y_val_predict))
train_Cls_Error=1 - np.mean(y_train == y_train_predict)
val_Cls_Error=1 - np.mean(y_val == y_val_predict)

Results.loc[:,"SGD_hinge_l2"]=train_MSE, val_MSE, train_Cls_Error,val_Cls_Error

#%% loss="hinge", penalty="l1"
sgd_hinge = SGDClassifier(loss="hinge", penalty="l1", random_state=42)
sgd_hinge.fit(X_tr_poly_std, y_train)

y_train_predict = sgd_hinge.predict(X_tr_poly_std)
y_val_predict = sgd_hinge.predict(X_val_poly_std)

train_MSE=(mean_squared_error(y_train, y_train_predict))
val_MSE=(mean_squared_error(y_val, y_val_predict))
train_Cls_Error=1 - np.mean(y_train == y_train_predict)
val_Cls_Error=1 - np.mean(y_val == y_val_predict)

Results.loc[:,"SGD_hinge_l1"]=train_MSE, val_MSE, train_Cls_Error,val_Cls_Error


#%%
txt=Results.columns[0:-1]
for i,t in enumerate(txt):
    y = Results.iloc[0,i]
    y1 = Results.iloc[1,i]
    x = i+1
    plt.scatter(x, y, marker='x', color='red',label="Train MSE")
    plt.scatter(x, y1, marker='.', color='blue',label="Val MSE")
#    plt.text(x, (y+y1)/2, t, fontsize=9, rotation=90)
plt.title("Train/Val MSE(Classification Error)")
plt.show()
