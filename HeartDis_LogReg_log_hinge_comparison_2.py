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
from sklearn.linear_model import SGDClassifier
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

#Results = pd.DataFrame(data, index =ind) 

#%%
def runDifferentModels(X,y,X_val,y_val,los,pen,Resutls):
    for i in range(len(los)):
        if i==0:
            model = LogisticRegression()
            model.fit(X, y)
        else:
            model = SGDClassifier(loss=los[i], penalty=pen[i], random_state=42)
            model.fit(X, y)
        y_train_predict = model.predict(X)
        y_val_predict = model.predict(X_val)
        
        train_MSE=(mean_squared_error(y, y_train_predict))
        val_MSE=(mean_squared_error(y_val, y_val_predict))
        train_Cls_Error=1 - np.mean(y == y_train_predict)
        val_Cls_Error=1 - np.mean(y_val == y_val_predict)
        Results.iloc[:,i]=train_MSE, val_MSE, train_Cls_Error,val_Cls_Error
        print("i=",i)
    return Results

#%%
Results = pd.DataFrame(data, index =ind) 
los=["logReg","log","log","log","hinge","hinge"]
pen=["logReg","None", "l2", "l1", "l2", "l1"]
X=X_tr_poly_std
y=y_train
X_val=X_val_poly_std
y_val=y_val

#%%
Results=runDifferentModels(X,y,X_val,y_val,los,pen,Results) 
Results       

#%% Train models with increasing number of samples to see Train/Val error rates
d1={'train_MSE': [0], 
     'val_MSE': [0],
     'train_Cls_Error': [0],
     'val_Cls_Error': [0]}
LogReg = pd.DataFrame(d1)
SGD_log_None = pd.DataFrame(d1)
SGD_log_l2 = pd.DataFrame(d1)
SGD_log_l1 = pd.DataFrame(d1)
SGD_hinge_l2 = pd.DataFrame(d1)
SGD_hinge_l1 = pd.DataFrame(d1)

#%% 
Results = pd.DataFrame(data, index =ind) 
X_val=X_val_poly_std
y_val=y_val
i=0
start=50
for m in range(start, len(X_train)):#len(X_train)
    X=X_tr_poly_std[:m]
    y=y_train[:m]

    Results=runDifferentModels(X,y,X_val,y_val,los,pen,Results)
    LogReg.loc[i] =  Results.iloc[0:4,0]
    SGD_log_None.loc[i] = Results.iloc[0:4,1]
    SGD_log_l2.loc[i] = Results.iloc[0:4,2]
    SGD_log_l1.loc[i] = Results.iloc[0:4,3]
    SGD_hinge_l2.loc[i] = Results.iloc[0:4,4]
    SGD_hinge_l1.loc[i] = Results.iloc[0:4,5]
    i+=1
#    print("i=",i)

#%%
fig, axs = plt.subplots(2, 3)
fig.suptitle('RMSE: Train vs Valiation', fontsize=16)
ii=np.arange(start+1,len(X_train)+1)

plt.subplot(2, 3, 1)
plt.plot(ii,np.sqrt(LogReg.train_MSE), "b-+", linewidth=2, label="train")
plt.plot(ii,np.sqrt(LogReg.val_MSE), "r-", linewidth=3, label="val")
plt.title("Logistic Regression", fontsize=14)  

plt.subplot(2, 3, 2)
plt.plot(ii,np.sqrt(SGD_log_None.train_MSE), "b-+", linewidth=2, label="train")
plt.plot(ii,np.sqrt(SGD_log_None.val_MSE), "r-", linewidth=3, label="val")
plt.title("SGD Loss=Log No Penalty", fontsize=14)  

plt.subplot(2, 3, 3)
plt.plot(ii,np.sqrt(SGD_log_l2.train_MSE), "b-+", linewidth=2, label="train")
plt.plot(ii,np.sqrt(SGD_log_l2.val_MSE), "r-", linewidth=3, label="val")
plt.title("SGD Loss=Log Penalty=l2", fontsize=14)

plt.subplot(2, 3, 4)
plt.plot(ii,np.sqrt(SGD_log_l1.train_MSE), "b-+", linewidth=2, label="train")
plt.plot(ii,np.sqrt(SGD_log_l1.val_MSE), "r-", linewidth=3, label="val")
plt.title("SGD Loss=Log Penalty=l1", fontsize=14)

plt.subplot(2, 3, 5)
plt.plot(ii,np.sqrt(SGD_hinge_l2.train_MSE), "b-+", linewidth=2, label="train")
plt.plot(ii,np.sqrt(SGD_hinge_l2.val_MSE), "r-", linewidth=3, label="val")
plt.title("SGD Loss=hinge Penalty=l2", fontsize=14)

plt.subplot(2, 3, 6)
plt.plot(ii,np.sqrt(SGD_hinge_l1.train_MSE), "b-+", linewidth=2, label="train")
plt.plot(ii,np.sqrt(SGD_hinge_l1.val_MSE), "r-", linewidth=3, label="val")
plt.title("SGD Loss=hinge Penalty=l1", fontsize=14)

