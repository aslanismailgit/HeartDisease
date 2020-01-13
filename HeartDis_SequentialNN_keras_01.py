#%% import modules
import tensorflow as tf
from tensorflow import keras


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import time
import os
#%%
np.random.seed(42)
Name="heart.csv"
path="/home/samsung-ub/Desktop/Python/Datasets/"
dataNameandPath = os.path.join("r",path, Name)
df = pd.read_csv(dataNameandPath)


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
m = len(X_org)
shuffled_indices = np.random.permutation(m)
X = X_org[shuffled_indices]
y  = y_org [shuffled_indices]

X_train, X_valid, y_train, y_valid = train_test_split(X, y,random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#%%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
#%% setup 
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

#%% create a hidden layer model
model = keras.models.Sequential([
    keras.layers.Dense(60, activation="relu",input_shape=X_train.shape[1:]),
    keras.layers.Dense(1, activation="sigmoid")
])

#%% model compile 
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

#%% plot summary table
model.layers
model.summary()

#%%
import graphviz
import pydot

keras.utils.plot_model(model, "Heart Disease Sequential NN.png", show_shapes=True)

#%%
history = model.fit(X_train, y_train, epochs=70,
                    validation_data=(X_valid, y_valid))

#%%
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

#%%
model.evaluate(X_test, y_test)

#%%
def prednew(model,n):
    data={'y_new': [], 
         'y_pred': []}
    X_new = X_test[:n]    
    y_pred = model.predict_classes(X_new)
    y_new = y_test[:n]
    ind =y_new.index.astype("str")
    Acc=pd.DataFrame(data)#,index=ind)
    Acc.loc[:,"y_new"]=y_new.values
    Acc.loc[:,"y_pred"]=y_pred
    Acc.index=ind
    return Acc
#%%
prednew(model,20)


