# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 01:57:01 2019

@author: Mandar Joshi
"""


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
import plotly.express as px
import streamlit as st

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
train.head()

st.title('House Prices')

if st.checkbox('Show dataframe'):
    st.write(train)
    
    
fig1 = px.histogram(x = train['SalePrice'],color = train['SalePrice'])


# Plot!
st.plotly_chart(fig1)



    
st.subheader('Scatter plot')    

col2 = st.selectbox('show plot of Sales price against', train.columns[0:81])

# create figure using plotly express
fig = px.scatter(x = train['SalePrice'],y = train[col2],color=train[col2])


# Plot!
st.plotly_chart(fig)

miss_col_train = [col for col in train.columns if train[col].isnull().any()]
miss_col_test = [col for col in test.columns if test[col].isnull().any()]

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

for col in miss_col_train:
    train[col]=train[col].fillna(train[col].mode()[0])
    
for col in miss_col_test:
    test[col]=test[col].fillna(test[col].mode()[0])

from sklearn.preprocessing import LabelEncoder  ###for encode a categorical values

    
le = LabelEncoder()
for col in train.select_dtypes(include=['object']):
    train[col]=le.fit_transform(train[col])
for col in test.select_dtypes(include=['object']):
    test[col]=le.fit_transform(test[col])
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']
train.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1)
test.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1)


train["SalePrice"] = np.log1p(train["SalePrice"])
from scipy.stats import norm, skew 
numeric_feats = train.dtypes[train.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    train[feat] = boxcox1p(train[feat], lam)
    
    
from scipy.stats import norm, skew 
numeric_feats = test.dtypes[test.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = test[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    test[feat] = boxcox1p(test[feat], lam)
    
    
X=train.drop(["SalePrice"],axis=1)
y=train["SalePrice"]
print(X)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X , y ,test_size = 0.1,random_state = 0)

alg = ['Lasso', 'XGB']
classifier = st.selectbox('Which algorithm?', alg)
if classifier=='Lasso':
    from sklearn.linear_model import Lasso

    lasso = Lasso(alpha=0.0004)

    model = lasso

    ### prediction
    model.fit(X, y)
    pred_lasso = model.predict(x_test)
    acc = model.score(x_test, y_test)
  
    st.write('Accuracy: ', acc)


elif classifier=='XGB':
    from xgboost import XGBRegressor
    
    model = XGBRegressor()
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
    model.fit(X, y, early_stopping_rounds=5, 
                 eval_set=[(x_test, y_test)], verbose=False)
    acc = model.score(x_test, y_test)
    pred_lasso = model.predict(x_test)  
    st.write('Accuracy: ', acc)
