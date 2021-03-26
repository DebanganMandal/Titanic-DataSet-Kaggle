#!/usr/bin/env python
# coding: utf-8

# In[175]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import missingno
import seaborn as sns
import math, time, random, datetime
import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool, cv
import warnings
warnings.filterwarnings('ignore')


# In[111]:


train=pd.read_csv("train.csv")


# In[112]:


train.head()


# In[113]:


train.info()


# In[114]:


train.describe()


# In[115]:


gensub=pd.read_csv("gender_submission.csv")


# In[116]:


gensub.head()


# In[117]:


train.isnull().sum()


# In[118]:


missingno.matrix(train, figsize = (30,10))
#We cannot use cabin


# In[119]:


plt.figure(figsize=(20,15))
sns.heatmap(data=train.corr().abs(), annot=True)


# In[120]:


fig=plt.figure(figsize=(20,2))
sns.countplot(y=train.Survived)
print(train.Survived.value_counts())


# In[121]:


df_bin=pd.DataFrame()
df_con=pd.DataFrame()


# In[122]:


df_bin.head()


# In[123]:


sns.distplot(train.Pclass)
train.Pclass.isnull().sum()


# In[124]:


plt.figure(figsize=(20,2))
sns.countplot(y=train.Pclass)
print(train.Pclass.value_counts())


# In[125]:


df_bin['Pclass']=train.Pclass
df_con['Pclass']=train.Pclass


# In[126]:


train.Name.value_counts()


# In[127]:


plt.figure(figsize=(20,2))
sns.countplot(y=train.Sex)
print(train.Sex.value_counts())


# In[128]:


train.Sex.isnull().sum()


# In[129]:


train['Sex']=np.where(train['Sex']==1, 'female', 'male')


# In[130]:


plt.figure(figsize=(10, 10))
sns.countplot(x=train.Survived, hue=train.Sex)


# In[131]:


train.Age.isnull().sum()


# In[132]:


train.SibSp.isnull().sum()


# In[133]:


train.SibSp.value_counts()


# In[134]:


plt.figure(figsize=(10,10))
sns.countplot(x=train.SibSp, hue=train.Survived)


# In[135]:


plt.figure(figsize=(10,10))
sns.countplot(x=train.SibSp)


# In[136]:


train.Parch.isnull().sum()


# In[137]:


train.Parch.value_counts()


# In[138]:


plt.figure(figsize=(10,10))
sns.countplot(x=train.Parch)


# In[139]:


plt.figure(figsize=(10,10))
sns.countplot(x=train.Parch, hue=train.Survived)


# In[140]:


train.Ticket.isnull().sum()


# In[141]:


train.Ticket.value_counts()


# In[142]:


train.Fare.isnull().sum()


# In[143]:


train.Fare.value_counts()


# In[144]:


train.Cabin.isnull().sum()


# In[145]:


train.drop(["Cabin"], axis=1, inplace=True)


# In[146]:


train


# In[147]:


train.Embarked.isnull().sum()


# In[148]:


train.Embarked.value_counts()


# In[149]:


plt.figure(figsize=(10,10))
sns.countplot(x=train.Survived, hue=train.Embarked)


# In[154]:


df_bin = pd.DataFrame(zip(train.Survived, train.Pclass, train.Sex, train.SibSp, train.Parch, train.Fare, train.Embarked))
d = {0:'Survived',1:'Pclass',2:'Sex',3:'SibSp',4:'Parch',5:'Fare',6:'Embarked'}
df_bin.rename(columns=d, inplace=True)
df_bin['Sex']=np.where(df_bin['Sex']=="female",1,0)
# Remove Embarked rows which are missing values
df_bin = df_bin.dropna(subset=['Embarked'])
print(len(df_con))
df_bin


# In[156]:


df_con = pd.DataFrame(zip(train.Survived, train.Pclass, train.Sex, train.SibSp, train.Parch, train.Fare, train.Embarked))
d = {0:'Survived',1:'Pclass',2:'Sex',3:'SibSp',4:'Parch',5:'Fare',6:'Embarked'}
df_con.rename(columns=d, inplace=True)
df_con = df_con.dropna(subset=['Embarked'])
print(len(df_con))
df_con


# In[157]:


# One-hot encode binned variables
one_hot_cols = df_bin.columns.tolist()
one_hot_cols.remove('Survived')
df_bin_enc = pd.get_dummies(df_bin, columns=one_hot_cols)

df_bin_enc.head()


# In[159]:


# One hot encode the categorical columns
df_embarked_one_hot = pd.get_dummies(df_con['Embarked'], 
                                     prefix='embarked')

df_sex_one_hot = pd.get_dummies(df_con['Sex'], 
                                prefix='sex')

df_plcass_one_hot = pd.get_dummies(df_con['Pclass'], 
                                   prefix='pclass')


# In[160]:


# Combine the one hot encoded columns with df_con_enc
df_con_enc = pd.concat([df_con, 
                        df_embarked_one_hot, 
                        df_sex_one_hot, 
                        df_plcass_one_hot], axis=1)

# Drop the original categorical columns (because now they've been one hot encoded)
df_con_enc = df_con_enc.drop(['Pclass', 'Sex', 'Embarked'], axis=1)


# In[161]:


df_con_enc.head(20)


# In[162]:


selected_df = df_con_enc


# In[163]:


X_train = selected_df.drop('Survived', axis=1)
y_train = selected_df.Survived


# In[164]:


X_train.head()


# In[165]:


X_train.shape


# In[166]:


y_train.shape


# In[178]:


# Function that runs the requested algorithm and returns the accuracy metrics
def fit_ml_algo(algo, X_train, y_train, cv):
    
    # One Pass
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)
    
    # Cross Validation 
    train_pred = model_selection.cross_val_predict(algo, 
                                                  X_train, 
                                                  y_train, 
                                                  cv=cv, 
                                                  n_jobs = -1)
    # Cross-validation accuracy metric
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    
    return train_pred, acc, acc_cv


# In[179]:


start_time = time.time()
train_pred_log, acc_log, acc_cv_log = fit_ml_algo(LogisticRegression(), 
                                                               X_train, 
                                                               y_train, 
                                                                    10)
log_time = (time.time() - start_time)
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))


# In[180]:


# k-Nearest Neighbours
start_time = time.time()
train_pred_knn, acc_knn, acc_cv_knn = fit_ml_algo(KNeighborsClassifier(), 
                                                  X_train, 
                                                  y_train, 
                                                  10)
knn_time = (time.time() - start_time)
print("Accuracy: %s" % acc_knn)
print("Accuracy CV 10-Fold: %s" % acc_cv_knn)
print("Running Time: %s" % datetime.timedelta(seconds=knn_time))


# In[181]:


# Gaussian Naive Bayes
start_time = time.time()
train_pred_gaussian, acc_gaussian, acc_cv_gaussian = fit_ml_algo(GaussianNB(), 
                                                                      X_train, 
                                                                      y_train, 
                                                                           10)
gaussian_time = (time.time() - start_time)
print("Accuracy: %s" % acc_gaussian)
print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)
print("Running Time: %s" % datetime.timedelta(seconds=gaussian_time))


# In[182]:


# Linear SVC
start_time = time.time()
train_pred_svc, acc_linear_svc, acc_cv_linear_svc = fit_ml_algo(LinearSVC(),
                                                                X_train, 
                                                                y_train, 
                                                                10)
linear_svc_time = (time.time() - start_time)
print("Accuracy: %s" % acc_linear_svc)
print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)
print("Running Time: %s" % datetime.timedelta(seconds=linear_svc_time))


# In[183]:


# Decision Tree Classifier
start_time = time.time()
train_pred_dt, acc_dt, acc_cv_dt = fit_ml_algo(DecisionTreeClassifier(), 
                                                                X_train, 
                                                                y_train,
                                                                10)
dt_time = (time.time() - start_time)
print("Accuracy: %s" % acc_dt)
print("Accuracy CV 10-Fold: %s" % acc_cv_dt)
print("Running Time: %s" % datetime.timedelta(seconds=dt_time))


# In[184]:


models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 
               'Linear SVC', 'Decision Tree'],
    'Score': [
        acc_knn, 
        acc_log,  
        acc_gaussian,  
        acc_linear_svc, 
        acc_dt,
    ]})
print("---Reuglar Accuracy Scores---")
models.sort_values(by='Score', ascending=False)


# In[ ]:




