#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:





# In[2]:


dataset=pd.read_csv('https://raw.githubusercontent.com/geniusai-research/interns_task/main/taskdata.csv')
 


# In[ ]:





# In[3]:


dataset.head()


# In[ ]:





# In[4]:


dataset.tail()


# In[ ]:





# In[5]:


dataset.shape


# In[ ]:





# In[6]:


dataset.describe()


# In[ ]:





# In[7]:


dataset.columns


# In[ ]:





# In[8]:


dataset.nunique()


# In[ ]:





# In[ ]:


#Cleaning the data


# In[9]:


dataset.isnull().sum() #cheacking for null values .Since we are dropping user_id and account_id therefore we do not have to worry about null values


# In[ ]:





# In[10]:


df2=dataset.drop(['user_id','account_id'],axis=1) #dropping the redundant data


# In[ ]:





# In[11]:


df2.head()


# In[ ]:





# In[ ]:


#Relationship analysis


# In[12]:


sns.pairplot(df2)


# In[ ]:





# In[13]:


sns.relplot(x='avg_email_replies',y='avg_call_replies',hue='target',data=df2)


# In[ ]:





# In[14]:


sns.distplot(df2['max_return_days'],bins=5)


# In[ ]:





# In[15]:


sns.distplot(df2['average_return_days'],bins=5)


# In[ ]:





# In[16]:


# Importing the dataset
dataset = pd.read_csv('C:\\Users\\Beast\\Downloads\\P14-Part10-Model-Selection_Boosting\\P14-Part10-Model-Selection_Boosting\\Section 43 - XGBoost\\Python\\Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)


# In[ ]:





# In[17]:


# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)


# In[ ]:





# In[18]:


# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)


# In[ ]:





# In[19]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:





# In[20]:


# Training XGBoost on the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


# In[ ]:





# In[21]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[ ]:





# In[22]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[ ]:





# In[23]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




