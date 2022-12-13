#!/usr/bin/env python
# coding: utf-8

# # Data Science: Logistic Regression 
# #### By: Javier Orduz
# <!--
# <img
# src="https://jaorduz.github.io/images/Javier%20Orduz_01.jpg" width="50" align="center">
# -->
# 
# [license-badge]: https://img.shields.io/badge/License-CC-orange
# [license]: https://creativecommons.org/licenses/by-nc-sa/3.0/deed.en
# 
# [![CC License][license-badge]][license]  [![DS](https://img.shields.io/badge/downloads-DS-green)](https://github.com/Earlham-College/DS_Fall_2022)  [![Github](https://img.shields.io/badge/jaorduz-repos-blue)](https://github.com/jaorduz/)  ![Follow @jaorduc](https://img.shields.io/twitter/follow/jaorduc?label=follow&logo=twitter&logoColor=lkj&style=plastic)
# 

# We load the different packages that we will use.

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


get_ipython().run_line_magic('matplotlib', 'inline')


# To build the model using ```LogReg```

# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# Evaluation and metrics

# In[3]:


from sklearn.metrics import jaccard_score


# <h1>Table of contents</h1>
# 
# <div class="alert  alert-block alert-info" style="margin-top: 20px">
#     <ol>
#         <li><a href="#logReg">Logistic Regression</a></li>
# <!---         <ol>
#              <li><a href="#reData">Reading</a></li>
#              <li><a href="#exData">Exploration</a></li>
#          </ol>
#          --->
#         <li><a href="#unData">Data</a></li>
#     </ol>
# </div>
# <br>
# <hr>

# <h2 id="reData">Logistic Regression</h2>

# 

# Previously, we normalize our data, what does it mean? We are going to create the datasets.
# Before revising the next cells, you should check our previous notebook.

# In[4]:


churn_df = pd.read_csv("ChurnData.csv")
churn_df.shape


# In[5]:


X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])


# In[6]:


X = preprocessing.StandardScaler().fit(X).transform(X)


# # Training the model

# ## Split the data set

# In[7]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)


# In[8]:


print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# # Model

# ```LogisticRegression``` is a class with ```liblinear``` solver. ```C``` is a __float__ value. It is the inverse of regularization strength; must be a positive float. Like in SVM, smaller values specify stronger regularization.

# In[9]:



LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# # Predictions

# We use test set.

# In[10]:


yhat = LR.predict(X_test)


# And we obtain the probability of class 0, $P(Y=0\mid X),$ and probability of class 1,
# $P(Y=1\mid X)$

# In[11]:


yhat_prob = LR.predict_proba(X_test)
yhat_prob


# # Exercises
# 1. Build a Logistic regression model for the same dataset, but use a different solver 1. Explain the regularization techniques and how it is implemented in this logistic regression notebook.
# 1. Explain the Jaccard index, confussion matrix, and f1-score.
# 1. Submmit your report in Moodle. Template https://www.overleaf.com/read/xqcnnnrsspcp
# 

# In[12]:


LR = LogisticRegression(C=0.01, solver='sag').fit(X_train,y_train)
LR


# ## Versions

# In[41]:


from platform import python_version
print("python version: ", python_version())
get_ipython().system('pip3 freeze | grep qiskit')


# # References

# [0] data https://tinyurl.com/2m3vr2xp
# 
# [1] numpy https://numpy.org/
# 
# [2] scipy https://docs.scipy.org/
# 
# [3] matplotlib https://matplotlib.org/
# 
# [4] matplotlib.cm https://matplotlib.org/stable/api/cm_api.html
# 
# [5] matplotlib.pyplot https://matplotlib.org/stable/api/pyplot_summary.html
# 
# [6] pandas https://pandas.pydata.org/docs/
# 
# [7] seaborn https://seaborn.pydata.org/
# 
# [8] Jaccard https://tinyurl.com/27bboh2u
# 
# [9] IBM course. Author: Saeed Aghabzorgi. IBM lab skills. Watson Studio.
# 
# 
