#!/usr/bin/env python
# coding: utf-8

# # Data Science: Logistic Regression 
# #### By: Javier Orduz
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
get_ipython().run_line_magic('matplotlib', 'inline')


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

# We use LogReg to know the most probable class for any particular data point. In other words, this is a good tool to estimate the class of a data point.

# LogReg is a variation of Liner regression. We use LR when when the observed dependent variable, <b>y</b>, is categorical. It produces a formula that predicts the probability of the class label as a function of the independent variables.
# 
# Logistic regression fits a special s-shaped curve by taking the linear regression function and transforming the numeric estimate into a probability with the following function, which is called the <b>sigmoid function,</b> $\sigma$:
# 
# $$
# h_\theta(x) = \sigma({\theta^TX}) =  \frac {e^{(\theta_0 + \theta_1  x_1 + \theta_2  x_2 +...)}}{1 + e^{(\theta_0 + \theta_1  x_1 + \theta_2  x_2 +\cdots)}}
# $$
# 
# Or:
# $$\text{Probability Of a Class
# }_1 =  P(Y=1|X) = \sigma({\theta^TX}) = \frac{e^{\theta^TX}}{1+e^{\theta^TX}}
# $$

# In this equation, ${\theta^TX}$ is the regression result (the sum of the variables weighted by the coefficients), $e$ is the exponential function and $\sigma(\theta^TX)$ is the sigmoid or [logistic function](http://en.wikipedia.org/wiki/Logistic_function?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01), also called logistic curve. It is a common "S" shape (sigmoid curve).
# 
# So, briefly, Logistic Regression passes the input through the logistic/sigmoid but then treats the result as a probability:
# 
# <img
# src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/images/mod_ID_24_final.png" width="400" align="center">
# 
# The objective of the **Logistic Regression** algorithm, is to find the best parameters $\theta$, for $h\_\theta(x)$ = $\sigma({\theta^TX})$, in such a way that the model best predicts the class of each case.
# 

# 
# <h2 id="unData">About the dataset</h2>
# 
# 
# We will use a telecommunications dataset for predicting customer churn. This is a historical customer dataset where each row represents one customer. The data is relatively easy to understand, and you may uncover insights you can use immediately. Typically it is less expensive to keep customers than acquire new ones, so the focus of this analysis is to predict the customers who will stay with the company. 
# 
# This data set provides information to help you predict what behavior will help you to retain customers. You can analyze all relevant customer data and develop focused customer retention programs.
# 
# The dataset includes information about:
# 
# *   Customers who left within the last month – the column is called Churn
# *   Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# *   Customer account information – how long they had been a customer, contract, payment method, paperless billing, monthly charges, and total charges
# *   Demographic info about customers – gender, age range, and if they have partners and dependents
# 

# In[3]:


churn_df = pd.read_csv("ChurnData.csv")
churn_df.head()


# ### Load the Telco Churn data
# 
# EC Churn is a hypothetical data file that concerns a telecommunications company's efforts to reduce turnover in its customer base. Each case corresponds to a separate customer and it records various demographic and service usage information. 
# 

# ## Data pre-processing
# 
# Let's select some features for the modeling. Also, we change the target data type to be an integer, as it is a requirement by the skitlearn algorithm:
# 

# In[4]:


churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head()


# Let's define X and y for our dataset

# In[5]:


X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]


# In[6]:


y = np.asarray(churn_df['churn'])
y [0:5]


# We normalize the dataset

# In[7]:


from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Exercises
# 1. Find the number of rows and columns are in this dataset in total.
# 1. Fit a linear regression model and interpret the coefficients. Do the coefficients make intuitive sense to you? Explore the Yelp website to see if you detect similar trends.
# 1. Submmit your report in Moodle. Template https://www.overleaf.com/read/xqcnnnrsspcp
# 

# In[11]:


print(len(churn_df))
print(len(churn_df.columns))


# In[14]:


from sklearn.linear_model import LinearRegression

X = churn_df.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
Y = churn_df.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# ## Versions

# In[ ]:


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
# [8] IBM course. Author: Saeed Aghabzorgi. IBM lab skills. Watson Studio.
# 

# In[ ]:




