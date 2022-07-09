#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Loading dataset
data1 = pd.read_csv("C:/Users/gayat/Downloads/DATA SETS/train.csv/train.csv")
data1


# In[3]:


#Loading test dataset
data2 = pd.read_csv("C:/Users/gayat/Downloads/DATA SETS/test.csv/test.csv")
data2


# In[4]:


data = data1.append(data2)
data.head()


# In[5]:


data.shape


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


data.info()


# In[9]:


#encoding
from sklearn.preprocessing import LabelEncoder
label_en = LabelEncoder()
data['Gender']  = label_en.fit_transform(data['Gender'])
data['Age'] = label_en.fit_transform(data['Age'])
data['City_Category'] = label_en.fit_transform(data['City_Category'])
data['Stay_In_Current_City_Years'] = label_en.fit_transform(data['Stay_In_Current_City_Years'])


# In[10]:


data


# In[11]:


data['Age'].unique()


# In[12]:


data.isnull().sum()


# In[18]:


data['Product_Category_2'].value_counts()
data['Product_Category_3'].value_counts()


# In[22]:


#imputing
data['Product_Category_2'] = data['Product_Category_2'].fillna(data['Product_Category_2'].mode()[0])
data['Product_Category_3'] = data['Product_Category_3'].fillna(data['Product_Category_3'].mode()[0])


# In[23]:


data


# In[28]:


data['Purchase'].mode()[0]


# In[29]:


data['Purchase'].value_counts()


# In[30]:


data['Purchase'] = data['Purchase'].fillna(data['Purchase'].mode()[0])


# In[31]:


data


# In[32]:


data.info()


# In[35]:


#coverting object to int
#data['Product_ID'] = data['Product_ID'].astype(int)


# In[37]:


sns.pairplot(data)


# In[38]:


sns.barplot('Age', 'Purchase', hue='Gender', data=data)


# In[42]:


sns.barplot('Occupation','Purchase', hue='Gender', data=data)


# In[43]:


sns.barplot('Product_Category_1','Purchase', hue='Gender', data=data)


# In[44]:


sns.barplot('Product_Category_2','Purchase', hue='Gender', data=data)


# In[45]:


sns.barplot('Product_Category_3','Purchase', hue='Gender', data=data)


# In[65]:


#
X = data.iloc[:, -10 :]


# In[66]:


X.head()


# In[67]:


y = data.iloc[:,-1 :]


# In[68]:


y


# In[69]:


#trainig model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# In[70]:


data.drop(['User_ID','Product_ID'], axis=1)


# In[71]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:





# In[ ]:




