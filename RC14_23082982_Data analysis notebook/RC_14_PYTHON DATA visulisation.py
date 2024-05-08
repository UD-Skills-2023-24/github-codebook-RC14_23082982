#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt


# In[101]:


#read the dataset
df = pd.read_csv('1traffic-flow-borough python.csv', index_col=0)  
df.head()


# In[115]:


print(df.tail())


# In[116]:


df.head()# Get the first 5 rows


# In[117]:


df.index = df.index.astype(str)
df.columns = df.columns.astype(str)


# In[127]:


plt.figure(figsize=(30, 10))
for column in df.columns:
    plt.plot(df.index, df[column], label=column)

plt.xlabel('Region')
plt.ylabel(' Traffic')
plt.title(' Traffic Flow Over the Years')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()


# In[130]:


df_transposed = df.T

# Draw a line chart
plt.figure(figsize=(10, 6))

for column in df_transposed.columns:
    plt.plot(df_transposed.index, df_transposed[column], label=column)

plt.xlabel('Year')
plt.ylabel('Traffic')
plt.title('Traffic Flow Over the Years')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  
plt.grid(True)
plt.show()


# # heatmap

# In[135]:


#read the dataset
df = pd.read_csv('K-menas 2.csv')
df.head()#read the dataset


# In[136]:


for column in df.columns:
    if df[column].dtype != 'object':  # Ignore non-numeric columns
        min_value = df[column].min()
        df[column].fillna(min_value, inplace=True)


# In[137]:


df.columns


# In[138]:


data=df[[ 'Wellbeing score', 'BuildingHeights', 'Integration', 'Flow',
       'PTAL', 'Casualty', 'Fuel', 'Supermarket', 'Choice']]


# In[139]:


data.isnull().values.any()#check if there is any null value


# In[140]:


data.corr()


# In[142]:


import seaborn as sns
plt.figure(figsize=(12,8))
heatmap = sns.heatmap(data.corr(), cmap='viridis')

plt.title('Heatmap of data correlation')


# In[145]:


#Jointplot
sns.jointplot(x='PTAL',y='Fuel', data=df, kind="hex")


# In[ ]:




