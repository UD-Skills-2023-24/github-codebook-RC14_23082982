#!/usr/bin/env python
# coding: utf-8

# ## Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt


# ## The Data

# In[15]:


#read the dataset
df = pd.read_csv('K-menas 2.csv')
df.head()


# In[16]:


df.shape


# In[17]:


for column in df.columns:
    if df[column].dtype != 'object':  # Ignore non-numeric columns
        min_value = df[column].min()
        df[column].fillna(min_value, inplace=True)


# In[18]:


df.head()# Get the first 5 rows


# In[19]:


df.info()


# In[22]:


df.columns


# ## Clean DATA

# In[23]:


data=df[[ 'Wellbeing score', 'BuildingHeights', 'Integration', 'Flow',
       'PTAL', 'Casualty', 'Fuel', 'Supermarket', 'Choice']]


# In[24]:


data.shape


# In[32]:


data.head()


# In[33]:


data.isnull().values.any()#check if there is any null value


# In[34]:


datalocation=df[['id']]


# In[35]:


datalocation.head()


# In[36]:


import os

# 2. Normalize the data
normalized_df = data.copy()
for column in data.columns:
    if normalized_df[column].dtype != 'object':  
        max_value = normalized_df[column].max()
        normalized_df[column] = normalized_df[column] / max_value

codebook_path = os.path.abspath('codebook.csv')
save_dir = os.path.dirname(codebook_path)

# Create the full path for the new CSV file
normalized_csv_file = os.path.join(save_dir, 'normalized_data.csv')

# 3. The normalized data is saved back to a CSV file
normalized_df.to_csv(normalized_csv_file, index=False)


# In[37]:


#Re-read the file, noting that the data read below belongs to the normalized_data file
df = pd.read_csv('normalized_data.csv')
df.head()


# In[38]:


df.info()


# In[39]:


df.columns


# In[42]:


#Group(If it is necessary to group the data.All the datas are counted as one group here)
data1=df[['Wellbeing score', 'BuildingHeights', 'Integration', 'Flow', 'PTAL',
       'Casualty', 'Fuel', 'Supermarket', 'Choice']]


# ## Visualisation

# In[43]:


data1.corr()


# In[48]:


import seaborn as sns
plt.figure(figsize=(12,8))
heatmap = sns.heatmap(data1.corr(), cmap='viridis')#heatmap

plt.title('Heatmap of data correlation')


# ## DIVIDE DATA FOR ANALYSIS

# In[49]:


#the data should be doublicated in order to run the analysis independantly
data2=data1.copy()
data3=data1.copy()

# Alternative you can divide the code into 2 files. DO NOT USE PCA data in KMEANS or the opposite


# ## K-Means Clustering

# ### A. Define the number of clusters (Elbow method)

# In[50]:


#Fit data and calculate sum of squares(wss)
wss=[]
from sklearn.cluster import KMeans

for i in range (1,21):    
    kmeans=KMeans(n_clusters=i,init="k-means++")
    kmeans.fit(data1)
    wss.append(kmeans.inertia_)


# In[51]:


#Visualisation of the k values in order to define the fittest
plt.figure(figsize=(10,8))
plt.plot(range(1,21),wss,marker="o", color="#dc00ff")
plt.xlabel("Number of K value")
plt.ylabel("WSS")
plt.title("Sum of squares vs number of clusters")
#plt.savefig("Elbow.png", dpi=300,transparent=True)
plt.show()


# ### B.Clustering the data

# In[52]:


#import library
from sklearn.cluster import KMeans


# In[53]:


#pick number of clusters
kmeans = KMeans(n_clusters=5)


# In[54]:


#fit the data
kmeans.fit(data1)


# In[55]:


#find the cluster centers
kmeans.cluster_centers_


# In[56]:


#identify cluster labels
kmeans.labels_


# In[57]:


#identify length of cluster labels
len(kmeans.labels_)


# In[58]:


#Add cluster column in dataset
data1['CLUSTERS'] = kmeans.fit_predict(data1)


# In[59]:


data1.head()


# ### C.Visualise your Kmeans result

# In[63]:


#Scatterplot of clustered data
sns.scatterplot(x=data1['Supermarket'], y=data1['Choice'], hue= kmeans.labels_, palette='rainbow')
plt.title('K-means Clustering')
plt.legend(loc=0,bbox_to_anchor=(1.2,0.5),title="Clusters")
plt.show()


# ## Principal Component Analysis

# ### A. Preprocessing

# In[64]:


from sklearn.preprocessing import StandardScaler


# In[65]:


scaler = StandardScaler()
scaler.fit(data2)


# In[66]:


scaled_data = scaler.transform(data2)


# ### B. Running PCA

# In[67]:


from sklearn.decomposition import PCA


# In[68]:


pca = PCA(n_components=3)


# In[69]:


pca.fit(scaled_data)


# In[70]:


x_pca = pca.transform(scaled_data)


# In[71]:


scaled_data.shape #15 dimensions


# In[72]:


x_pca.shape #3 dimensions


# ### C.Interpretation of results

# In[73]:


pca.components_


# In[74]:


df_comp = pd.DataFrame(pca.components_,columns=data2.columns)


# In[75]:


plt.figure(figsize=(12,6))

sns.heatmap(df_comp,cmap='vlag')
plt.show()


# In[76]:


cmap = sns.diverging_palette(180, 295, s=100, l=50,
                                  n=10, center= "dark", as_cmap = True)

#Documentation: https://seaborn.pydata.org/generated/seaborn.diverging_palette.html


# In[77]:


plt.figure(figsize=(16,12))
sns.heatmap(df_comp,cmap= cmap, annot=True,fmt = ".1f")


# In[78]:


df_comp.head()


# In[79]:


df=pd.DataFrame(x_pca, columns = ['0','1','2'])


# In[80]:


df.head()


# In[81]:


datalocation.head()


# In[82]:


df_prefinal=pd.concat([df,datalocation],axis=1)


# In[83]:


df_prefinal.head()


# In[84]:


df_final=pd.concat([df_prefinal, data1],axis=1)


# In[85]:


df_final.head()


# In[86]:


df_final.to_csv('FinalResult9.csv', encoding='utf-8', index=False)


# In[ ]:




